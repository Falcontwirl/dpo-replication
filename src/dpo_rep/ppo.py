"""PPO for RLHF, from scratch (Ziegler et al. 2019 / Stiennon et al. 2020 style).

One PPO step:
  1. Rollout: sample completions from the current policy for a batch of prompts.
  2. Score each completion with the reward function (learned RM or ground-truth classifier); optionally
     whiten the scores over the batch ("normalized rewards", paper Sec 6.1).
  3. Per-token reward = -kl_coef * (log pi - log pi_ref) on every completion token, plus the score on the
     last completion token.
  4. GAE advantages (whitened) and returns from the value head.
  5. ppo_epochs passes of clipped policy + clipped value loss over shuffled minibatches.
  6. Adaptive KL controller (Ziegler et al.) nudges kl_coef so the sequence KL approaches target_kl.

Layout: all rollouts in a step are re-padded to a common (P, T) = (max prefix tokens, max new tokens), so
completion token t is always at column P+t and its log-prob/value come from column P+t-1.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from dpo_rep.losses import gae, masked_mean, masked_whiten, ppo_policy_loss, ppo_value_loss
from dpo_rep.modeling import PolicyWithValue, lm_forward, token_logprobs
from dpo_rep.sampling import Samples, left_pad, right_pad, sample
from dpo_rep.utils import Timer, autocast_ctx


class AdaptiveKLController:
    """Ziegler et al. 2019: kl_coef *= 1 + clip(KL/target - 1, -0.2, 0.2) * n_steps / horizon."""

    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int) -> None:
        err = float(np.clip(current_kl / self.target - 1.0, -0.2, 0.2))
        self.value *= 1.0 + err * n_steps / self.horizon


def repad(samples: list[Samples], pad_id: int, P: int, T: int) -> Samples:
    """Concatenate sample batches into one with fixed prompt length P (left pad) and completion length T."""
    prompts = [p for s in samples for p in s.prompt_lists()]
    comps = [c for s in samples for c in s.completion_lists()]
    device = samples[0].prompt_ids.device
    pid, pm = left_pad(prompts, pad_id, device)
    if pid.shape[1] < P:
        extra = P - pid.shape[1]
        pid = torch.cat([torch.full((pid.shape[0], extra), pad_id, device=device, dtype=pid.dtype), pid], 1)
        pm = torch.cat([torch.zeros((pm.shape[0], extra), device=device, dtype=pm.dtype), pm], 1)
    cid, cm = right_pad(comps, pad_id, device, length=T)
    return Samples(pid, pm, cid, cm)


def _completion_logprobs_values(policy: PolicyWithValue, ids, attn, P: int):
    logits, values = policy(ids, attn)
    lp = token_logprobs(logits, ids)[:, P - 1:]
    T = lp.shape[1]
    return lp, values[:, P - 1:P - 1 + T]


def ppo_step(policy: PolicyWithValue, ref: torch.nn.Module, reward_fn: Callable[[Samples], torch.Tensor],
             prompts: list[list[int]], c, gen_cfg, kl_ctl: AdaptiveKLController, opt: torch.optim.Optimizer,
             tokenizer, device: torch.device, precision: str, timer: Timer, max_prompt_len: int,
             generator: torch.Generator | None = None) -> dict:
    pad_id, eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    amp = lambda: autocast_ctx(device, precision)  # noqa: E731

    # 1. Rollout
    policy.eval()
    with timer.section("rollout"), amp():
        chunks = [sample(policy.lm, prompts[i:i + c.gen_batch_size], gen_cfg.max_new_tokens, eos_id, pad_id,
                         gen_cfg.temperature, gen_cfg.top_k, gen_cfg.top_p, generator=generator)
                  for i in range(0, len(prompts), c.gen_batch_size)]
    s = repad(chunks, pad_id, max_prompt_len, gen_cfg.max_new_tokens)
    P = s.prompt_len
    ids, attn = s.full()
    mask = s.completion_mask.float()
    N = ids.shape[0]

    # 2. Scores
    with timer.section("score"):
        scores = reward_fn(s).to(device).float()

    # 3. Old log-probs, values and reference log-probs
    with timer.section("forward"), torch.no_grad(), amp():
        old_lp, old_v, ref_lp = [], [], []
        for i in range(0, N, c.micro_batch_size):
            lp, v = _completion_logprobs_values(policy, ids[i:i + c.micro_batch_size], attn[i:i + c.micro_batch_size], P)
            old_lp.append(lp)
            old_v.append(v)
            rl = token_logprobs(lm_forward(ref, ids[i:i + c.micro_batch_size], attn[i:i + c.micro_batch_size]).logits,
                                ids[i:i + c.micro_batch_size])[:, P - 1:]
            ref_lp.append(rl)
        old_lp, old_v, ref_lp = torch.cat(old_lp), torch.cat(old_v), torch.cat(ref_lp)

    # 4. Rewards and advantages
    kl = (old_lp - ref_lp) * mask
    score_used = (scores - scores.mean()) / (scores.std() + 1e-8) if c.whiten_scores else scores
    rewards = -kl_ctl.value * kl
    last = mask.sum(1).long() - 1
    rewards[torch.arange(N, device=device), last] += score_used
    adv, returns = gae(rewards, old_v, mask, c.gamma, c.lam)
    adv = masked_whiten(adv, mask)

    # 5. PPO epochs
    policy.train()
    stats = {"pg_loss": [], "vf_loss": [], "clipfrac": [], "approx_kl": [], "vf_clipfrac": []}
    with timer.section("update"):
        for _ in range(c.ppo_epochs):
            perm = torch.randperm(N, device=device)
            for m in range(0, N, c.minibatch_size):
                mb = perm[m:m + c.minibatch_size]
                for u in range(0, len(mb), c.micro_batch_size):
                    idx = mb[u:u + c.micro_batch_size]
                    with amp():
                        lp, v = _completion_logprobs_values(policy, ids[idx], attn[idx], P)
                    pg, st1 = ppo_policy_loss(lp, old_lp[idx], adv[idx], mask[idx], c.clip_range)
                    vf, st2 = ppo_value_loss(v, old_v[idx], returns[idx], mask[idx], c.clip_range_value)
                    ((pg + c.vf_coef * vf) * len(idx) / len(mb)).backward()
                    stats["pg_loss"].append(pg.item())
                    stats["vf_loss"].append(vf.item())
                    for k, val in {**st1, **st2}.items():
                        stats[k].append(val)
                if c.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), c.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)

    # 6. KL controller
    seq_kl = kl.sum(1).mean().item()
    kl_coef_used = kl_ctl.value
    kl_ctl.update(seq_kl, N)

    return {"rollout_score_mean": scores.mean().item(), "rollout_kl": seq_kl, "kl_coef": kl_coef_used,
            "rollout_len": mask.sum(1).mean().item(), "tokens": int(attn.sum().item()),
            **{k: float(np.mean(v)) for k, v in stats.items()}}
