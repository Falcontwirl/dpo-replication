"""Policy evaluation: mean ground-truth reward and mean sequence-level KL(pi || pi_ref) on fixed test prompts.

Every eval uses the same prompts and the same sampling seed (common random numbers), so differences
between checkpoints are not dominated by sampling noise.
"""

from __future__ import annotations

import math

import torch

from dpo_rep.gt_reward import SentimentReward
from dpo_rep.kl import completion_kl
from dpo_rep.sampling import decode_texts, make_generator, sample
from dpo_rep.utils import autocast_ctx


def _mean_se(x: torch.Tensor) -> tuple[float, float]:
    x = x.double()
    return x.mean().item(), (x.std(unbiased=True) / math.sqrt(len(x))).item() if len(x) > 1 else 0.0


@torch.no_grad()
def evaluate(policy: torch.nn.Module, ref: torch.nn.Module, gt: SentimentReward, tokenizer,
             prompts: list[list[int]], cfg, device: torch.device) -> dict:
    was_training = policy.training
    policy.eval()
    g = cfg.generation
    gen = make_generator(device, cfg.eval.seed)
    rewards, kl_exact, kl_sample, lengths = [], [], [], []
    bs = cfg.eval.batch_size
    for i in range(0, len(prompts), bs):
        with autocast_ctx(device, cfg.precision):
            s = sample(policy, prompts[i:i + bs], g.max_new_tokens, tokenizer.eos_token_id, tokenizer.pad_token_id,
                       g.temperature, g.top_k, g.top_p, generator=gen)
            ke, ks = completion_kl(policy, ref, s)
        rewards.append(gt(decode_texts(tokenizer, s)))
        kl_exact.append(ke.float().cpu())
        kl_sample.append(ks.float().cpu())
        lengths.append(s.completion_mask.sum(1).float().cpu())
    policy.train(was_training)

    r_mean, r_se = _mean_se(torch.cat(rewards))
    k_mean, k_se = _mean_se(torch.cat(kl_exact))
    ks_mean, _ = _mean_se(torch.cat(kl_sample))
    return {"reward_mean": r_mean, "reward_se": r_se, "kl_exact": k_mean, "kl_exact_se": k_se,
            "kl_sample": ks_mean, "completion_len": torch.cat(lengths).mean().item(),
            "reward_per_kl": r_mean / k_mean if k_mean > 1e-8 else None}
