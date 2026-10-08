"""Stage 5: PPO baselines (paper Sec 6.1).

  --set ppo.reward=rm   PPO with the learned reward model (paper "PPO")
  --set ppo.reward=gt   PPO on the ground-truth sentiment classifier (paper "PPO-GT (our impl.)")

Policy and reference are initialized from the SFT model; the value head shares the policy trunk.
Evaluated (true reward + KL, same protocol as DPO) at step 0 and every eval_every PPO steps.

Output: <runs_dir>/<name>/{metrics.jsonl, model/}
"""

from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from dpo_rep.config import parse_args
from dpo_rep.data import load_imdb, load_or_make_eval_prompts, sample_prefixes
from dpo_rep.evaluate import evaluate
from dpo_rep.gt_reward import SentimentReward
from dpo_rep.modeling import PolicyWithValue, RewardModel, load_lm, load_tokenizer
from dpo_rep.ppo import AdaptiveKLController, ppo_step
from dpo_rep.sampling import decode_texts, make_generator
from dpo_rep.utils import JsonlLogger, Timer, autocast_ctx, get_device, gpu_name, make_run_dir, set_seed

TRAIN_SECTIONS = ("rollout", "score", "forward", "update")


def main():
    cfg, args = parse_args(__doc__, extra=[("--sft", {"default": None, "help": "SFT model dir"}),
                                           ("--rm", {"default": None, "help": "reward model dir"})])
    c = cfg.ppo
    if c.reward not in ("rm", "gt"):
        raise ValueError("ppo.reward must be 'rm' or 'gt'")
    method = "ppo" if c.reward == "rm" else "ppo_gt"
    name = args.name or f"{method}_kl{c.target_kl:g}"
    runs = Path(cfg.paths.runs_dir)
    sft_path = args.sft or str(runs / "sft" / "model")
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    run_dir = make_run_dir(cfg, name)
    log = JsonlLogger(run_dir / "metrics.jsonl")
    timer = Timer(device)

    tok = load_tokenizer(sft_path)
    policy = PolicyWithValue(load_lm(sft_path, device, dropout=False))
    ref = load_lm(sft_path, device, dropout=False).eval().requires_grad_(False)
    gt = SentimentReward(cfg.models.gt_reward, device, cfg.prefs.score_batch_size)
    eval_prompts = load_or_make_eval_prompts(cfg, tok)

    if c.reward == "gt":
        def reward_fn(s):
            return gt(decode_texts(tok, s))
    else:
        rm = RewardModel.load(args.rm or runs / "rm" / "model", device).eval().requires_grad_(False)

        @torch.no_grad()
        def reward_fn(s):
            ids, attn = s.full()
            out = []
            for i in range(0, ids.shape[0], c.micro_batch_size * 2):
                with autocast_ctx(device, cfg.precision):
                    out.append(rm(ids[i:i + c.micro_batch_size * 2], attn[i:i + c.micro_batch_size * 2], normalize=True))
            return torch.cat(out)

    opt = torch.optim.AdamW([{"params": policy.lm.parameters(), "lr": c.lr},
                             {"params": policy.value_head.parameters(), "lr": c.value_head_lr}], weight_decay=0.0)
    kl_ctl = AdaptiveKLController(c.init_kl_coef, c.target_kl, c.kl_horizon)
    train_texts = load_imdb("train")
    rng = np.random.default_rng(cfg.seed)
    gen = make_generator(device, cfg.seed)
    print(f"{method} target_kl={c.target_kl}: {c.total_ppo_steps} PPO steps x {c.rollout_batch_size} rollouts, device={device}")

    meta = {"method": method, "target_kl": c.target_kl, "run": name, "gpu_name": gpu_name(device)}
    examples_seen = tokens = 0
    window: list[dict] = []

    def do_eval(step):
        with timer.section("eval"):
            m = evaluate(policy.lm, ref, gt, tok, eval_prompts, cfg, device)
        train_stats = {f"train_{k}": float(np.mean([w[k] for w in window])) for k in window[0]} if window else {}
        window.clear()
        t_train = sum(timer.get(k) for k in TRAIN_SECTIONS)
        row = {**meta, "step": step, "examples_seen": examples_seen, "tokens": tokens, "kl_coef": kl_ctl.value,
               "t_train": t_train, **{f"t_{k}": timer.get(k) for k in TRAIN_SECTIONS},
               "t_eval": timer.get("eval"), "t_total": timer.wall(), **m, **train_stats}
        log.log(row)
        print({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in row.items()})

    do_eval(0)
    for step in range(1, c.total_ppo_steps + 1):
        prompts = sample_prefixes(train_texts, tok, c.rollout_batch_size, cfg.prefix.min_tokens,
                                  cfg.prefix.max_tokens, rng)
        stats = ppo_step(policy, ref, reward_fn, prompts, c, cfg.generation, kl_ctl, opt, tok, device,
                         cfg.precision, timer, cfg.prefix.max_tokens, generator=gen)
        examples_seen += len(prompts)
        tokens += stats["tokens"]
        window.append(stats)
        if step % c.eval_every == 0 or step == c.total_ppo_steps:
            do_eval(step)

    policy.lm.save_pretrained(run_dir / "model")
    tok.save_pretrained(run_dir / "model")
    torch.save(policy.value_head.state_dict(), run_dir / "model" / "value_head.pt")
    (run_dir / "done").write_text(f"steps={c.total_ppo_steps}\n")


if __name__ == "__main__":
    main()
