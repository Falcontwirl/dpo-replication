"""Stage 4: DPO on the preference pairs (paper Eq 7, App B defaults: batch 64, RMSprop, lr 1e-6, 150-step warmup).

pi_theta and pi_ref are both initialized from the SFT model. The policy is evaluated (true reward + KL) at
step 0 and every eval_every steps; training runs for a fixed budget with no early stopping.

Output: <runs_dir>/<name>/{metrics.jsonl, model/}
"""

from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from dpo_rep.config import parse_args
from dpo_rep.data import load_or_make_eval_prompts, pair_batch
from dpo_rep.evaluate import evaluate
from dpo_rep.gt_reward import SentimentReward
from dpo_rep.losses import dpo_loss
from dpo_rep.modeling import load_lm, load_tokenizer, sequence_logprobs
from dpo_rep.utils import (JsonlLogger, Timer, autocast_ctx, get_device, gpu_name, linear_warmup, make_run_dir,
                           read_jsonl, set_seed)


def main():
    cfg, args = parse_args(__doc__, extra=[("--sft", {"default": None, "help": "SFT model dir"})])
    c = cfg.dpo
    name = args.name or f"dpo_beta{c.beta}"
    sft_path = args.sft or str(Path(cfg.paths.runs_dir) / "sft" / "model")
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    run_dir = make_run_dir(cfg, name)
    log = JsonlLogger(run_dir / "metrics.jsonl")
    timer = Timer(device)

    tok = load_tokenizer(sft_path)
    policy = load_lm(sft_path, device, dropout=False).train()
    ref = load_lm(sft_path, device, dropout=False).eval().requires_grad_(False)
    gt = SentimentReward(cfg.models.gt_reward, device, cfg.prefs.score_batch_size)
    eval_prompts = load_or_make_eval_prompts(cfg, tok)
    pairs = read_jsonl(Path(cfg.paths.data_dir) / "prefs.jsonl")

    if c.optimizer != "rmsprop":
        raise ValueError("Paper uses RMSprop; change deviations.md before using anything else")
    opt = torch.optim.RMSprop(policy.parameters(), lr=c.lr)
    sched = linear_warmup(opt, c.warmup_steps)
    steps_per_epoch = len(pairs) // c.batch_size
    total_steps = c.epochs * steps_per_epoch
    rng = np.random.default_rng(cfg.seed)
    print(f"DPO beta={c.beta}: {len(pairs)} pairs, {total_steps} steps, device={device}")

    meta = {"method": "dpo", "beta": c.beta, "run": name, "gpu_name": gpu_name(device)}
    examples_seen = tokens = 0  # DPO: preference pairs
    window = {"loss": [], "acc": [], "margin": [], "chosen": [], "rejected": []}

    def do_eval(step):
        with timer.section("eval"):
            m = evaluate(policy, ref, gt, tok, eval_prompts, cfg, device)
        train_stats = {f"train_{k}": float(np.mean(v)) if v else None for k, v in window.items()}
        for v in window.values():
            v.clear()
        row = {**meta, "step": step, "examples_seen": examples_seen, "tokens": tokens, "lr": sched.get_last_lr()[0],
               "t_train": timer.get("train"), "t_eval": timer.get("eval"), "t_total": timer.wall(), **m, **train_stats}
        log.log(row)
        print({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in row.items()})

    do_eval(0)
    step = 0
    for epoch in range(c.epochs):
        order = rng.permutation(len(pairs))
        for b in range(steps_per_epoch):
            batch = [pairs[i] for i in order[b * c.batch_size:(b + 1) * c.batch_size]]
            with timer.section("train"):
                for j in range(0, len(batch), c.micro_batch_size):
                    ex = batch[j:j + c.micro_batch_size]
                    n = len(ex)
                    ids, attn, comp = pair_batch(ex, tok.pad_token_id, device)
                    with autocast_ctx(device, cfg.precision):
                        pi_logps = sequence_logprobs(policy, ids, attn, comp)
                        with torch.no_grad():
                            ref_logps = sequence_logprobs(ref, ids, attn, comp)
                    yw, yl = torch.arange(n, device=device), torch.arange(n, 2 * n, device=device)
                    losses, rewards = dpo_loss(pi_logps, ref_logps, yw, yl, c.beta)
                    (losses.sum() / len(batch)).backward()
                    tokens += int(attn.sum())
                    window["loss"].append(losses.mean().item())
                    window["acc"].append((rewards[:n] > rewards[n:]).float().mean().item())
                    window["margin"].append((rewards[:n] - rewards[n:]).mean().item())
                    window["chosen"].append(rewards[:n].mean().item())
                    window["rejected"].append(rewards[n:].mean().item())
                if c.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), c.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            examples_seen += len(batch)
            if step % c.eval_every == 0 or step == total_steps:
                do_eval(step)

    policy.save_pretrained(run_dir / "model")
    tok.save_pretrained(run_dir / "model")
    (run_dir / "done").write_text(f"steps={step}\n")


if __name__ == "__main__":
    main()
