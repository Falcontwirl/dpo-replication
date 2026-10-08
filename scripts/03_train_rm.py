"""Stage 3: reward model for the PPO baseline (paper App C.1).

Initialized from the base LM (gpt2-large) with a scalar head, trained with the Bradley-Terry loss (Eq 2)
for 3 epochs on the preference pairs; the checkpoint with the best validation accuracy is kept. Rewards
are then shifted so E[r] = 0 over SFT samples (paper Sec 3).

Output: <runs_dir>/<name>/model (RewardModel.save format) and metrics.jsonl.
"""

import math
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from dpo_rep.config import parse_args
from dpo_rep.data import pair_batch
from dpo_rep.losses import bradley_terry_loss
from dpo_rep.modeling import RewardModel, load_tokenizer
from dpo_rep.sampling import right_pad
from dpo_rep.utils import (JsonlLogger, Timer, autocast_ctx, get_device, gpu_name, linear_warmup, make_run_dir,
                           read_jsonl, set_seed)


@torch.no_grad()
def rm_scores(rm, seqs, pad_id, device, precision, bs):
    out = []
    for i in range(0, len(seqs), bs):
        ids, attn = right_pad(seqs[i:i + bs], pad_id, device)
        with autocast_ctx(device, precision):
            out.append(rm(ids, attn).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def validate(rm, val, pad_id, device, precision, bs):
    rm.eval()
    correct, loss_sum = 0, 0.0
    for i in range(0, len(val), bs):
        ex = val[i:i + bs]
        ids, attn, _ = pair_batch(ex, pad_id, device)
        with autocast_ctx(device, precision):
            r = rm(ids, attn)
        rc, rr = r[:len(ex)], r[len(ex):]
        correct += (rc > rr).sum().item()
        loss_sum += bradley_terry_loss(rc, rr).sum().item()
    rm.train()
    return correct / len(val), loss_sum / len(val)


def main():
    cfg, args = parse_args(__doc__)
    name = args.name or "rm"
    c = cfg.rm
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    run_dir = make_run_dir(cfg, name)
    log = JsonlLogger(run_dir / "metrics.jsonl")
    timer = Timer(device)
    data_dir = Path(cfg.paths.data_dir)

    pairs = read_jsonl(data_dir / "prefs.jsonl")
    prefix_ids = sorted({p["prefix_idx"] for p in pairs})
    rng = np.random.default_rng(cfg.seed)
    val_prefixes = set(rng.permutation(prefix_ids)[:max(1, int(len(prefix_ids) * c.val_frac))].tolist())
    train = [p for p in pairs if p["prefix_idx"] not in val_prefixes]
    val = [p for p in pairs if p["prefix_idx"] in val_prefixes]
    print(f"RM: {len(train)} train pairs, {len(val)} val pairs, device={device}")

    tok = load_tokenizer(cfg.models.policy)
    rm = RewardModel.from_lm_name(cfg.models.policy, device)
    rm.train()
    opt = torch.optim.AdamW(rm.parameters(), lr=c.lr)
    sched = linear_warmup(opt, c.warmup_steps)
    steps_per_epoch = math.ceil(len(train) / c.batch_size)
    best_acc, step = -1.0, 0
    model_dir = run_dir / "model"

    def eval_and_maybe_save(epoch):
        nonlocal best_acc
        with timer.section("eval"):
            acc, vloss = validate(rm, val, tok.pad_token_id, device, cfg.precision, c.micro_batch_size * 2)
        improved = acc > best_acc
        if improved:
            best_acc = acc
            rm.save(model_dir)
        row = {"step": step, "epoch": epoch, "val_acc": acc, "val_loss": vloss, "best_val_acc": best_acc,
               "saved": improved, "t_train": timer.get("train"), "t_eval": timer.get("eval"), "t_total": timer.wall(),
               "gpu_name": gpu_name(device)}
        log.log(row)
        print(row)

    for epoch in range(c.epochs):
        order = rng.permutation(len(train))
        for b in range(steps_per_epoch):
            batch = [train[i] for i in order[b * c.batch_size:(b + 1) * c.batch_size]]
            with timer.section("train"):
                loss_total, correct = 0.0, 0
                for j in range(0, len(batch), c.micro_batch_size):
                    ex = batch[j:j + c.micro_batch_size]
                    ids, attn, _ = pair_batch(ex, tok.pad_token_id, device)
                    with autocast_ctx(device, cfg.precision):
                        r = rm(ids, attn)
                    rc, rr = r[:len(ex)], r[len(ex):]
                    loss = bradley_terry_loss(rc, rr).sum() / len(batch)
                    loss.backward()
                    loss_total += loss.item()
                    correct += (rc > rr).sum().item()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                print({"step": step, "loss": round(loss_total, 4), "train_acc": correct / len(batch)})
            if step % c.eval_every == 0:
                eval_and_maybe_save(epoch)
        eval_and_maybe_save(epoch)

    # Reload best checkpoint and set E[r] = 0 over SFT samples (paper Sec 3).
    with timer.section("normalize"):
        rm = RewardModel.load(model_dir, device).eval()
        samples = read_jsonl(data_dir / "samples.jsonl")
        seqs = [s["prompt_ids"] + comp for s in samples for comp in s["completions"]]
        seqs = [seqs[i] for i in rng.permutation(len(seqs))[:c.norm_samples]]
        scores = rm_scores(rm, seqs, tok.pad_token_id, device, cfg.precision, c.micro_batch_size * 2)
        rm.score_mean.fill_(scores.mean().item())
        rm.score_std.fill_(scores.std().item())
        rm.save(model_dir)
    summary = {"best_val_acc": best_acc, "score_mean": rm.score_mean.item(), "score_std": rm.score_std.item(),
               "t_train": timer.get("train"), "t_eval": timer.get("eval"), "t_normalize": timer.get("normalize"),
               "t_cost": timer.get("train") + timer.get("eval") + timer.get("normalize"), "gpu_name": gpu_name(device)}
    log.log({"final": True, **summary})
    (run_dir / "done").write_text(str(summary) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
