"""Stage 1: supervised fine-tuning of the base LM on IMDb train reviews (paper App C.1: 1 epoch on a subset).

Output: <runs_dir>/<name>/model (the SFT policy = pi_ref for all later stages) and metrics.jsonl.
"""

import math

import numpy as np
import torch

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from dpo_rep.config import parse_args
from dpo_rep.data import load_imdb, sft_batch, tokenize_reviews
from dpo_rep.modeling import lm_forward, load_lm, load_tokenizer, token_logprobs
from dpo_rep.utils import (JsonlLogger, Timer, autocast_ctx, get_device, gpu_name, linear_warmup, make_run_dir,
                           set_seed)


def main():
    cfg, args = parse_args(__doc__)
    name = args.name or "sft"
    c = cfg.sft
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    run_dir = make_run_dir(cfg, name)
    log = JsonlLogger(run_dir / "metrics.jsonl")
    timer = Timer(device)

    tok = load_tokenizer(cfg.models.policy)
    model = load_lm(cfg.models.policy, device, dropout=True)
    model.train()

    texts = load_imdb("train")
    order = np.random.default_rng(cfg.seed).permutation(len(texts))[:c.n_reviews]
    seqs = tokenize_reviews([texts[i] for i in order], tok, c.max_len)
    print(f"SFT on {len(seqs)} reviews, {sum(map(len, seqs))} tokens, device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    sched = linear_warmup(opt, c.warmup_steps)
    n_steps = c.epochs * math.ceil(len(seqs) / c.batch_size)
    tokens_seen = 0

    for step in range(n_steps):
        start = (step * c.batch_size) % len(seqs)
        batch = seqs[start:start + c.batch_size]
        n_tok = sum(len(s) - 1 for s in batch)
        with timer.section("train"):
            total = 0.0
            for j in range(0, len(batch), c.micro_batch_size):
                ids, attn = sft_batch(batch[j:j + c.micro_batch_size], tok.pad_token_id, device)
                with autocast_ctx(device, cfg.precision):
                    logits = lm_forward(model, ids, attn).logits
                lp = token_logprobs(logits, ids)
                loss = -(lp * attn[:, 1:]).sum() / n_tok
                loss.backward()
                total += loss.item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        tokens_seen += n_tok
        if step % 10 == 0 or step == n_steps - 1:
            row = {"step": step + 1, "loss": total, "lr": sched.get_last_lr()[0], "tokens": tokens_seen,
                   "t_train": timer.get("train"), "t_total": timer.wall(), "gpu_name": gpu_name(device)}
            log.log(row)
            print(row)

    model.save_pretrained(run_dir / "model")
    tok.save_pretrained(run_dir / "model")
    (run_dir / "done").write_text(f"t_train={timer.get('train')}\n")
    print(f"Saved SFT model to {run_dir / 'model'}")


if __name__ == "__main__":
    main()
