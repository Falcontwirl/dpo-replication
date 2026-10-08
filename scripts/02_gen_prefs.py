"""Stage 2: generate the synthetic preference dataset (paper App C.1).

The SFT model samples n_samples completions for each of n_prefixes IMDb-train prefixes; the ground-truth
sentiment classifier scores each (prefix + completion); all C(n,2) pairs per prefix are labeled by score.

Outputs in <data_dir>/:
  samples.jsonl     every prefix with its completions and scores (also used to normalize the RM)
  prefs.jsonl       one preference pair per line (token ids + scores)
  prefs_meta.json   counts and timing (generation cost is needed for the compute-matched comparison)
"""

import json
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from dpo_rep.config import parse_args
from dpo_rep.data import load_imdb, make_pairs, sample_prefixes
from dpo_rep.gt_reward import SentimentReward
from dpo_rep.modeling import load_lm, load_tokenizer
from dpo_rep.sampling import decode_texts, make_generator, sample
from dpo_rep.utils import Timer, autocast_ctx, get_device, gpu_name, set_seed, write_jsonl


def main():
    cfg, args = parse_args(__doc__, extra=[("--sft", {"default": None, "help": "SFT model dir"})])
    sft_path = args.sft or str(Path(cfg.paths.runs_dir) / "sft" / "model")
    c, g = cfg.prefs, cfg.generation
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    timer = Timer(device)
    out_dir = Path(cfg.paths.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer(sft_path)
    model = load_lm(sft_path, device).eval()
    gt = SentimentReward(cfg.models.gt_reward, device, c.score_batch_size)

    prefixes = sample_prefixes(load_imdb("train"), tok, c.n_prefixes, cfg.prefix.min_tokens, cfg.prefix.max_tokens,
                               np.random.default_rng(cfg.seed))
    gen = make_generator(device, cfg.seed)

    samples_rows, pair_rows, n_ties = [], [], 0
    for b in range(0, len(prefixes), c.gen_batch_size):
        batch = prefixes[b:b + c.gen_batch_size]
        prompts = [p for p in batch for _ in range(c.n_samples)]
        with timer.section("gen"), autocast_ctx(device, cfg.precision):
            s = sample(model, prompts, g.max_new_tokens, tok.eos_token_id, tok.pad_token_id,
                       g.temperature, g.top_k, g.top_p, generator=gen)
        with timer.section("score"):
            scores = gt(decode_texts(tok, s)).tolist()
        comps = s.completion_lists()
        for k, prompt in enumerate(batch):
            idx = b + k
            cs = comps[k * c.n_samples:(k + 1) * c.n_samples]
            sc = scores[k * c.n_samples:(k + 1) * c.n_samples]
            samples_rows.append({"prefix_idx": idx, "prompt_ids": prompt, "completions": cs, "scores": sc})
            pairs, ties = make_pairs(cs, sc)
            n_ties += ties
            for w, l in pairs:
                pair_rows.append({"prefix_idx": idx, "prompt_ids": prompt, "chosen_ids": cs[w], "rejected_ids": cs[l],
                                  "chosen_score": sc[w], "rejected_score": sc[l]})
        print(f"{b + len(batch)}/{len(prefixes)} prefixes, {len(pair_rows)} pairs, "
              f"gen {timer.get('gen'):.0f}s score {timer.get('score'):.0f}s")

    write_jsonl(out_dir / "samples.jsonl", samples_rows)
    write_jsonl(out_dir / "prefs.jsonl", pair_rows)
    all_scores = [s for r in samples_rows for s in r["scores"]]
    meta = {"n_prefixes": len(prefixes), "n_samples": c.n_samples, "n_pairs": len(pair_rows), "n_ties_dropped": n_ties,
            "mean_sft_score": float(np.mean(all_scores)), "t_gen": timer.get("gen"), "t_score": timer.get("score"),
            "t_total": timer.get("gen") + timer.get("score"), "gpu_name": gpu_name(device), "sft_path": sft_path}
    (out_dir / "prefs_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
