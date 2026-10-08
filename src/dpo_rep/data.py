"""IMDb loading, prefix sampling, SFT batches and preference pairs.

Preference data is stored as token ids (not text) so that training sees exactly the tokens that were
sampled and scored; re-tokenizing decoded text can change token boundaries.
"""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from dpo_rep.sampling import right_pad

_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_WS = re.compile(r"\s+")


def clean_review(text: str) -> str:
    """IMDb reviews contain literal <br /> tags; replace them with spaces."""
    return _WS.sub(" ", _BR.sub(" ", text)).strip()


def load_imdb(split: str) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb", split=split)
    return [clean_review(t) for t in ds["text"]]


def sample_prefixes(texts: list[str], tokenizer, n: int, min_tokens: int, max_tokens: int,
                    rng: np.random.Generator) -> list[list[int]]:
    """Take the first L tokens of n distinct random reviews, L ~ Uniform{min_tokens..max_tokens}."""
    idx = rng.choice(len(texts), size=n, replace=n > len(texts))
    lengths = rng.integers(min_tokens, max_tokens + 1, size=n)
    out = []
    for i, L in zip(idx, lengths):
        ids = tokenizer(texts[i][:400])["input_ids"]  # first 400 chars always cover >8 tokens' worth
        out.append(ids[:int(L)])
    return out


def load_or_make_eval_prompts(cfg, tokenizer) -> list[list[int]]:
    """Fixed eval prompt set for cfg.eval.split (IMDb test for reported results; the disjoint 'unsupervised'
    split for PPO lr tuning), cached so every run evaluates on identical prompts."""
    path = Path(cfg.paths.data_dir) / f"eval_prompts_{cfg.eval.split}.json"
    if path.exists():
        return json.loads(path.read_text())
    rng = np.random.default_rng(cfg.eval.seed)
    prompts = sample_prefixes(load_imdb(cfg.eval.split), tokenizer, cfg.eval.n_prompts,
                              cfg.prefix.min_tokens, cfg.prefix.max_tokens, rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prompts))
    return prompts


def tokenize_reviews(texts: list[str], tokenizer, max_len: int) -> list[list[int]]:
    """Reviews + EOS, truncated from the right to max_len tokens."""
    eos = tokenizer.eos_token_id
    return [(tokenizer(t)["input_ids"] + [eos])[:max_len] for t in texts]


def sft_batch(seqs: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return right_pad(seqs, pad_id, device)


def make_pairs(completions: list[list[int]], scores: list[float]) -> tuple[list[tuple[int, int]], int]:
    """All C(n,2) pairs, ordered (winner, loser) by ground-truth score. Exact ties are dropped (U4).
    Returns (pairs as index tuples, number of ties dropped)."""
    pairs, ties = [], 0
    for i, j in combinations(range(len(completions)), 2):
        if scores[i] > scores[j]:
            pairs.append((i, j))
        elif scores[j] > scores[i]:
            pairs.append((j, i))
        else:
            ties += 1
    return pairs, ties


def pair_batch(examples: list[dict], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack chosen sequences (rows 0..B-1) then rejected (rows B..2B-1), right-padded.

    Returns input_ids, attention_mask, completion_mask, each (2B, L).
    """
    seqs, comp_lens, prompt_lens = [], [], []
    for key in ("chosen_ids", "rejected_ids"):
        for ex in examples:
            seqs.append(ex["prompt_ids"] + ex[key])
            prompt_lens.append(len(ex["prompt_ids"]))
            comp_lens.append(len(ex[key]))
    ids, attn = right_pad(seqs, pad_id, device)
    comp = torch.zeros_like(attn)
    for r, (p, c) in enumerate(zip(prompt_lens, comp_lens)):
        comp[r, p:p + c] = 1
    return ids, attn, comp
