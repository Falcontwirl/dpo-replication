import numpy as np
import torch

from dpo_rep.data import clean_review, make_pairs, pair_batch, sample_prefixes


def test_clean_review():
    assert clean_review("Great film.<br /><br />Loved it.  ") == "Great film. Loved it."


def test_make_pairs_all_six_ordered_by_score():
    pairs, ties = make_pairs([[1], [2], [3], [4]], [0.1, 0.9, 0.5, 0.3])
    assert ties == 0 and len(pairs) == 6
    scores = [0.1, 0.9, 0.5, 0.3]
    assert all(scores[w] > scores[l] for w, l in pairs)
    assert {frozenset(p) for p in pairs} == {frozenset(p) for p in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]}


def test_make_pairs_drops_ties():
    pairs, ties = make_pairs([[1], [2], [3], [4]], [0.5, 0.5, 0.2, 0.9])
    assert ties == 1 and len(pairs) == 5


def test_pair_batch_layout():
    exs = [{"prompt_ids": [1, 2], "chosen_ids": [3, 4, 5], "rejected_ids": [6]},
           {"prompt_ids": [7], "chosen_ids": [8], "rejected_ids": [9, 10]}]
    ids, attn, comp = pair_batch(exs, pad_id=0, device=torch.device("cpu"))
    assert ids.shape == (4, 5)
    assert ids[0].tolist() == [1, 2, 3, 4, 5] and comp[0].tolist() == [0, 0, 1, 1, 1]
    assert ids[1].tolist()[:2] == [7, 8] and comp[1].tolist() == [0, 1, 0, 0, 0]
    assert ids[2].tolist()[:3] == [1, 2, 6] and comp[2].tolist() == [0, 0, 1, 0, 0]
    assert ids[3].tolist()[:3] == [7, 9, 10] and comp[3].tolist() == [0, 1, 1, 0, 0]
    assert attn.sum(1).tolist() == [5, 2, 3, 3]


class FakeTok:
    def __call__(self, text):
        return {"input_ids": [ord(c) for c in text]}


def test_sample_prefixes_lengths_and_distinct_reviews():
    texts = [f"{i:03d}" + "abcdefghijk" for i in range(50)]
    prefixes = sample_prefixes(texts, FakeTok(), 50, 2, 8, np.random.default_rng(0))
    assert all(2 <= len(p) <= 8 for p in prefixes)
    assert len({tuple(p[:3]) for p in prefixes if len(p) >= 3}) == sum(len(p) >= 3 for p in prefixes)
