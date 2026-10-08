import torch

from conftest import EOS, PAD, make_tiny_lm
from dpo_rep.modeling import (PolicyWithValue, RewardModel, last_token_index, lm_forward,
                              sequence_logprobs, token_logprobs)
from dpo_rep.sampling import left_pad, right_pad


def manual_completion_logprob(model, prompt, completion):
    """Token-by-token log p(completion | prompt) on an unpadded sequence."""
    ids = torch.tensor([prompt + completion])
    logits = model(input_ids=ids).logits[0].log_softmax(-1)
    return sum(logits[len(prompt) + i - 1, tok].item() for i, tok in enumerate(completion))


def test_sequence_logprobs_matches_manual_with_right_padding(tiny_lm):
    pairs = [([3, 4], [5, 6, 7]), ([8, 9, 10, 11], [12]), ([1], [2, 3, 4, 5, 6])]
    seqs = [p + c for p, c in pairs]
    ids, attn = right_pad(seqs, PAD, torch.device("cpu"))
    comp = torch.zeros_like(ids)
    for i, (p, c) in enumerate(pairs):
        comp[i, len(p):len(p) + len(c)] = 1
    got = sequence_logprobs(tiny_lm, ids, attn, comp)
    want = torch.tensor([manual_completion_logprob(tiny_lm, p, c) for p, c in pairs])
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_left_and_right_padding_give_same_logprobs(tiny_lm):
    """Position ids come from the attention mask, so padding side must not matter."""
    seqs = [[3, 4, 5, 6], [7, 8], [9, 10, 11, 12, 13, 14]]
    dev = torch.device("cpu")
    lid, lm = left_pad(seqs, PAD, dev)
    rid, rm = right_pad(seqs, PAD, dev)
    llp = token_logprobs(lm_forward(tiny_lm, lid, lm).logits, lid)
    rlp = token_logprobs(lm_forward(tiny_lm, rid, rm).logits, rid)
    for i, s in enumerate(seqs):
        n = len(s) - 1
        torch.testing.assert_close(llp[i, -n:], rlp[i, :n], atol=1e-4, rtol=1e-4)


def test_last_token_index():
    attn = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    assert last_token_index(attn).tolist() == [2, 4, 2]


def test_reward_model_pooling_independent_of_padding_side(tiny_lm):
    rm = RewardModel(tiny_lm.transformer).eval()
    seqs = [[3, 4, 5], [6, 7, 8, 9, 10]]
    dev = torch.device("cpu")
    left = rm(*left_pad(seqs, PAD, dev))
    right = rm(*right_pad(seqs, PAD, dev))
    single = torch.stack([rm(torch.tensor([s]), torch.ones(1, len(s), dtype=torch.long))[0] for s in seqs])
    torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(left, single, atol=1e-5, rtol=1e-5)


def test_reward_model_normalization_and_save_load(tmp_path, tiny_lm):
    rm = RewardModel(tiny_lm.transformer).eval()
    rm.score_mean.fill_(0.5)
    ids, attn = right_pad([[1, 2, 3]], PAD, torch.device("cpu"))
    raw = rm(ids, attn)
    torch.testing.assert_close(rm(ids, attn, normalize=True), raw - 0.5)
    rm.save(tmp_path)
    rm2 = RewardModel.load(tmp_path, torch.device("cpu")).eval()
    torch.testing.assert_close(rm2(ids, attn, normalize=True), raw - 0.5, atol=1e-5, rtol=1e-5)


def test_policy_with_value_shapes_and_logits_unchanged(tiny_lm):
    pv = PolicyWithValue(make_tiny_lm(0)).eval()
    ids, attn = right_pad([[1, 2, 3, 4], [5, 6]], PAD, torch.device("cpu"))
    logits, values = pv(ids, attn)
    assert values.shape == ids.shape
    assert torch.all(values == 0)  # zero-initialized head
    torch.testing.assert_close(logits, lm_forward(tiny_lm, ids, attn).logits)
