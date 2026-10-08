import torch

from conftest import EOS, PAD
from dpo_rep.kl import completion_kl, per_token_kl
from dpo_rep.modeling import sequence_logprobs
from dpo_rep.sampling import filter_logits, sample

PROMPTS = [[3, 4], [5, 6, 7, 8, 9], [10, 11, 12]]


def greedy_no_cache(model, prompt, n, eos):
    seq = list(prompt)
    out = []
    for _ in range(n):
        tok = model(input_ids=torch.tensor([seq])).logits[0, -1].argmax().item()
        out.append(tok)
        seq.append(tok)
        if tok == eos:
            break
    return out


def test_greedy_with_cache_and_left_padding_matches_naive_loop(tiny_lm):
    s = sample(tiny_lm, PROMPTS, max_new_tokens=10, eos_id=EOS, pad_id=PAD, temperature=0)
    for got, prompt in zip(s.completion_lists(), PROMPTS):
        assert got == greedy_no_cache(tiny_lm, prompt, 10, EOS)
    assert s.prompt_lists() == PROMPTS


def test_completion_mask_stops_after_eos(tiny_lm):
    # 60 rows x 12 tokens over a 50-token vocab: some rows hit EOS early (checked below).
    g = torch.Generator().manual_seed(0)
    s = sample(tiny_lm, PROMPTS * 20, max_new_tokens=12, eos_id=EOS, pad_id=PAD, temperature=1.0, generator=g)
    lengths = s.completion_mask.sum(1)
    assert (lengths < 12).any() and (lengths == 12).any()
    for ids, m in zip(s.completion_ids, s.completion_mask):
        n = int(m.sum())
        assert m[:n].all() and not m[n:].any(), "mask must be a prefix of ones"
        assert (ids[:n - 1] != EOS).all(), "EOS may only appear as the last real token"
        assert (ids[n:] == PAD).all()


def test_sampling_is_seeded(tiny_lm):
    a = sample(tiny_lm, PROMPTS, 8, EOS, PAD, generator=torch.Generator().manual_seed(1))
    b = sample(tiny_lm, PROMPTS, 8, EOS, PAD, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a.completion_ids, b.completion_ids)


def test_filter_logits():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
    assert torch.isinf(filter_logits(logits, top_k=2)).tolist() == [[True, False, False, True]]
    # probs ~ [0.09, 0.67, 0.24, 0.03]: top_p=0.7 keeps the two largest.
    assert torch.isinf(filter_logits(logits, top_p=0.7)).tolist() == [[True, False, False, True]]


def test_kl_zero_for_identical_models(tiny_lm):
    s = sample(tiny_lm, PROMPTS, 8, EOS, PAD, generator=torch.Generator().manual_seed(0))
    exact, samp = completion_kl(tiny_lm, tiny_lm, s)
    assert torch.allclose(exact, torch.zeros_like(exact), atol=1e-6)
    assert torch.allclose(samp, torch.zeros_like(samp), atol=1e-6)


def test_kl_exact_matches_manual_and_sample_matches_seq_logprobs(tiny_lm_pair):
    pol, ref = tiny_lm_pair
    s = sample(pol, PROMPTS, 6, EOS, PAD, generator=torch.Generator().manual_seed(0))
    exact, samp = completion_kl(pol, ref, s)
    assert (exact >= 0).all()

    for i, (p, c) in enumerate(zip(s.prompt_lists(), s.completion_lists())):
        ids = torch.tensor([p + c])
        lp = pol(input_ids=ids).logits[0].log_softmax(-1)
        lq = ref(input_ids=ids).logits[0].log_softmax(-1)
        manual = sum((lp[len(p) + t - 1].exp() * (lp[len(p) + t - 1] - lq[len(p) + t - 1])).sum().item()
                     for t in range(len(c)))
        assert abs(exact[i].item() - manual) < 1e-4

    ids, attn = s.full()
    comp = torch.cat([torch.zeros_like(s.prompt_mask), s.completion_mask], 1)
    diff = sequence_logprobs(pol, ids, attn, comp) - sequence_logprobs(ref, ids, attn, comp)
    torch.testing.assert_close(samp, diff, atol=1e-4, rtol=1e-4)


def test_per_token_kl_nonnegative_and_zero_on_equal():
    a, b = torch.randn(4, 7, 11), torch.randn(4, 7, 11)
    assert (per_token_kl(a, b) >= -1e-6).all()
    torch.testing.assert_close(per_token_kl(a, a), torch.zeros(4, 7), atol=1e-6, rtol=0)
