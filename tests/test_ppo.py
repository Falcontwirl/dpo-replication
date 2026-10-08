from types import SimpleNamespace

import pytest
import torch

from conftest import EOS, PAD, make_tiny_lm
from dpo_rep.modeling import PolicyWithValue
from dpo_rep.ppo import AdaptiveKLController, ppo_step, repad
from dpo_rep.sampling import sample
from dpo_rep.utils import Timer


def test_kl_controller_moves_toward_target():
    ctl = AdaptiveKLController(0.2, target=6.0, horizon=10000)
    ctl.update(12.0, 1000)  # KL too high -> coefficient grows (error clipped to +0.2)
    assert ctl.value == pytest.approx(0.2 * 1.02)
    ctl = AdaptiveKLController(0.2, target=6.0, horizon=10000)
    ctl.update(3.0, 1000)  # KL too low -> coefficient shrinks
    assert ctl.value == pytest.approx(0.2 * 0.98)


def test_repad_fixed_layout(tiny_lm):
    a = sample(tiny_lm, [[1, 2], [3, 4, 5]], 4, EOS, PAD, generator=torch.Generator().manual_seed(0))
    b = sample(tiny_lm, [[6]], 3, EOS, PAD, generator=torch.Generator().manual_seed(1))
    s = repad([a, b], PAD, P=8, T=6)
    assert s.prompt_ids.shape == (3, 8) and s.completion_ids.shape == (3, 6)
    assert s.prompt_lists() == [[1, 2], [3, 4, 5], [6]]
    assert s.completion_lists() == a.completion_lists() + b.completion_lists()


def test_ppo_increases_toy_reward():
    """Reward = number of times token TARGET appears in the completion. A few PPO steps must raise it."""
    torch.manual_seed(0)
    TARGET = 7
    policy = PolicyWithValue(make_tiny_lm(0))
    ref = make_tiny_lm(0).requires_grad_(False)
    tok = SimpleNamespace(pad_token_id=PAD, eos_token_id=EOS)
    c = SimpleNamespace(gen_batch_size=64, micro_batch_size=32, minibatch_size=64, ppo_epochs=4, clip_range=0.2,
                        clip_range_value=0.2, vf_coef=0.1, gamma=1.0, lam=0.95, whiten_scores=True, max_grad_norm=1.0)
    gen_cfg = SimpleNamespace(max_new_tokens=8, temperature=1.0, top_k=0, top_p=1.0)
    opt = torch.optim.Adam([{"params": policy.lm.parameters(), "lr": 3e-3},
                            {"params": policy.value_head.parameters(), "lr": 1e-2}])
    kl_ctl = AdaptiveKLController(0.01, target=6.0, horizon=10000)
    timer = Timer(torch.device("cpu"))
    prompts = [[1, 2, 3]] * 64

    def reward_fn(s):
        return ((s.completion_ids == TARGET) & s.completion_mask.bool()).sum(1).float()

    g = torch.Generator().manual_seed(0)
    scores = []
    for _ in range(12):
        st = ppo_step(policy, ref, reward_fn, prompts, c, gen_cfg, kl_ctl, opt, tok, torch.device("cpu"),
                      "fp32", timer, max_prompt_len=3, generator=g)
        scores.append(st["rollout_score_mean"])
        assert st["rollout_kl"] >= -1e-3  # sample KL estimate should not be very negative on average
    assert sum(scores[-3:]) / 3 > 2 * max(scores[0], 0.1), scores
