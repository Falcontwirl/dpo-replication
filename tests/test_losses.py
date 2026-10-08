import math

import torch
import torch.nn.functional as F

from dpo_rep.losses import (bradley_terry_loss, dpo_loss, gae, masked_mean, masked_whiten,
                            ppo_policy_loss, ppo_value_loss)


def paper_dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):
    """Verbatim copy of the paper's App B reference code, used as an oracle."""
    pi_yw_logps,  pi_yl_logps =  pi_logps[yw_idxs],  pi_logps[yl_idxs]
    ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs]

    pi_logratios  = pi_yw_logps - pi_yl_logps
    ref_logratios = ref_yw_logps - ref_yl_logps

    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    rewards = beta * (pi_logps - ref_logps).detach()

    return losses, rewards


def test_dpo_loss_matches_paper_reference():
    torch.manual_seed(0)
    pi, ref = torch.randn(8), torch.randn(8)
    yw, yl = torch.tensor([0, 2, 4, 6]), torch.tensor([1, 3, 5, 7])
    for beta in [0.05, 0.1, 1.0, 5.0]:
        got = dpo_loss(pi, ref, yw, yl, beta)
        want = paper_dpo_loss(pi, ref, yw, yl, beta)
        torch.testing.assert_close(got[0], want[0])
        torch.testing.assert_close(got[1], want[1])


def test_dpo_loss_is_log2_when_policy_equals_reference():
    logps = torch.randn(6)
    losses, rewards = dpo_loss(logps, logps.clone(), torch.arange(3), torch.arange(3, 6), beta=0.1)
    torch.testing.assert_close(losses, torch.full((3,), math.log(2)))
    torch.testing.assert_close(rewards, torch.zeros(6))


def test_dpo_gradient_matches_closed_form():
    """Paper Sec 4: grad = -beta * sigma(r_l - r_w) * (grad log pi(y_w) - grad log pi(y_l))."""
    beta = 0.5
    pi = torch.tensor([-3.0, -2.0], requires_grad=True)  # [y_w, y_l]
    ref = torch.tensor([-2.5, -2.5])
    loss, _ = dpo_loss(pi, ref, torch.tensor([0]), torch.tensor([1]), beta)
    loss.sum().backward()
    r_w, r_l = beta * (pi[0] - ref[0]), beta * (pi[1] - ref[1])
    w = torch.sigmoid(r_l - r_w).item()
    torch.testing.assert_close(pi.grad, torch.tensor([-beta * w, beta * w]))


def test_bradley_terry_loss():
    torch.testing.assert_close(bradley_terry_loss(torch.tensor([2.0]), torch.tensor([0.0])),
                               torch.tensor([math.log1p(math.exp(-2.0))]))


def test_masked_mean_and_whiten():
    x = torch.tensor([[1.0, 2.0, 100.0], [3.0, 4.0, 100.0]])
    m = torch.tensor([[1, 1, 0], [1, 1, 0]])
    assert masked_mean(x, m).item() == 2.5
    w = masked_whiten(x, m)
    assert abs(masked_mean(w, m).item()) < 1e-6
    assert abs(masked_mean(w ** 2, m).item() - 1.0) < 1e-4
    assert (w[:, 2] == 0).all()


def gae_reference(r, v, gamma, lam):
    """Plain single-sequence GAE, terminal value 0."""
    T = len(r)
    adv = [0.0] * T
    last = 0.0
    for t in reversed(range(T)):
        nv = v[t + 1] if t + 1 < T else 0.0
        delta = r[t] + gamma * nv - v[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv


def test_gae_matches_reference_with_padding():
    r = torch.tensor([[0.1, -0.2, 0.3, 1.0], [0.5, 2.0, 0.0, 0.0]])
    v = torch.tensor([[0.2, 0.1, -0.1, 0.4], [0.3, 0.7, 9.0, 9.0]])  # padded values must be ignored
    m = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    adv, ret = gae(r, v, m, gamma=0.99, lam=0.95)
    torch.testing.assert_close(adv[0], torch.tensor(gae_reference(r[0].tolist(), v[0].tolist(), 0.99, 0.95)))
    torch.testing.assert_close(adv[1, :2], torch.tensor(gae_reference(r[1, :2].tolist(), v[1, :2].tolist(), 0.99, 0.95)))
    assert (adv[1, 2:] == 0).all() and (ret[1, 2:] == 0).all()
    torch.testing.assert_close(ret[0], adv[0] + v[0])


def test_gae_gamma1_lam1_is_reward_to_go_minus_value():
    r = torch.tensor([[0.0, 0.0, 1.0]])
    v = torch.tensor([[0.3, 0.2, 0.1]])
    adv, ret = gae(r, v, torch.ones(1, 3), gamma=1.0, lam=1.0)
    torch.testing.assert_close(ret, torch.tensor([[1.0, 1.0, 1.0]]))


def test_ppo_policy_loss_clipping():
    old = torch.zeros(1, 2)
    mask = torch.ones(1, 2)
    adv = torch.tensor([[1.0, -1.0]])
    # ratio = e^0.5 ~ 1.65 > 1.2: positive advantage is clipped (no gradient), negative is not.
    logp = torch.full((1, 2), 0.5, requires_grad=True)
    loss, stats = ppo_policy_loss(logp, old, adv, mask, clip_range=0.2)
    loss.backward()
    assert logp.grad[0, 0] == 0
    assert logp.grad[0, 1] > 0
    assert stats["clipfrac"] == 0.5
    expected = 0.5 * (-1.0 * 1.2 + 1.0 * math.exp(0.5))
    assert abs(loss.item() - expected) < 1e-5


def test_ppo_value_loss_clipping():
    old = torch.zeros(1, 1)
    ret = torch.ones(1, 1)
    v = torch.full((1, 1), 2.0, requires_grad=True)
    loss, _ = ppo_value_loss(v, old, ret, torch.ones(1, 1), clip_range_value=0.2)
    # unclipped err = 1, clipped (v=0.2) err = 0.64 -> max = 1
    assert abs(loss.item() - 0.5) < 1e-6
