"""Losses: DPO (paper Eq 7 / App B), Bradley-Terry reward modelling (Eq 2), and PPO pieces (GAE, clipped
policy and value losses). All masked reductions are means over mask==1 entries."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):
    """Paper App B, unchanged.

    pi_logps: policy logprobs, shape (B,)
    ref_logps: reference model logprobs, shape (B,)
    yw_idxs: preferred completion indices in [0, B-1], shape (T,)
    yl_idxs: dispreferred completion indices in [0, B-1], shape (T,)
    beta: temperature controlling strength of KL penalty

    Each pair of (yw_idxs[i], yl_idxs[i]) represents the indices of a single preference pair.
    """
    pi_yw_logps, pi_yl_logps = pi_logps[yw_idxs], pi_logps[yl_idxs]
    ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs]

    pi_logratios = pi_yw_logps - pi_yl_logps
    ref_logratios = ref_yw_logps - ref_yl_logps

    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    rewards = beta * (pi_logps - ref_logps).detach()

    return losses, rewards


def bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """Paper Eq 2: -log sigma(r(x, y_w) - r(x, y_l)), per pair."""
    return -F.logsigmoid(r_chosen - r_rejected)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp(min=1)


def masked_whiten(x: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(x.dtype)
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    out = (x - mean) * torch.rsqrt(var + eps)
    if not shift_mean:
        out = out + mean
    return out * mask


def gae(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation over right-padded token sequences.

    rewards, values, mask: (B, T); mask is a prefix of ones per row (completion tokens). The value after
    the final real token is 0 (episode ends). Returns (advantages, returns), zero outside the mask.
    """
    B, T = rewards.shape
    mask = mask.to(rewards.dtype)
    adv = torch.zeros_like(rewards)
    last = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        next_mask = mask[:, t + 1] if t + 1 < T else torch.zeros_like(last)
        next_value = values[:, t + 1] * next_mask if t + 1 < T else torch.zeros_like(last)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last = delta + gamma * lam * next_mask * last
        adv[:, t] = last
    adv = adv * mask
    returns = (adv + values) * mask
    return adv, returns


def ppo_policy_loss(logp: torch.Tensor, old_logp: torch.Tensor, advantages: torch.Tensor, mask: torch.Tensor,
                    clip_range: float) -> tuple[torch.Tensor, dict]:
    ratio = torch.exp(logp - old_logp)
    unclipped = -advantages * ratio
    clipped = -advantages * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    loss = masked_mean(torch.maximum(unclipped, clipped), mask)
    with torch.no_grad():
        stats = {
            "clipfrac": masked_mean((clipped > unclipped).float(), mask).item(),
            "approx_kl": masked_mean(0.5 * (logp - old_logp) ** 2, mask).item(),
        }
    return loss, stats


def ppo_value_loss(values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor, mask: torch.Tensor,
                   clip_range_value: float) -> tuple[torch.Tensor, dict]:
    v_clipped = old_values + (values - old_values).clamp(-clip_range_value, clip_range_value)
    loss = 0.5 * masked_mean(torch.maximum((values - returns) ** 2, (v_clipped - returns) ** 2), mask)
    with torch.no_grad():
        stats = {"vf_clipfrac": masked_mean(((v_clipped - returns) ** 2 > (values - returns) ** 2).float(), mask).item()}
    return loss, stats
