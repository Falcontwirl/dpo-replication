"""Sequence-level KL(pi || pi_ref) on sampled completions.

Paper footnote 3: sequence-level KL is "the sum of the per-timestep KL-divergences". We compute it
exactly per timestep over the full vocabulary, at the contexts visited by pi's own samples:

    KL_exact(x, y) = sum_t sum_v pi(v | x, y_<t) [log pi(v | x, y_<t) - log pi_ref(v | x, y_<t)]

Averaging over y ~ pi gives an unbiased estimate of KL(pi(.|x) || pi_ref(.|x)) with lower variance
than the single-sample estimate sum_t [log pi(y_t|.) - log pi_ref(y_t|.)], which we also return for comparison.
"""

from __future__ import annotations

import torch

from dpo_rep.modeling import lm_forward
from dpo_rep.sampling import Samples


def per_token_kl(policy_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """Exact KL(p || q) per position over the last (vocab) dim. Inputs (..., V); returns (...) float32."""
    p_logp = policy_logits.float().log_softmax(-1)
    q_logp = ref_logits.float().log_softmax(-1)
    return (p_logp.exp() * (p_logp - q_logp)).sum(-1)


@torch.no_grad()
def completion_kl(policy: torch.nn.Module, ref: torch.nn.Module, samples: Samples) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (kl_exact, kl_sample), each shape (B,): per-sequence sums over completion tokens."""
    input_ids, attn = samples.full()
    P = samples.prompt_len
    pol = lm_forward(policy, input_ids, attn).logits[:, P - 1:-1].float()
    ref_ = lm_forward(ref, input_ids, attn).logits[:, P - 1:-1].float()
    mask = samples.completion_mask.float()

    kl_exact = (per_token_kl(pol, ref_) * mask).sum(-1)

    tok = samples.completion_ids.unsqueeze(-1)
    lp_pol = pol.log_softmax(-1).gather(-1, tok).squeeze(-1)
    lp_ref = ref_.log_softmax(-1).gather(-1, tok).squeeze(-1)
    kl_sample = ((lp_pol - lp_ref) * mask).sum(-1)
    return kl_exact, kl_sample
