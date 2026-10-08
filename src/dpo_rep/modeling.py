"""Model loading and log-prob / value / reward computations.

HF `transformers` is used only to load pretrained weights and tokenizers; all scoring logic is ours.

Sequence layout convention: sequences may be padded on either side (prompts are left-padded for
sampling, completions right-padded). Position ids are always derived from the attention mask, so
padding never shifts the positions of real tokens.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from dpo_rep.utils import disable_dropout


def load_tokenizer(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_lm(name_or_path: str, device: torch.device, dropout: bool = True) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=torch.float32).to(device)
    if not dropout:
        disable_dropout(model)
    return model


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


def lm_forward(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
    return model(input_ids=input_ids, attention_mask=attention_mask,
                 position_ids=position_ids_from_mask(attention_mask), **kwargs)


def token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """log p(input_ids[:, t+1] | input_ids[:, :t+1]) for t = 0..L-2. Returns (B, L-1) float32."""
    logits = logits[:, :-1].float()
    targets = input_ids[:, 1:].unsqueeze(-1)
    return logits.gather(-1, targets).squeeze(-1) - logits.logsumexp(-1)


def sequence_logprobs(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      completion_mask: torch.Tensor) -> torch.Tensor:
    """Sum of log-probs of completion tokens only (prompt and padding excluded). Returns (B,).

    completion_mask is aligned with input_ids: 1 where the token belongs to the completion.
    """
    logits = lm_forward(model, input_ids, attention_mask).logits
    lp = token_logprobs(logits, input_ids)
    return (lp * completion_mask[:, 1:].to(lp.dtype)).sum(-1)


class PolicyWithValue(nn.Module):
    """Causal LM with a scalar value head on the shared trunk (final hidden states), for PPO."""

    def __init__(self, lm: PreTrainedModel):
        super().__init__()
        self.lm = lm
        self.value_head = nn.Linear(lm.config.hidden_size, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.value_head.to(next(lm.parameters()).device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = lm_forward(self.lm, input_ids, attention_mask, output_hidden_states=True)
        values = self.value_head(out.hidden_states[-1]).squeeze(-1).float()
        return out.logits, values


def last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last attended position per row (works for left, right or two-sided padding)."""
    L = attention_mask.shape[1]
    return L - 1 - attention_mask.long().flip(-1).argmax(-1)


class RewardModel(nn.Module):
    """Transformer backbone + linear head on the last non-pad token, trained with the Bradley-Terry
    loss (paper Eq 2). `score_mean` is set after training so E[r] = 0 on SFT samples (paper Sec 3)."""

    def __init__(self, backbone: PreTrainedModel):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, 1)
        # Zero init: every pair starts at r_w - r_l = 0, i.e. BT loss = ln 2.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("score_mean", torch.zeros(()))
        self.register_buffer("score_std", torch.ones(()))

    @classmethod
    def from_lm_name(cls, name_or_path: str, device: torch.device) -> "RewardModel":
        backbone = AutoModel.from_pretrained(name_or_path, dtype=torch.float32)
        return cls(backbone).to(device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        h = lm_forward(self.backbone, input_ids, attention_mask).last_hidden_state
        idx = last_token_index(attention_mask)
        r = self.head(h[torch.arange(h.shape[0], device=h.device), idx]).squeeze(-1).float()
        if normalize:
            r = r - self.score_mean
        return r

    def save(self, path: str | Path) -> None:
        path = Path(path)
        self.backbone.save_pretrained(path / "backbone")
        torch.save({"head": self.head.state_dict(), "score_mean": self.score_mean.cpu(),
                    "score_std": self.score_std.cpu()}, path / "head.pt")

    @classmethod
    def load(cls, path: str | Path, device: torch.device) -> "RewardModel":
        path = Path(path)
        rm = cls(AutoModel.from_pretrained(path / "backbone", dtype=torch.float32))
        state = torch.load(path / "head.pt", map_location="cpu")
        rm.head.load_state_dict(state["head"])
        rm.score_mean.copy_(state["score_mean"])
        rm.score_std.copy_(state["score_std"])
        return rm.to(device)
