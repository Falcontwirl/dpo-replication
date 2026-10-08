"""Autoregressive sampling with a KV cache, written from scratch (no `model.generate`).

Output layout (fixed offsets, which PPO and KL code rely on):
    prompt_ids      (B, P)  left-padded
    completion_ids  (B, T)  right-padded with pad_id after EOS / max length
Concatenating gives (B, P+T); completion token t sits at column P+t and is predicted by the logits
at column P+t-1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dpo_rep.modeling import position_ids_from_mask


@dataclass
class Samples:
    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor

    @property
    def prompt_len(self) -> int:
        return self.prompt_ids.shape[1]

    def full(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(input_ids, attention_mask) of prompt + completion, shape (B, P+T)."""
        return (torch.cat([self.prompt_ids, self.completion_ids], dim=1),
                torch.cat([self.prompt_mask, self.completion_mask], dim=1))

    def completion_lists(self) -> list[list[int]]:
        return [ids[m.bool()].tolist() for ids, m in zip(self.completion_ids.cpu(), self.completion_mask.cpu())]

    def prompt_lists(self) -> list[list[int]]:
        return [ids[m.bool()].tolist() for ids, m in zip(self.prompt_ids.cpu(), self.prompt_mask.cpu())]


def left_pad(seqs: list[list[int]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    P = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), P), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), P), dtype=torch.long)
    for i, s in enumerate(seqs):
        if s:
            ids[i, P - len(s):] = torch.tensor(s)
            mask[i, P - len(s):] = 1
    return ids.to(device), mask.to(device)


def right_pad(seqs: list[list[int]], pad_id: int, device: torch.device, length: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    T = length if length is not None else max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


def filter_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Top-k then nucleus filtering; filtered entries set to -inf. logits: (B, V) float32."""
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        # Remove tokens once the cumulative mass *before* them already exceeds top_p (always keep the first).
        remove = (cum - sorted_logits.softmax(-1)) > top_p
        remove_orig = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove_orig, float("-inf"))
    return logits


def make_generator(device: torch.device, seed: int) -> torch.Generator | None:
    try:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return g
    except RuntimeError:
        return None


@torch.no_grad()
def sample(model: torch.nn.Module, prompts: list[list[int]], max_new_tokens: int, eos_id: int, pad_id: int,
           temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
           generator: torch.Generator | None = None) -> Samples:
    """Sample completions for a batch of prompts (lists of token ids). temperature == 0 means greedy.

    A completion ends at (and includes) the first EOS, or after max_new_tokens tokens.
    """
    device = next(model.parameters()).device
    prompt_ids, prompt_mask = left_pad(prompts, pad_id, device)
    B = prompt_ids.shape[0]

    attn = prompt_mask
    out = model(input_ids=prompt_ids, attention_mask=attn, position_ids=position_ids_from_mask(attn), use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1].float()
    next_pos = attn.sum(-1)  # position id of the first generated token (B,)

    tokens, masks = [], []
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for step in range(max_new_tokens):
        if temperature == 0:
            tok = next_logits.argmax(-1)
        else:
            probs = filter_logits(next_logits / temperature, top_k, top_p).softmax(-1)
            tok = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        alive = ~finished
        tok = torch.where(alive, tok, torch.full_like(tok, pad_id))
        tokens.append(tok)
        masks.append(alive.long())
        finished = finished | (alive & (tok == eos_id))
        if finished.all() or step == max_new_tokens - 1:
            break
        attn = torch.cat([attn, alive.long()[:, None]], dim=1)
        out = model(input_ids=tok[:, None], attention_mask=attn, position_ids=next_pos[:, None],
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1].float()
        next_pos = next_pos + alive.long()

    return Samples(prompt_ids, prompt_mask, torch.stack(tokens, 1), torch.stack(masks, 1))


def decode_texts(tokenizer, samples: Samples) -> list[str]:
    """Decoded prompt + completion text (special tokens dropped), as scored by the sentiment classifier."""
    return [tokenizer.decode(p + c, skip_special_tokens=True)
            for p, c in zip(samples.prompt_lists(), samples.completion_lists())]
