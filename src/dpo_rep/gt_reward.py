"""Ground-truth reward: p(positive | x, y) from siebert/sentiment-roberta-large-english (paper App C.1)."""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentReward:
    def __init__(self, name: str, device: torch.device, batch_size: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float32).to(device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        if "positive" not in labels:
            raise ValueError(f"No POSITIVE label in {self.model.config.id2label}")
        self.pos_idx = labels["positive"]
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> torch.Tensor:
        """Returns p(positive) for each text, float32 tensor on CPU."""
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                 max_length=512, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.float()
            out.append(logits.softmax(-1)[:, self.pos_idx].cpu())
        return torch.cat(out)
