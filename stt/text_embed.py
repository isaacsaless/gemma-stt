# -*- coding: utf-8 -*-
"""EmbeddingGemma 2 text embedder (text-only, fp32) for semantic search over transcripts."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class TextEmbedder:
    """Loads only the text backbone (vision_config=None, audio_config=None → 270M)."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        from transformers import AutoTokenizer, EmbeddingGemma2Model

        self.device = device
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # bf16: same 8-bit exponent as fp32 (safe per model card), halves VRAM
        self.model = EmbeddingGemma2Model.from_pretrained(
            model_dir, dtype=torch.bfloat16, vision_config=None, audio_config=None,
        ).to(device).eval()

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 16, max_tokens: int = 2048) -> np.ndarray:
        """[B, 768] unit embeddings (mean pool over tokens -> L2)."""
        embs = []
        for a in range(0, len(texts), batch_size):
            batch = texts[a:a + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                 truncation=True, max_length=max_tokens)
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            hs = self.model.language_model(input_ids=ids, attention_mask=mask)["last_hidden_state"]
            hs = hs.float()
            m = mask.unsqueeze(-1)
            mean = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            embs.append(F.normalize(mean.float(), dim=-1).cpu())
        return torch.cat(embs).numpy()
