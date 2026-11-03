# -*- coding: utf-8 -*-
"""STT engine: faster-whisper (large-v3-turbo) transcribes every language.

After any transcript, the EmbeddingGemma 2 text encoder indexes it for semantic search.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch

from .text_embed import TextEmbedder
from .whisper_stt import WhisperSTT


def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Engine:
    """Lazy engine. One per server process."""

    def __init__(self, model_dir: str, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_dir = model_dir
        self._embedder: TextEmbedder | None = None
        self._whisper: WhisperSTT | None = None

    @property
    def embedder(self) -> TextEmbedder:
        if self._embedder is None:
            self._embedder = TextEmbedder(self.model_dir, self.device)
        return self._embedder

    @property
    def whisper(self) -> WhisperSTT:
        if self._whisper is None:
            self._whisper = WhisperSTT(device=self.device, compute="int8")
        return self._whisper

    def transcribe(self, path: Path, progress: Callable[[str], None] | None = None) -> dict:
        t0 = time.perf_counter()
        r = self.whisper.transcribe(path, progress=progress)
        r["n_words"] = len(r["words"])
        r["n_tokens"] = None
        r["realtime_x"] = round(r["duration"] / max(1e-6, time.perf_counter() - t0), 1)
        progress and progress("pronto")
        return r
