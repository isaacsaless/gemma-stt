# -*- coding: utf-8 -*-
"""Multilingual STT (faster-whisper) for any language.
Word timestamps enable click-to-seek and search chunks.
Audio is decoded here (ffmpeg -> float32 16 kHz mono) and handed to faster-whisper as a
ndarray: PyAV (faster-whisper's default decoder) has no Python 3.13 wheel compatible
with the installed version.
"""
from __future__ import annotations

import numpy as np

from .audio import SR, load_audio_16k


class WhisperSTT:
    def __init__(self, size: str = "large-v3-turbo", device: str = "cuda", compute: str = "int8_float16"):
        import faster_whisper

        self.name = f"whisper-{size}"
        self.model = faster_whisper.WhisperModel(size, device=device, compute_type=compute)

    def transcribe(self, path: str | Path, progress=None, language: str | None = None) -> dict:
        wav = load_audio_16k(path)
        return self.transcribe_array(wav, progress=progress, language=language)

    def transcribe_array(self, wav: np.ndarray, progress=None, language: str | None = None,
                         language_only: bool = False) -> dict:
        """wav: float32 mono 16 kHz. language_only: detect and return, no transcript."""
        segments, info = self.model.transcribe(
            wav, language=language or (None if language_only else None),
            beam_size=1, vad_filter=not language_only, word_timestamps=not language_only,
        )
        if language_only:
            for _ in segments:
                break  # language resolution happens before the first segment is produced
            dur = len(wav) / SR
            return {"language": info.language, "duration": dur, "words": [], "text": "",
                    "engine": self.name}
        words = []
        seg_texts = []
        for seg in segments:
            seg_texts.append(seg.text.lstrip())
            for w in seg.words or []:
                tok = w.word.strip()
                if tok:
                    words.append({"w": tok, "start": round(w.start, 2), "end": round(w.end, 2)})
            if progress is not None and seg.end:
                progress(f"whisper {seg.end:.0f}/{info.duration:.0f} s")
        text = " ".join(t.strip() for t in seg_texts if t.strip())
        return {
            "words": words,
            "text": text,
            "language": info.language,
            "duration": info.duration,
            "engine": self.name,
        }
