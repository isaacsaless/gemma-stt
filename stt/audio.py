# -*- coding: utf-8 -*-
"""Audio decoding: any format -> float32 mono 16 kHz waveform (ffmpeg pipe)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SR = 16000


def load_audio_16k(path: Path) -> np.ndarray:
    """-> float32 mono 16 kHz waveform."""
    from imageio_ffmpeg import get_ffmpeg_exe
    exe = get_ffmpeg_exe()
    cmd = [exe or "", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"]
    proc = subprocess.run([c for c in cmd if c], capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode(errors='replace')[:400]}")
    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    return np.ascontiguousarray(wav)
