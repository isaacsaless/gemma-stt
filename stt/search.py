# -*- coding: utf-8 -*-
"""Semantic search over a transcript with the EmbeddingGemma 2 text encoder (multilingual)."""
from __future__ import annotations

import re

import numpy as np

from .text_embed import TextEmbedder

QUERY_PREFIX = "task: search result | query: "
DOC_PREFIX = "title: none | text: "
CHUNK_MIN_WORDS = 10   # don't close a chunk before this
CHUNK_MAX_WORDS = 56   # hard cap even mid-sentence
_CLOSERS = (".", "!", "?", "…")  # sentence end -> close chunk

# Queries here are often meta-descriptions ("a parte que a moça fala com quem ela
# cresceu"); the scaffolding dilutes the embedding. Strip leading meta words until
# only topical content remains, but keep at least M keep words of the original.
_KEEP = 3
_SCHEMES = (
    re.compile(r"^(?:a|as|o|os)\s+(?:partes?|momentos?|trechos?|instantes?|minuto?s?)\s+(?:que|onde|em\s+que|em\s+q)\s+", re.I),
    re.compile(r"^(?:(?:a|o|as|os)\s+)?(?:moça|moço|cara|pessoa|palestrante|apresentadora?|senhora?|menina|menino|garota|garoto|doutora?|professora?|guri|guria|vencedora?|ele|ela|eles|elas)\s+", re.I),
    re.compile(r"^(?:fal(?:ando|ar|a|ou|aram|am)|diz(?:endo)?|disse|disseram|coment(?:ando|ar|ou|am|a)|cont(?:ando|ar|ou|am|a)|explic(?:ando|ar|ou|am|a)|mencion(?:ando|ar|ou|a)|pergunt(?:ando|ar|ou|a)|respon(?:dendo|der|deu|dem))\s+", re.I),
    re.compile(r"^(?:sobre|de|da|do|das|dos|em|na|no|nas|nos|que|o\s+que|quem|com|quando|qual|quais|onde|como|por\s+que|pra|para|por|olha|tipo)\s+", re.I),
)


def normalize_query(q: str) -> str:
    """-> topic-only query (keeps the original when nothing meaningful remains)."""
    t = q.strip()
    if not t:
        return t
    orig_words = len(t.split())
    while True:
        m = next((p.match(t) for p in _SCHEMES if p.match(t)), None)
        if m is None:
            break
        new = t[m.end():].strip()
        if len(new.split()) < max(_KEEP, round(orig_words * 0.3)):
            break
        t = new
    return t or q.strip()


class TextIndexer:
    def __init__(self, embedder: TextEmbedder):
        self.emb = embedder

    def chunk(self, words: list[dict]) -> list[dict]:
        """Split at sentence boundaries (word tokens carry punctuation) between MIN and MAX."""
        chunks = []
        n = len(words)
        i = 0
        while i < n:
            j = i + 1
            while j < n:
                if j - i >= CHUNK_MAX_WORDS or (
                    j - i >= CHUNK_MIN_WORDS and words[j - 1]["w"].endswith(_CLOSERS)
                ):
                    break
                j += 1
            chunks.append({
                "i0": i, "i1": j,
                "text": " ".join(w["w"] for w in words[i:j]),
                "start": words[i]["start"],
                "end": words[j - 1]["end"],
            })
            i = j
        return chunks

    def index(self, words: list[dict]) -> dict:
        chunks = self.chunk(words)
        if not chunks:
            return {"chunks": chunks, "embs": np.zeros((0, 768), np.float32)}
        docs = [DOC_PREFIX + c["text"] for c in chunks]
        embs = self.emb.embed(docs, batch_size=8)
        return {"chunks": chunks, "embs": np.ascontiguousarray(embs.astype(np.float32))}

    def search(self, index: dict, q: str, top_k: int = 8) -> list[dict]:
        if not index["chunks"] or not q.strip():
            return []
        qv = self.emb.embed([QUERY_PREFIX + normalize_query(q.strip())])[0]
        sims = (index["embs"] @ qv).astype(float)
        order = np.argsort(-sims)[:min(top_k, len(sims))]
        return [
            {
                "score": round(float(sims[k]), 4),
                "start": index["chunks"][k]["start"],
                "end": index["chunks"][k]["end"],
                "text": index["chunks"][k]["text"],
            }
            for k in order
        ]
