# © 2026 Plant Operations Brain Team — ET AI Hackathon. All rights reserved.
# Proprietary & confidential. Unauthorised copying, reuse, or redistribution is prohibited (see LICENSE).
"""
embed_store.py
--------------
Step 2 of the pipeline: turn text chunks into numbers (embeddings) so we can
find the most relevant ones for any question, using meaning -- not keywords.

We keep it dependency-light on purpose: a local embedding model + NumPy
cosine similarity. For a hackathon-sized corpus (hundreds/thousands of
chunks) this is instant and never breaks during a live demo.
"""

from __future__ import annotations
import os

# Force the transformers library to use PyTorch only. Without this, if TensorFlow
# + Keras 3 happen to be installed system-wide (common on Windows), transformers
# tries to load its TF backend and crashes. We never use TF here.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from .ingest import Chunk

# Small, fast, runs on CPU. Good enough quality; swap for a bigger model later.
_MODEL_NAME = "all-MiniLM-L6-v2"
# Cross-encoder reranker: reads (query, passage) TOGETHER, so it is far more
# precise than bi-encoder cosine — but too slow to run over the whole corpus.
# We use it as a second stage: fast vector search proposes, this disposes.
_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Two-stage retrieval stage 2: re-score candidate chunks on the GPU."""

    def __init__(self, model_name: str = _RERANK_MODEL):
        self._model = None
        self._name = model_name
        # Warm-up happens on a background thread at app startup; the lock makes
        # a first query arriving mid-warm-up simply wait a moment instead of
        # loading the model twice.
        import threading
        self._lock = threading.Lock()

    def _lazy(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                device = "cpu"
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    pass
                self._model = CrossEncoder(self._name, device=device,
                                           max_length=512)
            return self._model

    def warm(self) -> None:
        """Load the model now (at startup) so the first user query isn't slow."""
        try:
            self._lazy().predict([("warm up", "warm up")], show_progress_bar=False)
        except Exception:
            pass

    def rerank(self, query: str, hits: list[tuple[Chunk, float]],
               top_k: int) -> list[tuple[Chunk, float]]:
        """Reorder `hits` by cross-encoder relevance; return best `top_k`."""
        if len(hits) <= 1:
            return hits[:top_k]
        model = self._lazy()
        pairs = [(query, c.text) for c, _ in hits]
        scores = model.predict(pairs, show_progress_bar=False)
        order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))
        # Normalise the cross-encoder logit to a 0..1 confidence for display.
        import math
        return [(hits[i][0], 1 / (1 + math.exp(-float(scores[i]))))
                for i in order[:top_k]]


class VectorStore:
    """Holds chunks + their embeddings and answers 'what's most relevant?'."""

    def __init__(self, model_name: str = _MODEL_NAME):
        # Use the GPU automatically if one is present (e.g. an RTX 4060),
        # which makes embedding the document corpus 5-10x faster.
        device = None
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None  # shape (n_chunks, dim)

    # ---------- building ----------
    def build(self, chunks: list[Chunk]) -> None:
        """Embed every chunk once and store the matrix."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        print(f"Embedding {len(texts)} chunks (first run downloads the model)...")
        vecs = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,  # makes cosine == dot product
        )
        self.embeddings = np.asarray(vecs, dtype=np.float32)

    # ---------- searching ----------
    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the top-k chunks most similar in meaning to the query."""
        if self.embeddings is None:
            raise RuntimeError("Store not built yet. Call build() or load() first.")
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q          # cosine similarity for all chunks
        top_idx = np.argsort(-scores)[:k]     # highest scores first
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    # ---------- persistence (so we don't re-embed every app restart) ----------
    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "embeddings.npy", self.embeddings)
        with open(folder / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.as_dict() for c in self.chunks], f)
        print(f"Saved index to {folder}")

    def load(self, folder: str | Path) -> bool:
        """Load a previously built index. Returns False if none exists yet."""
        folder = Path(folder)
        emb_path = folder / "embeddings.npy"
        chunk_path = folder / "chunks.json"
        if not (emb_path.exists() and chunk_path.exists()):
            return False
        self.embeddings = np.load(emb_path)
        with open(chunk_path, encoding="utf-8") as f:
            self.chunks = [Chunk(**d) for d in json.load(f)]
        print(f"Loaded index: {len(self.chunks)} chunks from {folder}")
        return True
