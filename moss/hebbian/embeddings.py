"""
Mock Embedding Layer for Hebbian Memory

Accepts text and returns a fixed-dimension vector for testing.
In production, replace this stub with your actual embedding model.

Production note:
    Production embeddings use qwen3-embedding (4096-dimensional vectors).
    Replace this stub with your embedding model of choice.
    The Hebbian recall pipeline is embedding-model-agnostic — any model
    that returns a consistent-dimension float vector will work.

    Example production swap:
        import httpx
        def get_embedding(text: str) -> List[float]:
            resp = httpx.post(
                "http://your-embedding-server/api/embeddings",
                json={"model": "your-model", "prompt": text},
                timeout=30.0,
            )
            return resp.json()["embedding"]
"""

import hashlib
import math
import random
from typing import List

# Embedding dimension. Production uses 4096D (qwen3-embedding).
EMBEDDING_DIM = 4096


def get_embedding(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """
    Return a deterministic pseudo-random embedding for the given text.

    Uses the text's hash as a seed so that the same text always produces
    the same vector — necessary for reproducible test assertions.

    NOTE: This stub produces random vectors unsuitable for semantic search.
    Replace with a real embedding model for production use.

    Args:
        text: Input text.
        dim: Output dimensionality.

    Returns:
        List of `dim` floats in [-1, 1], L2-normalised.
    """
    # Deterministic seed from content hash
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    raw = [rng.gauss(0, 1) for _ in range(dim)]

    # L2 normalise
    norm = math.sqrt(sum(x * x for x in raw))
    if norm == 0:
        return [0.0] * dim
    return [x / norm for x in raw]


def get_embedding_batch(texts: List[str], dim: int = EMBEDDING_DIM) -> List[List[float]]:
    """Return embeddings for a batch of texts."""
    return [get_embedding(t, dim) for t in texts]
