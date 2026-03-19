"""
BM25 + Reciprocal Rank Fusion — Hybrid Lexical/Semantic Retrieval

Drop-in hybrid retrieval module. Combines BM25 lexical search with vector
similarity search via Reciprocal Rank Fusion (RRF) to improve recall on
temporal queries, named entities, and exact-match facts where embedding
similarity alone under-retrieves.

Requirements:
    pip install rank_bm25

Patent pending — Lichen Research Inc., Canadian application filed March 2026.
"""

from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "that", "this", "it",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "what", "which", "who", "whom",
})


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if w and w not in _STOPWORDS]


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

class BM25Index:
    """Wraps rank_bm25.BM25Okapi with a document store for scored retrieval.

    Each document is a dict with at minimum:
        - "id": unique identifier
        - "content": text to index

    Any additional keys (metadata, etc.) are preserved in search results.
    """

    def __init__(self):
        self._docs: List[Dict[str, Any]] = []
        self._corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._dirty = True

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Add documents to the index.

        Args:
            docs: List of dicts, each with "id" and "content" keys.
        """
        for doc in docs:
            if "content" not in doc:
                continue
            self._docs.append(doc)
            self._corpus.append(tokenize(doc["content"]))
        self._dirty = True

    def _rebuild(self) -> None:
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
        self._dirty = False

    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """Return top_k documents ranked by BM25 score.

        Returns list of dicts with original fields plus "score" and "rank".
        """
        if self._dirty:
            self._rebuild()
        if not self._bm25 or not self._docs:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(scored[:top_k], start=1):
            if score <= 0:
                break
            entry = dict(self._docs[idx])
            entry["score"] = float(score)
            entry["rank"] = rank
            results.append(entry)

        return results

    @property
    def size(self) -> int:
        return len(self._docs)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    id_key: str = "id",
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for a document = sum over all lists of 1 / (k + rank_i)
    where rank_i is the 1-indexed position in list i (if present).

    Args:
        result_lists: List of ranked result lists. Each result must contain
                      id_key for deduplication and optionally "rank"
                      (1-indexed). If "rank" is absent, list position is used.
        k: RRF constant (default 60 per Cormack 2009). Higher k reduces
           the influence of high-ranking items.
        id_key: Key used to identify unique documents across lists.

    Returns:
        Fused list sorted by RRF score descending, with "rrf_score" and
        "rrf_rank" added to each dict.
    """
    rrf_scores: Dict[Any, float] = {}
    doc_map: Dict[Any, Dict[str, Any]] = {}

    for result_list in result_lists:
        for position, doc in enumerate(result_list, start=1):
            doc_id = doc.get(id_key)
            if doc_id is None:
                continue
            rank = doc.get("rank", position)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = dict(doc)

    sorted_ids = sorted(rrf_scores, key=lambda did: rrf_scores[did], reverse=True)

    fused = []
    for rrf_rank, doc_id in enumerate(sorted_ids, start=1):
        entry = doc_map[doc_id]
        entry["rrf_score"] = rrf_scores[doc_id]
        entry["rrf_rank"] = rrf_rank
        entry.pop("rank", None)
        fused.append(entry)

    return fused


# ---------------------------------------------------------------------------
# Hybrid Search (convenience wrapper)
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    vector_results: List[Dict[str, Any]],
    bm25_index: BM25Index,
    top_k: int = 20,
    bm25_top_k: int = 50,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """Combine vector search results with BM25 search via RRF.

    Args:
        query: Search query string.
        vector_results: Pre-computed vector similarity results. Each dict
                        must have "id" and "content" fields.
        bm25_index: A pre-built BM25Index over the same document corpus.
        top_k: Number of fused results to return.
        bm25_top_k: BM25 candidates to retrieve before fusion.
        rrf_k: RRF constant.

    Returns:
        Top-k fused results sorted by RRF score. Each dict has "id",
        "content", "rrf_score", "rrf_rank", and original metadata.
    """
    for i, doc in enumerate(vector_results, start=1):
        doc["rank"] = i

    bm25_results = bm25_index.search(query, top_k=bm25_top_k)

    fused = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        k=rrf_k,
    )

    return fused[:top_k]


# ---------------------------------------------------------------------------
# Index builder (in-memory)
# ---------------------------------------------------------------------------

def build_bm25_index(documents: List[Dict[str, Any]]) -> BM25Index:
    """Build a BM25 index from a list of document dicts.

    Each document must have "id" and "content" keys. Additional metadata
    is preserved and returned on search.

    Args:
        documents: List of dicts with at least {"id": ..., "content": ...}.

    Returns:
        Populated BM25Index ready for .search() calls.
    """
    index = BM25Index()
    index.add_documents(documents)
    return index


# ---------------------------------------------------------------------------
# Per-conversation index cache
# ---------------------------------------------------------------------------

_index_cache: Dict[str, BM25Index] = {}


def get_or_build_index(
    key: str,
    documents: Optional[List[Dict[str, Any]]] = None
) -> BM25Index:
    """Return a cached BM25 index for a key, building from documents if needed.

    Args:
        key: Cache key (e.g., conversation ID).
        documents: Documents to index if not cached. Required on first call.

    Returns:
        BM25Index for the given key.
    """
    if key not in _index_cache:
        if documents is None:
            raise ValueError(f"No cached index for '{key}' and no documents provided.")
        _index_cache[key] = build_bm25_index(documents)
    return _index_cache[key]


def clear_index_cache() -> None:
    """Clear all cached BM25 indices."""
    _index_cache.clear()
