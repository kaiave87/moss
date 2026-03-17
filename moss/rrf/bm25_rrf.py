"""
BM25 + Reciprocal Rank Fusion for hybrid memory retrieval.

Combines BM25 lexical search with vector similarity search
via Reciprocal Rank Fusion (RRF) to improve recall
on conversational memory benchmarks — especially for temporal queries, named entities, and
exact-match facts where embedding similarity alone under-retrieves.

Requirements:
    pip install rank_bm25

Usage with your memory adapter:
    See INTEGRATION EXAMPLE at bottom of file.
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

    Any additional keys (metadata, conversation_id, etc.) are preserved and
    returned in search results.
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
                  Extra keys are stored and returned on search.
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

        Returns list of dicts: original doc fields + "score" (BM25 score)
        and "rank" (1-indexed position).
        """
        if self._dirty:
            self._rebuild()
        if not self._bm25 or not self._docs:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)

        # Pair scores with doc indices, sort descending
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score) in enumerate(scored[:top_k], start=1):
            if score <= 0:
                break
            entry = dict(self._docs[idx])  # copy
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
        result_lists: List of ranked result lists. Each result is a dict
                      that must contain `id_key` for dedup and optionally
                      "rank" (1-indexed). If "rank" is missing, position
                      in the list is used.
        k: RRF constant (default 60 per original paper). Higher k reduces
           the influence of high-ranking items.
        id_key: Key used to identify unique documents across lists.

    Returns:
        Fused list sorted by RRF score descending. Each dict has the original
        fields plus "rrf_score" and "rrf_rank".
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
            # Keep the first (highest-quality) version of the doc metadata
            if doc_id not in doc_map:
                doc_map[doc_id] = dict(doc)

    # Sort by fused score
    sorted_ids = sorted(rrf_scores, key=lambda did: rrf_scores[did], reverse=True)

    fused = []
    for rrf_rank, doc_id in enumerate(sorted_ids, start=1):
        entry = doc_map[doc_id]
        entry["rrf_score"] = rrf_scores[doc_id]
        entry["rrf_rank"] = rrf_rank
        # Remove per-source rank/score to avoid confusion
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
        query: The search query string.
        vector_results: Pre-computed vector similarity results. Each dict
                        must have "id", "content", "score". These come
                        directly from _scoped_recall().
        bm25_index: A pre-built BM25Index over the same document corpus.
        top_k: Number of fused results to return.
        bm25_top_k: How many BM25 candidates to retrieve before fusion.
        rrf_k: RRF constant.

    Returns:
        Top-k fused results sorted by RRF score. Each dict has "id",
        "content", "rrf_score", "rrf_rank", and any metadata from the
        original documents.
    """
    # Assign ranks to vector results (they should already be sorted by score)
    for i, doc in enumerate(vector_results, start=1):
        doc["rank"] = i

    bm25_results = bm25_index.search(query, top_k=bm25_top_k)

    fused = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        k=rrf_k,
    )

    return fused[:top_k]


# ---------------------------------------------------------------------------
# Index builder for LOCOMO memories
# ---------------------------------------------------------------------------

def build_bm25_index_from_db(conversation_id: Optional[str] = None) -> BM25Index:
    """Build a BM25 index from unified_memory rows (LOCOMO namespace).

    This mirrors the scoping logic in _scoped_recall: only rows whose
    content starts with '[LOCOMO Conv ...' are included.

    Call this once per evaluation run (or once per conversation_id) and
    reuse the index across questions.

    Args:
        conversation_id: If set, scope to a single LOCOMO conversation.

    Returns:
        Populated BM25Index ready for .search() calls.
    """
    from lib.db import get_connection

    conn = get_connection()
    cur = conn.cursor()

    if conversation_id:
        cur.execute(
            "SELECT id, content FROM unified_memory WHERE content LIKE %s",
            (f"[LOCOMO Conv {conversation_id}%",),
        )
    else:
        cur.execute(
            "SELECT id, content FROM unified_memory WHERE content LIKE '[LOCOMO Conv%%'"
        )

    index = BM25Index()
    batch = []
    for row in cur.fetchall():
        batch.append({"id": row[0], "content": row[1]})
    index.add_documents(batch)

    conn.close()
    return index


# ---------------------------------------------------------------------------
# Per-conversation index cache
# ---------------------------------------------------------------------------

_conv_index_cache: Dict[str, BM25Index] = {}


def get_or_build_index(conversation_id: str) -> BM25Index:
    """Return a cached BM25 index for a conversation, building if needed."""
    if conversation_id not in _conv_index_cache:
        _conv_index_cache[conversation_id] = build_bm25_index_from_db(conversation_id)
    return _conv_index_cache[conversation_id]


def clear_index_cache() -> None:
    """Clear cached BM25 indices (call between benchmark runs)."""
    _conv_index_cache.clear()


# ===========================================================================
# INTEGRATION EXAMPLE
# ===========================================================================
#
# To integrate into your memory adapter, replace the retrieval step in
# evaluate_question() with hybrid retrieval. Minimal diff:
#
#   from bm25_rrf import hybrid_search, get_or_build_index, clear_index_cache
#
#   def evaluate_question(question, use_cascade=True, model_tier="balanced"):
#       # Step 1a: Vector recall (existing)
#       t0 = time.time()
#       vector_results = _scoped_recall(
#           question.question,
#           limit=50,                # widen the vector net
#           conversation_id=question.conversation_id,
#       )
#       retrieval_ms = (time.time() - t0) * 1000
#
#       # Step 1b: BM25 + RRF fusion (new)
#       bm25_idx = get_or_build_index(question.conversation_id)
#       results = hybrid_search(
#           query=question.question,
#           vector_results=vector_results,
#           bm25_index=bm25_idx,
#           top_k=20,
#       )
#
#       # Step 2: Build context (same as before, but results are now fused)
#       context_parts = []
#       for r in results[:15]:
#           content = r["content"][:800] if r.get("content") else ""
#           score_label = f"rrf={r['rrf_score']:.4f}"
#           context_parts.append(f"[{score_label}] {content}")
#       context = "\n\n".join(context_parts)
#
#       # ... rest unchanged ...
#
#   # In run_evaluation(), add at the top:
#       clear_index_cache()
#
# The BM25 index for each conversation is built once and cached, so the
# overhead is only on the first question per conversation (~tens of ms for
# typical LOCOMO conversation sizes of 100-300 memories).
# ===========================================================================
