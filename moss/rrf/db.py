"""
Mock Database Layer for RRF Search Engine

Implements the storage interface expected by RRFSearchEngine using
in-memory Python dicts. Suitable for testing, prototyping, and
demonstrating the retrieval algorithms without a real database.

For production use, replace MockDB with a backend that wraps
your actual storage layer (PostgreSQL + pgvector, Elasticsearch, etc.).
The interface methods are:
    - store_document(doc: dict) -> None
    - search_by_text(query: str, limit: int) -> List[dict]
    - search_by_vector(query_vec: List[float], limit: int) -> List[dict]
    - get_document(doc_id: str) -> Optional[dict]
    - update_retrieval_stats(doc_ids: List[str]) -> None
"""

import math
import re
import string
from typing import Any, Dict, List, Optional


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _simple_text_score(query: str, content: str) -> float:
    """Simple keyword overlap score for mock FTS."""
    stop = {"a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "for", "and", "or"}
    q_words = set(
        w.strip(string.punctuation).lower()
        for w in query.split()
        if w.strip(string.punctuation).lower() not in stop
    )
    c_words = set(
        w.strip(string.punctuation).lower()
        for w in content.split()
        if w.strip(string.punctuation).lower() not in stop
    )
    if not q_words:
        return 0.0
    overlap = len(q_words & c_words)
    return overlap / len(q_words)


class MockDB:
    """
    In-memory mock database for RRF and Hebbian testing.

    Stores documents with optional pre-computed embeddings.
    Text search uses keyword overlap scoring.
    Vector search uses cosine similarity.

    Usage:
        db = MockDB()
        db.store_document({
            "id": "doc1",
            "title": "My Document",
            "content": "Some content here",
            "path": "/docs/doc1",
            "embedding": [0.1, 0.2, ...],  # optional
        })

        results = db.search_by_text("content", limit=5)
        results = db.search_by_vector(query_embedding, limit=5)
    """

    def __init__(self):
        # doc_id -> document dict
        self._documents: Dict[str, Dict[str, Any]] = {}

    def store_document(self, doc: Dict[str, Any]) -> None:
        """Store a document. Must have an "id" field."""
        doc_id = str(doc.get("id", ""))
        if not doc_id:
            raise ValueError("Document must have an 'id' field")
        self._documents[doc_id] = dict(doc)

    def store_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Store multiple documents at once."""
        for doc in docs:
            self.store_document(doc)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID."""
        return self._documents.get(str(doc_id))

    def search_by_text(
        self,
        query: str,
        limit: int = 50,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Text search using keyword overlap scoring.

        Returns documents sorted by score descending, with "score" field added.
        """
        scored = []
        for doc in self._documents.values():
            content = doc.get("content", "") + " " + doc.get("title", "")
            score = _simple_text_score(query, content)
            if score > 0:
                result = dict(doc)
                result["score"] = score
                scored.append(result)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def search_by_vector(
        self,
        query_vec: List[float],
        limit: int = 50,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Vector search using cosine similarity.

        Only returns documents that have an "embedding" field.
        Returns documents sorted by similarity descending, with "similarity" field.
        """
        scored = []
        for doc in self._documents.values():
            emb = doc.get("embedding")
            if emb is None:
                continue
            sim = _cosine_similarity(query_vec, emb)
            if sim > 0:
                result = dict(doc)
                result["similarity"] = sim
                scored.append(result)

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:limit]

    def update_retrieval_stats(self, doc_ids: List[str]) -> None:
        """Increment retrieval_count for the given document IDs."""
        for doc_id in doc_ids:
            if doc_id in self._documents:
                self._documents[doc_id]["retrieval_count"] = (
                    self._documents[doc_id].get("retrieval_count", 0) + 1
                )

    def count(self) -> int:
        """Return total number of stored documents."""
        return len(self._documents)

    def clear(self) -> None:
        """Remove all stored documents."""
        self._documents.clear()
