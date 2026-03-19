"""
Cross-Encoder Reranking for RRF Search

Reranks top-N candidates from RRF using a cross-encoder model.
Cross-encoders are more accurate than bi-encoders because they see
query and document together, enabling full attention over both.

Models (accuracy/speed tradeoff):
- cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, good accuracy (default)
- cross-encoder/ms-marco-MiniLM-L-12-v2: Better accuracy, slower
- BAAI/bge-reranker-base: Best accuracy, slowest

Usage:
    from moss.rrf.cross_encoder_reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranked = await reranker.rerank(query, candidates, top_k=10)

Performance:
    ~50-100ms for 50 candidates on GPU
    ~200-400ms on CPU
    Model loads lazily on first use.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

_cross_encoder = None
_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankedResult:
    """Result after cross-encoder reranking."""

    original_rank: int
    rerank_score: float
    doc_id: str
    title: str
    summary: Optional[str] = None
    path: Optional[str] = None
    original_score: float = 0.0


def _load_cross_encoder():
    """Lazy load cross-encoder model."""
    global _cross_encoder

    if _cross_encoder is not None:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading cross-encoder model: {_model_name}")
        start = time.time()
        _cross_encoder = CrossEncoder(
            _model_name,
            max_length=512,
            device="cuda"
        )
        elapsed = time.time() - start
        logger.info(f"Cross-encoder loaded in {elapsed:.1f}s")
        return _cross_encoder

    except ImportError:
        logger.warning("sentence-transformers not installed; cross-encoder unavailable")
        return None
    except Exception as e:
        logger.error(f"Failed to load cross-encoder: {e}")
        return None


class CrossEncoderReranker:
    """
    Cross-encoder reranker for search candidates.

    Takes RRF candidates and reranks using full query-document attention.
    Falls back gracefully if sentence-transformers is not installed.
    """

    def __init__(self, model_name: Optional[str] = None):
        global _model_name
        if model_name:
            _model_name = model_name
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            self._encoder = _load_cross_encoder()
        return self._encoder

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        score_threshold: float = 0.0,
    ) -> List[RerankedResult]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: Search query.
            candidates: List of dicts with "title", "summary", "doc_id".
            top_k: Number of results to return.
            score_threshold: Minimum rerank score to include.

        Returns:
            List of RerankedResult sorted by rerank_score descending.
        """
        if not candidates:
            return []

        encoder = self._get_encoder()
        if encoder is None:
            return self._fallback_results(candidates, top_k)

        start = time.time()
        pairs = []
        for cand in candidates:
            doc_text = cand.get("title", "")
            if cand.get("summary"):
                doc_text += " " + cand["summary"][:500]
            pairs.append([query, doc_text])

        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: encoder.predict(pairs, show_progress_bar=False)
        )

        results = []
        for i, (cand, score) in enumerate(zip(candidates, scores)):
            if score >= score_threshold:
                results.append(RerankedResult(
                    original_rank=i + 1,
                    rerank_score=float(score),
                    doc_id=cand.get("doc_id", cand.get("gold_id", "")),
                    title=cand.get("title", "Untitled"),
                    summary=cand.get("summary"),
                    path=cand.get("path", cand.get("canonical_path")),
                    original_score=cand.get("rrf_score", cand.get("score", 0.0)),
                ))

        results.sort(key=lambda x: x.rerank_score, reverse=True)
        elapsed_ms = (time.time() - start) * 1000
        logger.info(f"Reranked {len(candidates)} → {min(top_k, len(results))} in {elapsed_ms:.0f}ms")
        return results[:top_k]

    def _fallback_results(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> List[RerankedResult]:
        """Return candidates in original order when encoder unavailable."""
        results = []
        for i, cand in enumerate(candidates[:top_k]):
            results.append(RerankedResult(
                original_rank=i + 1,
                rerank_score=cand.get("rrf_score", cand.get("score", 0.0)),
                doc_id=cand.get("doc_id", cand.get("gold_id", "")),
                title=cand.get("title", "Untitled"),
                summary=cand.get("summary"),
                path=cand.get("path", cand.get("canonical_path")),
                original_score=cand.get("rrf_score", cand.get("score", 0.0)),
            ))
        return results


async def rerank_results(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Convenience function: rerank candidates and return as dicts.
    """
    reranker = CrossEncoderReranker()
    results = await reranker.rerank(query, candidates, top_k)
    return [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "summary": r.summary,
            "path": r.path,
            "rerank_score": r.rerank_score,
            "original_rank": r.original_rank,
            "original_score": r.original_score,
        }
        for r in results
    ]
