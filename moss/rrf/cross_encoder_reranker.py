"""
Cross-Encoder Reranking for Moss RRF Search

Reranks top-N candidates from RRF using a cross-encoder model.
Cross-encoders are more accurate than bi-encoders because they see
query and document together, enabling full attention.

Models (by accuracy/speed tradeoff):
- ms-marco-MiniLM-L-6-v2: Fast, good accuracy (default)
- ms-marco-MiniLM-L-12-v2: Better accuracy, slower
- BAAI/bge-reranker-base: Best accuracy, slowest

Integration:
    RRF returns 50 → Reranker picks top 10 → Return to user

Usage:
    from moss.rrf.cross_encoder_reranker import CrossEncoderReranker, rerank_results

    reranker = CrossEncoderReranker()
    reranked = await reranker.rerank(query, candidates, top_k=10)

Performance:
    - ~50-100ms for 50 candidates on GPU
    - ~200-400ms on CPU
    - Model loads lazily on first use
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# Lazy load to avoid startup delay
_cross_encoder = None
_model_name = "mixedbread-ai/mxbai-rerank-base-v1"


@dataclass
class RerankedResult:
    """Result after cross-encoder reranking."""

    original_rank: int
    rerank_score: float
    gold_id: str
    title: str
    summary: Optional[str] = None
    canonical_path: Optional[str] = None
    original_score: float = 0.0  # RRF score


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
            device="cuda"  # Will fallback to CPU if no CUDA
        )

        elapsed = time.time() - start
        logger.info(f"Cross-encoder loaded in {elapsed:.1f}s")

        return _cross_encoder

    except ImportError:
        logger.error("sentence-transformers not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to load cross-encoder: {e}")
        return None


class CrossEncoderReranker:
    """
    Cross-encoder reranker for search results.

    Takes RRF candidates and reranks using full query-document attention.
    """

    def __init__(self, model_name: Optional[str] = None):
        global _model_name
        if model_name:
            _model_name = model_name
        self._encoder = None

    def _get_encoder(self):
        """Get or load the cross-encoder."""
        if self._encoder is None:
            self._encoder = _load_cross_encoder()
        return self._encoder

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        score_threshold: float = 0.0
    ) -> List[RerankedResult]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: Search query
            candidates: List of dicts with 'title', 'summary', 'gold_id', etc.
            top_k: Number of results to return
            score_threshold: Minimum rerank score to include

        Returns:
            List of RerankedResult sorted by rerank_score
        """
        if not candidates:
            return []

        encoder = self._get_encoder()
        if encoder is None:
            logger.warning("Cross-encoder not available, returning original order")
            return self._fallback_results(candidates, top_k)

        start = time.time()

        # Build query-document pairs
        pairs = []
        for cand in candidates:
            # Handle both dict and object (RecallResult) types
            if isinstance(cand, dict):
                doc_text = cand.get('title', '')
                summary = cand.get('summary', '')
                content = cand.get('content', '')
            else:
                doc_text = getattr(cand, 'title', '') or ''
                summary = getattr(cand, 'summary', '') or ''
                content = getattr(cand, 'content', '') or ''
            # Prefer content (full text) for ranking accuracy, fall back to summary
            if content:
                doc_text = content[:800]
            elif summary:
                doc_text = (doc_text + " " + summary)[:800]
            pairs.append([query, doc_text])

        # Run cross-encoder (CPU-bound, run in thread pool)
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: encoder.predict(pairs, show_progress_bar=False)
        )

        # Build results with scores
        results = []
        for i, (cand, score) in enumerate(zip(candidates, scores)):
            if score >= score_threshold:
                # Handle both dict and object types
                if isinstance(cand, dict):
                    gold_id = cand.get('gold_id', '')
                    title = cand.get('title', 'Untitled')
                    summary_val = cand.get('summary')
                    path = cand.get('canonical_path') or cand.get('path')
                    orig_score = cand.get('rrf_score', 0.0) or cand.get('score', 0.0)
                else:
                    gold_id = getattr(cand, 'gold_id', '') or getattr(cand, 'memory_id', '')
                    title = getattr(cand, 'title', 'Untitled') or ''
                    summary_val = getattr(cand, 'summary', None)
                    path = getattr(cand, 'canonical_path', None)
                    orig_score = getattr(cand, 'rrf_score', 0.0) or getattr(cand, 'score', 0.0)
                results.append(RerankedResult(
                    original_rank=i,  # 0-indexed to match list indexing
                    rerank_score=float(score),
                    gold_id=gold_id,
                    title=title,
                    summary=summary_val,
                    canonical_path=path,
                    original_score=orig_score,
                ))

        # Sort by rerank score descending
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            f"Reranked {len(candidates)} → {min(top_k, len(results))} results "
            f"in {elapsed_ms:.0f}ms"
        )

        return results[:top_k]

    def _fallback_results(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int
    ) -> List[RerankedResult]:
        """Fallback when cross-encoder unavailable."""
        results = []
        for i, cand in enumerate(candidates[:top_k]):
            if isinstance(cand, dict):
                orig_score = cand.get('rrf_score', 0.0) or cand.get('score', 0.0)
                gold_id = cand.get('gold_id', '')
                title = cand.get('title', 'Untitled')
                summary_val = cand.get('summary')
                path = cand.get('canonical_path') or cand.get('path')
            else:
                orig_score = getattr(cand, 'rrf_score', 0.0) or getattr(cand, 'score', 0.0)
                gold_id = getattr(cand, 'gold_id', '') or getattr(cand, 'memory_id', '')
                title = getattr(cand, 'title', 'Untitled') or ''
                summary_val = getattr(cand, 'summary', None)
                path = getattr(cand, 'canonical_path', None)
            results.append(RerankedResult(
                original_rank=i,
                rerank_score=orig_score,
                gold_id=gold_id,
                title=title,
                summary=summary_val,
                canonical_path=path,
                original_score=orig_score
            ))
        return results

    async def rerank_with_ollama(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        model: str = "phi3.5:latest"
    ) -> List[RerankedResult]:
        """
        Alternative: Rerank using Ollama model (slower but no extra deps).

        Uses LLM to score relevance of each candidate.
        """
        import httpx

        if not candidates:
            return []

        start = time.time()
        results = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for i, cand in enumerate(candidates[:30]):  # Limit to 30 for speed
                doc_text = cand.get('title', '')
                if cand.get('summary'):
                    doc_text += ": " + cand['summary'][:200]

                prompt = f"""Rate relevance of this document to the query.
Query: {query}
Document: {doc_text}

Reply with ONLY a number 0-10 (10=highly relevant, 0=not relevant)."""

                try:
                    resp = await client.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {"num_ctx": 512, "temperature": 0.1}
                        }
                    )

                    if resp.status_code == 200:
                        text = resp.json().get("response", "0").strip()
                        # Extract number from response
                        import re
                        match = re.search(r'\d+', text)
                        score = int(match.group()) if match else 0
                        score = min(10, max(0, score)) / 10.0  # Normalize to 0-1
                    else:
                        score = 0.5

                except Exception as e:
                    logger.warning(f"Ollama rerank failed for doc {i}: {e}")
                    score = 0.5

                results.append(RerankedResult(
                    original_rank=i + 1,
                    rerank_score=score,
                    gold_id=cand.get('gold_id', ''),
                    title=cand.get('title', 'Untitled'),
                    summary=cand.get('summary'),
                    canonical_path=cand.get('canonical_path') or cand.get('path'),
                    original_score=cand.get('rrf_score', 0.0)
                ))

        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        elapsed_ms = (time.time() - start) * 1000
        logger.info(f"Ollama reranked {len(candidates)} → {top_k} in {elapsed_ms:.0f}ms")

        return results[:top_k]


# Convenience function
async def rerank_results(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 10,
    use_ollama: bool = False
) -> List[Dict[str, Any]]:
    """
    Quick rerank function for integration.

    Returns list of dicts for easy serialization.
    """
    reranker = CrossEncoderReranker()

    if use_ollama:
        results = await reranker.rerank_with_ollama(query, candidates, top_k)
    else:
        results = await reranker.rerank(query, candidates, top_k)

    return [
        {
            "gold_id": r.gold_id,
            "title": r.title,
            "summary": r.summary,
            "path": r.canonical_path,
            "rerank_score": r.rerank_score,
            "original_rank": r.original_rank,
            "original_score": r.original_score
        }
        for r in results
    ]
