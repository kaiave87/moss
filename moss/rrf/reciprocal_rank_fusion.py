"""
Reciprocal Rank Fusion (RRF) — Multi-Channel Hybrid Search

Combines results from multiple retrieval sources into a unified ranking
using Reciprocal Rank Fusion (Cormack et al., 2009).

References:
    Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
    "Reciprocal Rank Fusion outperforms Condorcet and individual Relevance
    Feedback Methods." SIGIR 2009.

    k=60 is empirically optimal (standard in the literature).

Patent pending — Lichen Research Inc., Canadian application filed March 2026.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Unified search result from any retrieval source."""

    # Identity
    doc_id: str
    title: str
    path: str

    # Content
    summary: Optional[str] = None
    snippet: Optional[str] = None

    # Scoring
    source: str = "unknown"        # fts, vector, graph, reranker
    rank: int = 0                  # Original rank in source (1-indexed)
    raw_score: float = 0.0        # Original score from source
    rrf_score: float = 0.0        # Computed RRF score

    # Metadata
    doc_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)

    # Provenance
    retrieval_count: int = 0
    last_retrieved: Optional[datetime] = None


@dataclass
class RRFConfig:
    """Configuration for RRF fusion."""

    k: int = 60                        # RRF constant (higher = more democratic)

    # Source weights (weighted RRF variant)
    fts_weight: float = 1.0            # Full-text search
    vector_weight: float = 1.0         # Semantic similarity
    graph_weight: float = 0.8          # Graph traversal (exploratory, lower weight)
    reranker_weight: float = 1.5       # Cross-encoder reranker (most accurate)

    # Retrieval limits per source
    fts_limit: int = 50
    vector_limit: int = 50
    graph_limit: int = 30

    # Reranking
    enable_reranker: bool = False
    reranker_top_k: int = 20

    # Diversity (Maximal Marginal Relevance)
    enable_mmr: bool = False
    mmr_lambda: float = 0.7            # 1.0 = pure relevance, 0.0 = pure diversity


class RRFSearchEngine:
    """
    Unified search engine using Reciprocal Rank Fusion.

    Combines multiple retrieval sources into a single ranked list.

    RRF formula (Cormack 2009):
        RRF(d) = Σ (1 / (k + rank_s(d)))

        Where:
        - d = document
        - s = retrieval source (fts, vector, graph)
        - rank_s(d) = rank of document d in source s
        - k = constant (typically 60)

    Weighted variant:
        RRF_weighted(d) = Σ w_s * (1 / (k + rank_s(d)))

    This implementation accepts pluggable retrieval backends via
    the db and embed_func arguments. For testing, use MockDB from
    moss.rrf.db.

    Patent pending — Canadian application filed March 2026.
    """

    def __init__(
        self,
        db: Any = None,            # Database backend (see moss.rrf.db.MockDB)
        embed_func: Any = None,    # Embedding function: (str) -> List[float]
        config: Optional[RRFConfig] = None
    ):
        self.db = db
        self.embed_func = embed_func
        self.config = config or RRFConfig()

    async def search(
        self,
        query: str,
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Execute hybrid search with RRF fusion.

        Args:
            query: Search query string
            max_results: Number of results to return
            filters: Optional dict with filter criteria (doc_type, tags, etc.)

        Returns:
            Ranked list of SearchResult objects
        """
        start_time = datetime.now()
        tasks = []

        if self.db is not None:
            tasks.append(self._search_fts(query, filters))
            if self.embed_func is not None:
                tasks.append(self._search_vector(query, filters))

        if not tasks:
            logger.warning("No search sources configured")
            return []

        source_results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for i, result in enumerate(source_results):
            if isinstance(result, Exception):
                logger.warning(f"Source {i} failed: {result}")
            else:
                valid_results.append(result)

        if not valid_results:
            return []

        fused_results = self._apply_rrf(valid_results)

        if self.config.enable_reranker and fused_results:
            fused_results = await self._rerank(query, fused_results)

        if self.config.enable_mmr and len(fused_results) > max_results:
            fused_results = self._apply_mmr(fused_results, max_results)

        final_results = fused_results[:max_results]

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            f"RRF search: {len(final_results)} results, "
            f"{len(valid_results)} sources, {elapsed_ms:.1f}ms"
        )

        return final_results

    async def _search_fts(
        self,
        query: str,
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Full-text search via the database backend."""
        if self.db is None:
            return []
        rows = self.db.search_by_text(query, limit=self.config.fts_limit)
        results = []
        for i, row in enumerate(rows):
            results.append(SearchResult(
                doc_id=str(row.get("id", f"fts_{i}")),
                title=row.get("title", "Untitled"),
                path=row.get("path", ""),
                summary=row.get("summary"),
                source="fts",
                rank=i + 1,
                raw_score=float(row.get("score", 0.0)),
                doc_type=row.get("doc_type"),
                tags=row.get("tags", []),
                topics=row.get("topics", []),
                retrieval_count=row.get("retrieval_count", 0),
            ))
        return results

    async def _search_vector(
        self,
        query: str,
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Semantic vector search via the database backend."""
        if self.db is None or self.embed_func is None:
            return []

        query_vec = self.embed_func(query)
        rows = self.db.search_by_vector(query_vec, limit=self.config.vector_limit)
        results = []
        for i, row in enumerate(rows):
            results.append(SearchResult(
                doc_id=str(row.get("id", f"vec_{i}")),
                title=row.get("title", "Untitled"),
                path=row.get("path", ""),
                summary=row.get("summary"),
                source="vector",
                rank=i + 1,
                raw_score=float(row.get("similarity", 0.0)),
                doc_type=row.get("doc_type"),
                tags=row.get("tags", []),
                topics=row.get("topics", []),
                retrieval_count=row.get("retrieval_count", 0),
            ))
        return results

    def _apply_rrf(
        self,
        source_results: List[List[SearchResult]]
    ) -> List[SearchResult]:
        """
        Apply Reciprocal Rank Fusion across source result lists.

        Weighted RRF score(d) = Σ w_s * (1 / (k + rank_s(d)))
        """
        k = self.config.k
        weights = {
            "fts": self.config.fts_weight,
            "vector": self.config.vector_weight,
            "graph": self.config.graph_weight,
            "reranker": self.config.reranker_weight,
        }

        doc_scores: Dict[str, float] = {}
        doc_data: Dict[str, SearchResult] = {}

        for source_list in source_results:
            for result in source_list:
                doc_id = result.doc_id
                source = result.source
                rank = result.rank
                weight = weights.get(source, 1.0)

                # Core RRF formula
                rrf_contribution = weight * (1.0 / (k + rank))

                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_data[doc_id] = result
                doc_scores[doc_id] += rrf_contribution

        ranked = []
        for doc_id, rrf_score in sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        ):
            result = doc_data[doc_id]
            result.rrf_score = rrf_score
            ranked.append(result)

        return ranked

    async def _rerank(
        self,
        query: str,
        candidates: List[SearchResult]
    ) -> List[SearchResult]:
        """Optional cross-encoder reranking step."""
        top_candidates = candidates[:self.config.reranker_top_k]
        try:
            from moss.rrf.cross_encoder_reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker()
            cand_dicts = [
                {
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "summary": c.summary,
                    "path": c.path,
                    "rrf_score": c.rrf_score,
                }
                for c in top_candidates
            ]
            reranked = await reranker.rerank(query, cand_dicts, top_k=len(top_candidates))
            id_to_result = {c.doc_id: c for c in top_candidates}
            result_list = []
            for r in reranked:
                gid = r.get("doc_id", "")
                if gid in id_to_result:
                    orig = id_to_result[gid]
                    orig.rrf_score = (orig.rrf_score + r.get("rerank_score", 0) / 10.0) / 2
                    orig.source = "reranker"
                    result_list.append(orig)
            reranked_ids = {r.get("doc_id") for r in reranked}
            for c in candidates:
                if c.doc_id not in reranked_ids:
                    result_list.append(c)
            return result_list
        except ImportError:
            logger.warning("Cross-encoder reranker not available")
            return candidates
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return candidates

    def _apply_mmr(
        self,
        results: List[SearchResult],
        target_count: int
    ) -> List[SearchResult]:
        """
        Maximal Marginal Relevance for result diversity.

        MMR = λ * relevance - (1-λ) * max_similarity_to_selected
        """
        if len(results) <= target_count:
            return results

        selected = [results[0]]
        remaining = results[1:]

        while len(selected) < target_count and remaining:
            mmr_scores = []
            for candidate in remaining:
                relevance = candidate.rrf_score
                max_sim = 0.0
                for selected_item in selected:
                    shared = len(
                        set(candidate.path.split("/")) &
                        set(selected_item.path.split("/"))
                    )
                    total = len(set(candidate.path.split("/")))
                    sim = shared / total if total > 0 else 0.0
                    max_sim = max(max_sim, sim)
                mmr = (
                    self.config.mmr_lambda * relevance -
                    (1 - self.config.mmr_lambda) * max_sim
                )
                mmr_scores.append((candidate, mmr))
            best_candidate, _ = max(mmr_scores, key=lambda x: x[1])
            selected.append(best_candidate)
            remaining.remove(best_candidate)

        return selected


def reciprocal_rank_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    id_key: str = "id",
) -> List[Dict[str, Any]]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for a document d = Σ 1 / (k + rank_i(d))

    Args:
        result_lists: List of ranked result lists. Each result must contain
                      the id_key field for deduplication. Optional "rank"
                      field (1-indexed); if absent, list position is used.
        k: RRF constant (default 60, per Cormack 2009).
        id_key: Key identifying unique documents across lists.

    Returns:
        Fused list sorted by rrf_score descending, with "rrf_score" and
        "rrf_rank" fields added to each result dict.
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
