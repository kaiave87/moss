"""
Reciprocal Rank Fusion (RRF) for Hybrid Memory Retrieval

Combines results from multiple retrieval channels (lexical, semantic, graph)
into a unified ranking via weighted RRF.

References:
    Cormack et al. (2009) "Reciprocal Rank Fusion outperforms Condorcet and
    individual relevance methods" — k=60 is empirically optimal.

RRF formula:
    score(d) = Σ w_s * (1 / (k + rank_s(d)))

    Where:
    - d = document
    - s = retrieval source (lexical, vector, graph, reranker)
    - rank_s(d) = rank of document d in source s (1-indexed)
    - k = smoothing constant (default 60)
    - w_s = per-channel weight

Usage:
    from moss.rrf.reciprocal_rank_fusion import fuse, RRFConfig

    # Each channel returns a list of (doc_id, score) pairs, ordered by score
    lexical_results  = [("doc1", 0.9), ("doc3", 0.7), ("doc2", 0.5)]
    semantic_results = [("doc2", 0.95), ("doc1", 0.8), ("doc4", 0.6)]

    config = RRFConfig(lexical_weight=1.0, semantic_weight=1.0)
    fused = fuse(
        channels={"lexical": lexical_results, "semantic": semantic_results},
        config=config,
        top_k=10,
    )
    # Returns list of (doc_id, rrf_score) sorted by rrf_score descending
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A ranked result list from a single retrieval channel.
# Each entry is (document_id, raw_score). Order determines rank.
ChannelResults = List[Tuple[str, float]]


@dataclass
class RRFConfig:
    """Configuration for weighted RRF fusion."""

    # RRF smoothing constant — higher = more democratic across ranks
    # k=60 is empirically optimal (Cormack et al. 2009)
    k: int = 60

    # Per-channel weights (higher = more influence on final ranking)
    lexical_weight: float = 1.0     # BM25 / keyword search
    semantic_weight: float = 1.0    # Dense vector similarity
    graph_weight: float = 0.8       # Graph / entity traversal
    temporal_weight: float = 1.0    # Date-aware temporal search
    reranker_weight: float = 1.5    # Cross-encoder reranker (most precise)

    # Minimum RRF score threshold (0.0 = no filter)
    min_score: float = 0.0


@dataclass
class FusedResult:
    """A document with its fused RRF score and channel provenance."""

    doc_id: str
    rrf_score: float
    # Per-channel contributions: channel_name -> (rank, raw_score, weighted_rrf_contribution)
    channels: Dict[str, Tuple[int, float, float]] = field(default_factory=dict)

    def __repr__(self) -> str:
        channel_str = ", ".join(
            f"{ch}@{rank}={contrib:.4f}"
            for ch, (rank, _, contrib) in sorted(self.channels.items())
        )
        return f"FusedResult(doc_id={self.doc_id!r}, rrf={self.rrf_score:.4f}, [{channel_str}])"


# ---------------------------------------------------------------------------
# Core fusion function
# ---------------------------------------------------------------------------

def fuse(
    channels: Dict[str, ChannelResults],
    config: Optional[RRFConfig] = None,
    top_k: Optional[int] = None,
    channel_weights: Optional[Dict[str, float]] = None,
) -> List[FusedResult]:
    """
    Apply weighted Reciprocal Rank Fusion across multiple retrieval channels.

    Args:
        channels: Mapping of channel_name -> list of (doc_id, raw_score).
                  Each list should be ordered by score (best first).
                  Channel names: "lexical", "semantic", "graph", "temporal",
                  "reranker", or any custom name.
        config: RRF configuration (weights, k constant). Defaults to RRFConfig().
        top_k: Return only the top-k results. None = return all.
        channel_weights: Override per-channel weights for this call only.
                         Merged with config weights (call-level takes precedence).

    Returns:
        List of FusedResult, sorted by rrf_score descending.
    """
    cfg = config or RRFConfig()
    k = cfg.k

    # Build effective weight map
    default_weights = {
        "lexical": cfg.lexical_weight,
        "semantic": cfg.semantic_weight,
        "graph": cfg.graph_weight,
        "temporal": cfg.temporal_weight,
        "reranker": cfg.reranker_weight,
    }
    effective_weights = {**default_weights, **(channel_weights or {})}

    # Accumulate RRF scores
    doc_scores: Dict[str, float] = {}
    doc_channels: Dict[str, Dict[str, Tuple[int, float, float]]] = {}

    for channel_name, results in channels.items():
        if not results:
            continue
        weight = effective_weights.get(channel_name, 1.0)

        for rank_0, (doc_id, raw_score) in enumerate(results):
            rank_1 = rank_0 + 1  # 1-indexed
            contribution = weight / (k + rank_1)

            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + contribution
            if doc_id not in doc_channels:
                doc_channels[doc_id] = {}
            doc_channels[doc_id][channel_name] = (rank_1, raw_score, contribution)

    # Build and sort results
    fused = [
        FusedResult(
            doc_id=doc_id,
            rrf_score=score,
            channels=doc_channels.get(doc_id, {}),
        )
        for doc_id, score in doc_scores.items()
        if score >= cfg.min_score
    ]
    fused.sort(key=lambda r: r.rrf_score, reverse=True)

    if top_k is not None:
        fused = fused[:top_k]

    logger.debug(
        "RRF fusion: %d channels, %d unique docs, %d results returned",
        len(channels),
        len(doc_scores),
        len(fused),
    )

    return fused


# ---------------------------------------------------------------------------
# Convenience: fuse raw result lists (no doc_id/score tuple wrapping needed)
# ---------------------------------------------------------------------------

def fuse_ranked(
    channels: Dict[str, List[str]],
    config: Optional[RRFConfig] = None,
    top_k: Optional[int] = None,
    channel_weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, float]]:
    """
    Simplified RRF for channels that provide only ranked doc_id lists
    (no raw scores). Assigns synthetic scores based on rank position.

    Args:
        channels: Mapping of channel_name -> ordered list of doc_ids.
        config, top_k, channel_weights: Same as fuse().

    Returns:
        List of (doc_id, rrf_score) sorted by rrf_score descending.
    """
    scored_channels: Dict[str, ChannelResults] = {
        name: [(doc_id, 1.0 / (i + 1)) for i, doc_id in enumerate(docs)]
        for name, docs in channels.items()
    }
    results = fuse(scored_channels, config=config, top_k=top_k,
                   channel_weights=channel_weights)
    return [(r.doc_id, r.rrf_score) for r in results]
