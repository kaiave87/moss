"""Multi-Channel Reciprocal Rank Fusion.

Combines results from multiple retrieval channels (BM25, vector, graph,
temporal, session) into unified ranking via Reciprocal Rank Fusion.

Reference: Cormack et al. (2009) - k=60 empirically optimal.
"""

from .bm25_rrf import BM25Index, reciprocal_rank_fusion, hybrid_search, tokenize

__all__ = ["BM25Index", "reciprocal_rank_fusion", "hybrid_search", "tokenize"]
