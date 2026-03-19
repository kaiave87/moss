"""
Tests for Multi-Channel RRF retrieval module.
"""

from moss.rrf.reciprocal_rank_fusion import fuse, fuse_ranked, RRFConfig, FusedResult
from moss.rrf.bm25_rrf import tokenize, BM25Index, reciprocal_rank_fusion, hybrid_search


# ============================================================
# RRF core (reciprocal_rank_fusion.py)
# ============================================================

def test_fuse_single_channel():
    """Single channel should pass through rankings (highest score = rank 1)."""
    channel = [("id_a", 1.0), ("id_b", 0.8), ("id_c", 0.5)]
    results = fuse({"semantic": channel})
    ids = [r.doc_id for r in results]
    assert ids[0] == "id_a"


def test_fuse_two_channels_merge():
    """Two channels should be merged — all doc IDs present in output."""
    vec = [("a", 0.9), ("b", 0.7), ("c", 0.3)]
    bm25 = [("c", 0.95), ("a", 0.6), ("b", 0.4)]
    results = fuse({"semantic": vec, "lexical": bm25})
    assert len(results) >= 3
    assert all(isinstance(r, FusedResult) for r in results)


def test_fuse_empty_channels():
    results = fuse({})
    assert results == []


def test_fuse_top_k():
    channel = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]
    results = fuse({"semantic": channel}, top_k=2)
    assert len(results) == 2


def test_fuse_scores_descending():
    """Fused results must be sorted by rrf_score descending."""
    vec = [("a", 0.9), ("b", 0.7), ("c", 0.3)]
    bm25 = [("b", 0.95), ("a", 0.5), ("c", 0.2)]
    results = fuse({"semantic": vec, "lexical": bm25})
    scores = [r.rrf_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_fuse_channel_weights():
    """Higher channel weight on one channel should rank that channel's exclusive top item
    above an exclusive top item from a lower-weight channel."""
    # Each item appears in only ONE channel:
    vec = [("vec_exclusive", 0.9), ("vec_second", 0.5)]
    bm25 = [("bm25_exclusive", 0.9), ("bm25_second", 0.5)]
    results = fuse(
        {"semantic": vec, "lexical": bm25},
        channel_weights={"semantic": 10.0, "lexical": 1.0},
    )
    ids = [r.doc_id for r in results]
    # vec_exclusive should rank above bm25_exclusive (10x weight difference)
    assert ids.index("vec_exclusive") < ids.index("bm25_exclusive")


def test_fuse_channel_provenance():
    """FusedResult.channels should record where each doc came from."""
    vec = [("a", 0.9)]
    bm25 = [("b", 0.8), ("a", 0.5)]
    results = fuse({"semantic": vec, "lexical": bm25})
    a_result = next(r for r in results if r.doc_id == "a")
    assert "semantic" in a_result.channels
    assert "lexical" in a_result.channels


def test_fuse_ranked_basic():
    """fuse_ranked takes ranked ID lists (no scores)."""
    list1 = ["a", "b", "c"]
    list2 = ["b", "a", "d"]
    results = fuse_ranked({"l1": list1, "l2": list2})
    ids = [r[0] for r in results]
    # 'a' and 'b' appear in both — should rank above 'c' and 'd'
    assert ids.index("a") < ids.index("c")
    assert ids.index("b") < ids.index("d")


def test_fuse_ranked_returns_tuples():
    results = fuse_ranked({"ch": ["x", "y", "z"]})
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


# ============================================================
# BM25 (bm25_rrf.py)
# ============================================================

def test_tokenize_basic():
    tokens = tokenize("The quick brown fox")
    assert "quick" in tokens
    assert "fox" in tokens
    # Common stopwords should be removed
    assert "the" not in tokens


def test_tokenize_empty():
    assert tokenize("") == []


def _make_index(*texts):
    """Helper: build a BM25Index from text strings."""
    index = BM25Index()
    docs = [{"id": f"m{i}", "content": t} for i, t in enumerate(texts)]
    index.add_documents(docs)
    return index


def test_bm25_index_build_and_search():
    index = _make_index(
        "the cat sat on the mat",
        "the dog ran in the park",
        "cats and dogs are pets",
    )
    results = index.search("cat", top_k=3)
    assert len(results) >= 1
    ids = [r["id"] for r in results]
    # "m0" (cat sat) or "m2" (cats) should match
    assert "m0" in ids or "m2" in ids


def test_bm25_index_top_k_limit():
    index = _make_index(*[f"word{i} common" for i in range(10)])
    results = index.search("common", top_k=3)
    assert len(results) <= 3


def test_bm25_index_empty_query():
    index = _make_index("content", "stuff")
    results = index.search("", top_k=5)
    assert isinstance(results, list)


def test_bm25_index_no_match():
    index = _make_index("cat dog fish")
    results = index.search("quantum entanglement", top_k=5)
    # BM25 returns zero-score results filtered out
    assert isinstance(results, list)


def test_bm25_index_size():
    index = _make_index("a", "b", "c")
    assert index.size == 3


def test_rrf_function_basic():
    """reciprocal_rank_fusion() standalone function."""
    lists = [
        [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.6}],
        [{"id": "b", "score": 0.8}, {"id": "a", "score": 0.5}, {"id": "c", "score": 0.3}],
    ]
    results = reciprocal_rank_fusion(lists, k=60)
    ids = [r["id"] for r in results]
    assert "a" in ids
    assert "b" in ids
    assert "c" in ids
    # b appears in both lists — ranked higher than c (one list only)
    assert ids.index("b") < ids.index("c")


def test_rrf_function_rrf_score_present():
    lists = [[{"id": "x", "score": 1.0}]]
    results = reciprocal_rank_fusion(lists)
    assert "rrf_score" in results[0]
    assert results[0]["rrf_score"] > 0.0


def test_hybrid_search_basic():
    """hybrid_search with an in-memory BM25 index."""
    docs = [
        {"id": "id1", "content": "The Eiffel Tower is in Paris"},
        {"id": "id2", "content": "The Colosseum is in Rome"},
        {"id": "id3", "content": "Paris is the capital of France"},
    ]
    index = BM25Index()
    index.add_documents(docs)

    vector_results = [
        {"id": "id3", "content": docs[2]["content"], "score": 0.9},
        {"id": "id1", "content": docs[0]["content"], "score": 0.7},
    ]

    results = hybrid_search(
        query="Paris France",
        vector_results=vector_results,
        bm25_index=index,
        top_k=3,
    )
    ids = [r["id"] for r in results]
    assert "id3" in ids or "id1" in ids
