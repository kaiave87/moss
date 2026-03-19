"""
Tests for Multi-Channel Reciprocal Rank Fusion

Verifies the RRF math, BM25 indexing, hybrid search, and the
RRFSearchEngine with the MockDB backend.
"""

import asyncio
import pytest
from moss.rrf.bm25_rrf import (
    BM25Index,
    reciprocal_rank_fusion,
    hybrid_search,
    build_bm25_index,
)
from moss.rrf.db import MockDB
from moss.rrf.reciprocal_rank_fusion import RRFSearchEngine, RRFConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {"id": "doc1", "title": "Paris France", "content": "Paris is the capital city of France in Europe.", "path": "/docs/paris", "tags": []},
    {"id": "doc2", "title": "Berlin Germany", "content": "Berlin is the capital city of Germany in Europe.", "path": "/docs/berlin", "tags": []},
    {"id": "doc3", "title": "Tokyo Japan", "content": "Tokyo is the capital city of Japan in Asia.", "path": "/docs/tokyo", "tags": []},
    {"id": "doc4", "title": "Machine Learning", "content": "Machine learning is a subset of artificial intelligence.", "path": "/docs/ml", "tags": []},
    {"id": "doc5", "title": "Deep Learning", "content": "Deep learning uses neural networks with many layers.", "path": "/docs/dl", "tags": []},
]


@pytest.fixture
def bm25_index():
    return build_bm25_index(SAMPLE_DOCS)


@pytest.fixture
def mock_db():
    db = MockDB()
    db.store_documents(SAMPLE_DOCS)
    return db


# ---------------------------------------------------------------------------
# BM25 Tests
# ---------------------------------------------------------------------------

def test_bm25_index_size(bm25_index):
    assert bm25_index.size == len(SAMPLE_DOCS)


def test_bm25_search_returns_results(bm25_index):
    results = bm25_index.search("Paris France")
    assert len(results) > 0


def test_bm25_search_top_result_relevant(bm25_index):
    results = bm25_index.search("Paris France capital")
    assert results[0]["id"] == "doc1"


def test_bm25_search_no_match_returns_empty(bm25_index):
    # Query with only stopwords — should return empty
    results = bm25_index.search("the and or")
    assert len(results) == 0


def test_bm25_search_machine_learning(bm25_index):
    results = bm25_index.search("machine learning intelligence")
    ids = [r["id"] for r in results]
    assert "doc4" in ids


def test_bm25_search_respects_top_k(bm25_index):
    results = bm25_index.search("capital city Europe", top_k=2)
    assert len(results) <= 2


def test_bm25_results_have_rank_and_score(bm25_index):
    results = bm25_index.search("Japan Tokyo")
    for r in results:
        assert "rank" in r
        assert "score" in r
        assert r["score"] > 0


# ---------------------------------------------------------------------------
# RRF Math Tests
# ---------------------------------------------------------------------------

def test_rrf_basic_fusion():
    """Documents appearing in more lists should rank higher."""
    list1 = [{"id": "A", "rank": 1}, {"id": "B", "rank": 2}, {"id": "C", "rank": 3}]
    list2 = [{"id": "B", "rank": 1}, {"id": "A", "rank": 2}, {"id": "D", "rank": 3}]

    fused = reciprocal_rank_fusion([list1, list2])
    ids = [r["id"] for r in fused]

    # A appears rank 1 in list1, rank 2 in list2
    # B appears rank 2 in list1, rank 1 in list2
    # Both should beat C and D (appear in only one list)
    assert ids.index("A") < ids.index("C")
    assert ids.index("B") < ids.index("D")


def test_rrf_scores_are_positive():
    list1 = [{"id": "X", "rank": 1}, {"id": "Y", "rank": 2}]
    list2 = [{"id": "Y", "rank": 1}, {"id": "X", "rank": 2}]

    fused = reciprocal_rank_fusion([list1, list2])
    for r in fused:
        assert r["rrf_score"] > 0


def test_rrf_adds_rrf_rank():
    list1 = [{"id": "A"}, {"id": "B"}]
    fused = reciprocal_rank_fusion([list1])
    for i, r in enumerate(fused):
        assert r["rrf_rank"] == i + 1


def test_rrf_formula_correctness():
    """Verify exact RRF score for k=60."""
    # Single document at rank 1 in one list: score = 1/(60+1) ≈ 0.01639
    list1 = [{"id": "doc", "rank": 1}]
    fused = reciprocal_rank_fusion([list1], k=60)
    expected = 1.0 / (60 + 1)
    assert abs(fused[0]["rrf_score"] - expected) < 1e-10


def test_rrf_two_lists_same_doc():
    """Doc appearing in two lists at rank 1 should have 2x score of one list."""
    list1 = [{"id": "doc", "rank": 1}]
    list2 = [{"id": "doc", "rank": 1}]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    expected = 2.0 / (60 + 1)
    assert abs(fused[0]["rrf_score"] - expected) < 1e-10


def test_rrf_deduplicates_across_lists():
    """Same doc in two lists should appear once in output."""
    list1 = [{"id": "A"}, {"id": "B"}]
    list2 = [{"id": "A"}, {"id": "C"}]
    fused = reciprocal_rank_fusion([list1, list2])
    ids = [r["id"] for r in fused]
    assert ids.count("A") == 1


# ---------------------------------------------------------------------------
# Hybrid Search Tests
# ---------------------------------------------------------------------------

def test_hybrid_search_combines_results(bm25_index):
    vector_results = [
        {"id": "doc4", "content": "Machine learning is AI.", "score": 0.9, "rank": 1},
        {"id": "doc5", "content": "Deep learning uses neural nets.", "score": 0.7, "rank": 2},
    ]
    fused = hybrid_search("machine learning", vector_results, bm25_index, top_k=5)
    ids = [r["id"] for r in fused]
    # doc4 should rank high — it appears in both vector and BM25 results
    assert "doc4" in ids


def test_hybrid_search_respects_top_k(bm25_index):
    vector_results = [{"id": f"doc{i}", "content": f"doc {i}", "score": 0.5} for i in range(1, 6)]
    fused = hybrid_search("Paris capital", vector_results, bm25_index, top_k=2)
    assert len(fused) <= 2


# ---------------------------------------------------------------------------
# MockDB Tests
# ---------------------------------------------------------------------------

def test_mock_db_store_and_retrieve(mock_db):
    doc = mock_db.get_document("doc1")
    assert doc is not None
    assert doc["title"] == "Paris France"


def test_mock_db_text_search(mock_db):
    results = mock_db.search_by_text("Paris capital France")
    assert len(results) > 0
    assert results[0]["id"] == "doc1"


def test_mock_db_vector_search(mock_db):
    """Vector search should work when embeddings are provided."""
    import math
    db = MockDB()
    # Store docs with simple embeddings
    db.store_document({"id": "v1", "title": "cats", "content": "cats are animals",
                        "path": "/v1", "embedding": [1.0, 0.0, 0.0]})
    db.store_document({"id": "v2", "title": "dogs", "content": "dogs are animals",
                        "path": "/v2", "embedding": [0.9, 0.1, 0.0]})
    db.store_document({"id": "v3", "title": "cars", "content": "cars have engines",
                        "path": "/v3", "embedding": [0.0, 0.0, 1.0]})

    results = db.search_by_vector([1.0, 0.0, 0.0], limit=3)
    assert results[0]["id"] == "v1"
    assert results[1]["id"] == "v2"


def test_mock_db_update_retrieval_stats(mock_db):
    mock_db.update_retrieval_stats(["doc1", "doc2"])
    doc1 = mock_db.get_document("doc1")
    assert doc1["retrieval_count"] >= 1


# ---------------------------------------------------------------------------
# RRFSearchEngine Tests (async)
# ---------------------------------------------------------------------------

def test_rrf_engine_no_sources():
    """With no db configured, should return empty list."""
    engine = RRFSearchEngine()
    results = asyncio.run(engine.search("test query"))
    assert results == []


def test_rrf_engine_with_db_fts(mock_db):
    """Engine with db should return FTS results."""
    engine = RRFSearchEngine(db=mock_db)
    results = asyncio.run(engine.search("Paris France capital", max_results=3))
    assert len(results) > 0


def test_rrf_engine_results_sorted_by_score(mock_db):
    engine = RRFSearchEngine(db=mock_db)
    results = asyncio.run(engine.search("Europe capital city", max_results=5))
    scores = [r.rrf_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rrf_engine_with_vector_search(mock_db):
    """Engine with both db and embed_func should use vector search too."""
    from moss.hebbian.embeddings import get_embedding as embed_fn

    # Add embeddings to mock_db docs
    for doc in SAMPLE_DOCS:
        stored = mock_db.get_document(doc["id"])
        if stored:
            stored["embedding"] = embed_fn(doc["content"])

    engine = RRFSearchEngine(db=mock_db, embed_func=embed_fn)
    results = asyncio.run(engine.search("artificial intelligence", max_results=3))
    assert isinstance(results, list)


def test_rrf_engine_respects_max_results(mock_db):
    engine = RRFSearchEngine(db=mock_db)
    results = asyncio.run(engine.search("city", max_results=2))
    assert len(results) <= 2
