"""Tests for Multi-Channel RRF module."""

from moss.rrf import BM25Index, reciprocal_rank_fusion, hybrid_search, tokenize


def test_tokenize_basic():
    tokens = tokenize("What is the fastest way to learn Python?")
    assert "fastest" in tokens
    assert "python" in tokens
    assert "is" not in tokens  # stopword
    assert "the" not in tokens  # stopword


def test_bm25_index_basic():
    idx = BM25Index()
    idx.add_documents([
        {"id": "1", "content": "Alice started her new job at the tech company"},
        {"id": "2", "content": "Bob went to the grocery store yesterday"},
        {"id": "3", "content": "Alice met Bob at the tech conference"},
    ])
    results = idx.search("Alice tech job")
    assert len(results) > 0
    assert results[0]["id"] in ("1", "3")


def test_bm25_index_empty():
    idx = BM25Index()
    assert idx.search("anything") == []
    assert idx.size == 0


def test_reciprocal_rank_fusion_basic():
    list_a = [
        {"id": "1", "score": 0.9, "rank": 1},
        {"id": "2", "score": 0.7, "rank": 2},
        {"id": "3", "score": 0.5, "rank": 3},
    ]
    list_b = [
        {"id": "2", "score": 0.95, "rank": 1},
        {"id": "4", "score": 0.6, "rank": 2},
        {"id": "1", "score": 0.3, "rank": 3},
    ]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    # Doc "2" appears rank 2 in A and rank 1 in B -> should be top
    # Doc "1" appears rank 1 in A and rank 3 in B -> should be second
    assert fused[0]["id"] in ("1", "2")
    assert len(fused) == 4  # 4 unique docs


def test_reciprocal_rank_fusion_single_list():
    results = [{"id": str(i), "rank": i} for i in range(1, 6)]
    fused = reciprocal_rank_fusion([results])
    assert len(fused) == 5
    assert fused[0]["id"] == "1"


def test_hybrid_search():
    idx = BM25Index()
    idx.add_documents([
        {"id": "1", "content": "machine learning algorithms for classification"},
        {"id": "2", "content": "deep neural networks and transformers"},
        {"id": "3", "content": "natural language processing with BERT"},
    ])
    vector_results = [
        {"id": "2", "content": "deep neural networks", "score": 0.92},
        {"id": "3", "content": "NLP with BERT", "score": 0.85},
    ]
    results = hybrid_search("neural network classification", vector_results, idx)
    assert len(results) > 0
    # All three docs should appear (vector has 2, BM25 matches all 3)
