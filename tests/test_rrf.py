"""Tests for Multi-Channel RRF module."""
import pytest
from moss.rrf.reciprocal_rank_fusion import reciprocal_rank_fusion, RRFConfig
from moss.rrf.bm25_rrf import BM25RRFRetriever


def test_rrf_basic():
    """Basic RRF fusion of two ranked lists."""
    ranking_a = ["doc1", "doc2", "doc3"]
    ranking_b = ["doc2", "doc1", "doc4"]
    
    fused = reciprocal_rank_fusion([ranking_a, ranking_b])
    assert isinstance(fused, list)
    assert len(fused) > 0
    # doc2 and doc1 should be high in fused ranking (present in both lists)
    top_ids = [item[0] if isinstance(item, tuple) else item for item in fused[:2]]
    assert "doc1" in top_ids or "doc2" in top_ids


def test_rrf_single_list():
    """RRF with a single list should preserve order."""
    ranking = ["doc1", "doc2", "doc3"]
    fused = reciprocal_rank_fusion([ranking])
    result_ids = [item[0] if isinstance(item, tuple) else item for item in fused]
    assert result_ids == ranking


def test_rrf_empty():
    """RRF with empty lists should return empty."""
    fused = reciprocal_rank_fusion([])
    assert fused == []


def test_rrf_k_parameter():
    """k parameter affects score magnitude but not relative ranking."""
    ranking_a = ["doc1", "doc2", "doc3"]
    ranking_b = ["doc1", "doc3", "doc2"]
    
    fused_60 = reciprocal_rank_fusion([ranking_a, ranking_b], k=60)
    fused_10 = reciprocal_rank_fusion([ranking_a, ranking_b], k=10)
    
    # doc1 should be first in both (appears first in both lists)
    for fused in [fused_60, fused_10]:
        first = fused[0]
        first_id = first[0] if isinstance(first, tuple) else first
        assert first_id == "doc1"


def test_bm25_retriever_basic():
    """BM25 retriever with a small corpus."""
    documents = [
        {"id": "1", "text": "the quick brown fox jumps over the lazy dog"},
        {"id": "2", "text": "cats and dogs are common pets"},
        {"id": "3", "text": "machine learning and neural networks"},
    ]
    
    retriever = BM25RRFRetriever(documents)
    results = retriever.search("fox dog", top_k=2)
    
    assert isinstance(results, list)
    assert len(results) <= 2
    # Doc 1 should be relevant to "fox dog"
    result_ids = [r["id"] if isinstance(r, dict) else r for r in results]
    assert "1" in result_ids
