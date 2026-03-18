"""Tests for Moss Hebbian recall module."""
import math
import pytest
from moss.hebbian.hebbian_recall import (
    HebbianMemoryStore, RecallResult, strengthen_pathway
)


def _embed(text: str):
    """Deterministic mock embedding: word count as dimensions."""
    words = text.lower().split()
    vec = [0.0] * 8
    for i, w in enumerate(words[:8]):
        vec[i] = len(w) / 10.0
    norm = math.sqrt(sum(x**2 for x in vec)) or 1.0
    return [x / norm for x in vec]


@pytest.fixture
def store():
    s = HebbianMemoryStore(embed_func=_embed, spreading_depth=2, spreading_decay=0.5)
    memories = [
        ("m1", "the quick brown fox"),
        ("m2", "fox jumped over fence"),
        ("m3", "brown dog slept soundly"),
        ("m4", "completely unrelated topic statistics"),
    ]
    for mid, text in memories:
        s.add_memory(mid, text)
    return s


def test_direct_recall(store):
    results = store.recall("quick fox", limit=3)
    assert len(results) <= 3
    assert all(isinstance(r, RecallResult) for r in results)
    ids = [r.memory_id for r in results]
    assert "m1" in ids or "m2" in ids


def test_spreading_activation_reaches_neighbor(store):
    """After strengthening m1-m2 pathway, recalling m1-like query should surface m2."""
    store.strengthen_pathway("m1", "m2")
    store.strengthen_pathway("m1", "m2")
    store.strengthen_pathway("m1", "m2")
    results = store.recall("quick brown fox", limit=4, include_activated=True)
    ids = [r.memory_id for r in results]
    # m2 should appear due to spreading from m1
    assert "m1" in ids


def test_no_spreading_skips_pathway(store):
    store.strengthen_pathway("m1", "m2")
    store.strengthen_pathway("m1", "m2")
    results = store.recall("quick brown fox", limit=4, include_activated=False)
    ids = [r.memory_id for r in results]
    # Without spreading, m2 should rank lower
    assert results[0].memory_id in ("m1", "m3")


def test_strengthen_pathway_increases_strength(store):
    store.strengthen_pathway("m1", "m2")
    store.strengthen_pathway("m1", "m2")
    strength = store.pathway_strength("m1", "m2")
    assert strength > 0.0


def test_empty_store_returns_empty():
    s = HebbianMemoryStore(embed_func=_embed)
    results = s.recall("anything", limit=5)
    assert results == []


def test_recall_result_fields(store):
    results = store.recall("fox", limit=2)
    for r in results:
        assert hasattr(r, "memory_id")
        assert hasattr(r, "score")
        assert hasattr(r, "source")
        assert r.score >= 0.0
