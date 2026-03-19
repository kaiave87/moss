"""
Tests for Hebbian memory retrieval module.
"""

import math
from moss.hebbian.hebbian_recall import HebbianMemoryStore, MemoryEntry, RecallResult


def mock_embed(text: str):
    """Deterministic bag-of-words embedding for testing (no external deps)."""
    words = set(text.lower().split())
    vocab = ["cat", "dog", "fish", "sky", "water", "run", "eat", "sleep", "paris", "france"]
    vec = [1.0 if w in words else 0.0 for w in vocab]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# --- Basic store operations ---

def test_add_and_recall_single():
    store = HebbianMemoryStore(embed_func=mock_embed)
    entry = store.add("m1", "The cat sat on the mat")
    assert isinstance(entry, MemoryEntry)
    assert entry.memory_id == "m1"

    results = store.recall("cat mat", limit=5)
    assert len(results) >= 1
    assert isinstance(results[0], RecallResult)
    assert results[0].memory_id == "m1"


def test_recall_top_k():
    store = HebbianMemoryStore(embed_func=mock_embed)
    store.add("a", "cat dog fish")
    store.add("b", "sky water run")
    store.add("c", "eat sleep cat")
    results = store.recall("cat", limit=2)
    assert len(results) <= 2
    ids = [r.memory_id for r in results]
    # "a" and "c" both mention cat — one should be in top 2
    assert "a" in ids or "c" in ids


def test_recall_returns_scores():
    store = HebbianMemoryStore(embed_func=mock_embed)
    store.add("x", "paris france")
    results = store.recall("paris", limit=5)
    assert len(results) >= 1
    assert results[0].score > 0.0


def test_recall_empty_store():
    store = HebbianMemoryStore(embed_func=mock_embed)
    results = store.recall("anything", limit=5)
    assert results == []


def test_strengthen_pathway_no_crash():
    """strengthen_pathway should not crash and should return a value."""
    store = HebbianMemoryStore(embed_func=mock_embed)
    store.add("a", "cat run")
    store.add("b", "fish water sky")
    strength = store.strengthen_pathway("a", "b")
    assert isinstance(strength, float)
    assert 0.0 <= strength <= 1.0


def test_strengthen_pathway_missing_ids():
    """Missing IDs should return 0.0 without raising."""
    store = HebbianMemoryStore(embed_func=mock_embed)
    result = store.strengthen_pathway("nonexistent1", "nonexistent2")
    assert result == 0.0


def test_multiple_strengthen_accumulates():
    """Strengthening multiple times should increase strength monotonically."""
    store = HebbianMemoryStore(embed_func=mock_embed, learning_rate=0.2)
    store.add("a", "cat")
    store.add("b", "dog")
    s1 = store.strengthen_pathway("a", "b")
    s2 = store.strengthen_pathway("a", "b")
    assert s2 >= s1
    assert s2 <= 1.0


def test_get_neighbors():
    store = HebbianMemoryStore(embed_func=mock_embed)
    store.add("a", "cat")
    store.add("b", "dog")
    store.add("c", "fish")
    store.strengthen_pathway("a", "b")
    neighbors = store.get_neighbors("a")
    assert any(n[0] == "b" for n in neighbors)


def test_decay_pathways():
    store = HebbianMemoryStore(embed_func=mock_embed, learning_rate=0.9, decay_rate=0.99)
    store.add("a", "cat")
    store.add("b", "dog")
    store.strengthen_pathway("a", "b")
    pruned = store.decay_pathways()
    # Very high decay should prune the pathway
    assert isinstance(pruned, int)


def test_recall_without_spreading():
    """Pure vector recall with spreading_depth=0 should still work."""
    store = HebbianMemoryStore(embed_func=mock_embed, spreading_depth=0)
    store.add("q1", "cat dog")
    store.add("q2", "fish water")
    results = store.recall("cat", limit=5)
    assert len(results) >= 1


def test_stats():
    store = HebbianMemoryStore(embed_func=mock_embed)
    store.add("a", "cat")
    store.add("b", "dog")
    store.strengthen_pathway("a", "b")
    s = store.stats()
    assert "total_memories" in s
    assert s["total_memories"] == 2
    assert "total_pathways" in s
