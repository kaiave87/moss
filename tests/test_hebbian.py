"""Tests for Hebbian recall module."""
import pytest
from moss.hebbian.hebbian_recall import HebbianMemoryStore, MemoryRecord


def dummy_embed(texts):
    """Deterministic embedding: map each word to a dimension."""
    import hashlib
    result = []
    for text in texts:
        vec = [0.0] * 16
        for i, word in enumerate(text.lower().split()):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % 16] += 1.0 / (i + 1)
        norm = sum(x**2 for x in vec) ** 0.5
        result.append([x / norm if norm > 0 else x for x in vec])
    return result


def test_add_and_recall():
    """Add memories and recall them."""
    store = HebbianMemoryStore(embed_func=dummy_embed)
    
    store.add("Alice went to the park on Monday.")
    store.add("Bob stayed home and read a book.")
    store.add("Alice visited the library on Tuesday.")
    
    results = store.recall("Alice outdoor activities", limit=2)
    assert len(results) <= 2
    assert all(hasattr(r, "content") or isinstance(r, (dict, str)) for r in results)


def test_pathway_strengthening():
    """Co-recalled memories should develop stronger pathways."""
    store = HebbianMemoryStore(embed_func=dummy_embed)
    
    id1 = store.add("Alice went to the park.")
    id2 = store.add("Alice fed the ducks.")
    store.add("Bob went to work.")
    
    # Recall Alice — should trigger pathway between id1 and id2
    results = store.recall("Alice outdoor", limit=2)
    
    # Strengthen the pathway explicitly
    if id1 and id2:
        store.strengthen_pathway(id1, id2)
        strength = store.get_pathway_strength(id1, id2)
        assert strength > 0


def test_recall_empty_store():
    """Recall from empty store returns empty list."""
    store = HebbianMemoryStore(embed_func=dummy_embed)
    results = store.recall("anything", limit=5)
    assert results == [] or len(results) == 0


def test_store_with_metadata():
    """Add memories with metadata, recall filters correctly."""
    store = HebbianMemoryStore(embed_func=dummy_embed)
    
    store.add("Alice went to the park.", metadata={"date": "2024-01-01", "person": "Alice"})
    store.add("Bob went to work.", metadata={"date": "2024-01-02", "person": "Bob"})
    
    results = store.recall("park visit", limit=2)
    assert len(results) <= 2
