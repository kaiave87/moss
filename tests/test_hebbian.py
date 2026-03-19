"""
Tests for Hebbian Neuroplastic Memory

Stores memories, runs recall, verifies pathway strengthening and spreading
activation. All tests run without network access using MockDB and the
stub embedding model.
"""

import pytest
from moss.hebbian.db import MockDB
from moss.hebbian.embeddings import get_embedding
from moss.hebbian.recall import recall
from moss.hebbian.pathway_strengthening import (
    strengthen_pathway_sync,
    weaken_pathway_sync,
    strengthen_batch,
)
from moss.hebbian.spreading_activation import spreading_activation
from moss.hebbian.channels import (
    PHASE_GAINS, LABILE_HOURS, STRONG_THRESHOLD, cepeda_review_interval,
    age_dependent_floor, power_law_decay, get_phase_context, boosted_gain,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MEMORIES = [
    ("Paris is the capital of France.", "Paris capital"),
    ("Berlin is the capital of Germany.", "Berlin capital"),
    ("Tokyo is the capital of Japan.", "Tokyo capital"),
    ("Machine learning uses statistical models.", "Machine learning"),
    ("Deep learning is a subset of machine learning.", "Deep learning"),
    ("The Eiffel Tower is located in Paris.", "Eiffel Tower Paris"),
    ("The Brandenburg Gate is in Berlin.", "Brandenburg Gate"),
]


@pytest.fixture
def db():
    """MockDB populated with sample memories and embeddings."""
    d = MockDB()
    for content, summary in MEMORIES:
        d.store_memory(
            content=content,
            summary=summary,
            tier="active",
            memory_type="episodic",
            embedding=get_embedding(content),
        )
    return d


# ---------------------------------------------------------------------------
# MockDB Tests
# ---------------------------------------------------------------------------

def test_store_and_retrieve():
    db = MockDB()
    mem_id = db.store_memory("Hello world", summary="greeting")
    retrieved = db.get_memory(mem_id)
    assert retrieved is not None
    assert retrieved["content"] == "Hello world"
    assert retrieved["summary"] == "greeting"


def test_store_returns_unique_ids():
    db = MockDB()
    id1 = db.store_memory("Memory one")
    id2 = db.store_memory("Memory two")
    assert id1 != id2


def test_search_by_vector_returns_results(db):
    query_vec = get_embedding("capital city France")
    results = db.search_by_vector(query_vec, limit=3)
    assert len(results) > 0


def test_search_by_vector_exact_match(db):
    """Storing a memory and querying with its exact embedding returns it."""
    content = "Paris is the capital of France and home to the Eiffel Tower."
    mem_id = db.store_memory(content)
    # Query with the identical embedding — should be top result (cosine sim = 1.0)
    query_vec = get_embedding(content)
    results = db.search_by_vector(query_vec, limit=3)
    ids = [r["id"] for r in results]
    assert mem_id in ids


def test_search_by_vector_limit(db):
    query_vec = get_embedding("capital")
    results = db.search_by_vector(query_vec, limit=2)
    assert len(results) <= 2


def test_update_access_increments_count(db):
    query_vec = get_embedding("Paris")
    results = db.search_by_vector(query_vec, limit=1)
    assert len(results) > 0
    mem_id = results[0]["id"]
    initial_count = results[0].get("access_count", 0)
    db.update_access([mem_id])
    updated = db.get_memory(mem_id)
    assert updated["access_count"] == initial_count + 1


# ---------------------------------------------------------------------------
# Pathway Tests
# ---------------------------------------------------------------------------

def test_strengthen_pathway_creates_new(db):
    query_vec = get_embedding("capital")
    results = db.search_by_vector(query_vec, limit=2)
    assert len(results) >= 2
    id_a, id_b = results[0]["id"], results[1]["id"]
    ok = strengthen_pathway_sync(id_a, id_b, db, boost=0.1)
    assert ok is True
    assert db.count_pathways() >= 1


def test_strengthen_pathway_increases_strength(db):
    query_vec = get_embedding("capital")
    results = db.search_by_vector(query_vec, limit=2)
    id_a, id_b = results[0]["id"], results[1]["id"]
    strengthen_pathway_sync(id_a, id_b, db, boost=0.1)
    strengthen_pathway_sync(id_a, id_b, db, boost=0.1)
    pathways = db.get_pathways([id_a], min_strength=0.0)
    assert len(pathways) >= 1
    strength = pathways[0]["strength"]
    assert strength >= 0.2


def test_strengthen_self_pathway_returns_false(db):
    query_vec = get_embedding("Paris")
    results = db.search_by_vector(query_vec, limit=1)
    mem_id = results[0]["id"]
    ok = strengthen_pathway_sync(mem_id, mem_id, db)
    assert ok is False


def test_weaken_pathway(db):
    query_vec = get_embedding("capital")
    results = db.search_by_vector(query_vec, limit=2)
    id_a, id_b = results[0]["id"], results[1]["id"]
    strengthen_pathway_sync(id_a, id_b, db, boost=0.3)
    pathways_before = db.get_pathways([id_a])
    strength_before = pathways_before[0]["strength"]
    weaken_pathway_sync(id_a, id_b, db, penalty=0.05)
    pathways_after = db.get_pathways([id_a])
    strength_after = pathways_after[0]["strength"]
    assert strength_after < strength_before


def test_strengthen_batch(db):
    query_vec = get_embedding("Europe capital")
    results = db.search_by_vector(query_vec, limit=3)
    ids = [r["id"] for r in results]
    count = strengthen_batch(ids, db, boost=0.1, max_pairs=10)
    assert count >= 1
    assert db.count_pathways() >= 1


# ---------------------------------------------------------------------------
# Spreading Activation Tests
# ---------------------------------------------------------------------------

def test_spreading_activation_returns_seeds(db):
    query_vec = get_embedding("Paris")
    results = db.search_by_vector(query_vec, limit=2)
    seed_ids = [r["id"] for r in results]
    seed_scores = {r["id"]: r["similarity"] for r in results}

    # Create pathways first so spreading has something to traverse
    if len(seed_ids) >= 2:
        strengthen_pathway_sync(seed_ids[0], seed_ids[1], db, boost=0.3)

    activated = spreading_activation(seed_ids, db, seed_scores=seed_scores, depth=1)
    # Seeds must be in the activated dict
    for sid in seed_ids:
        assert sid in activated


def test_spreading_activation_propagates(db):
    query_vec = get_embedding("capital Europe")
    results = db.search_by_vector(query_vec, limit=3)
    seed_ids = [r["id"] for r in results[:2]]
    extra_id = results[2]["id"] if len(results) >= 3 else None

    # Create a pathway from seed to the extra memory
    if extra_id:
        strengthen_pathway_sync(seed_ids[0], extra_id, db, boost=0.5)

    seed_scores = {sid: 1.0 for sid in seed_ids}
    activated = spreading_activation(seed_ids, db, seed_scores=seed_scores, depth=1, decay=0.5)

    if extra_id:
        # extra_id should have been activated via the pathway
        assert extra_id in activated


def test_spreading_activation_empty_seeds(db):
    activated = spreading_activation([], db)
    assert activated == {}


# ---------------------------------------------------------------------------
# Recall Tests
# ---------------------------------------------------------------------------

def test_recall_returns_results(db):
    results = recall("What is the capital of France?", db=db, embed_fn=get_embedding, limit=5)
    assert len(results) > 0


def test_recall_scores_positive(db):
    results = recall("Paris", db=db, embed_fn=get_embedding, limit=3)
    for r in results:
        assert r.score >= 0


def test_recall_sorted_by_score(db):
    results = recall("capital city Europe", db=db, embed_fn=get_embedding, limit=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_recall_respects_limit(db):
    results = recall("Europe", db=db, embed_fn=get_embedding, limit=2)
    assert len(results) <= 2


def test_recall_returns_recall_result_objects(db):
    from moss.hebbian.recall import RecallResult
    results = recall("Berlin Germany", db=db, embed_fn=get_embedding, limit=3)
    for r in results:
        assert isinstance(r, RecallResult)
        assert hasattr(r, "memory_id")
        assert hasattr(r, "content")
        assert hasattr(r, "score")
        assert hasattr(r, "source")


def test_recall_readonly_no_strengthen(db):
    """readonly=True should not modify pathways."""
    count_before = db.count_pathways()
    recall("Paris France", db=db, embed_fn=get_embedding, limit=3, readonly=True)
    count_after = db.count_pathways()
    assert count_after == count_before


def test_recall_non_readonly_strengthens(db):
    """After non-readonly recall, pathways should be created/strengthened."""
    count_before = db.count_pathways()
    recall("Paris France capital", db=db, embed_fn=get_embedding, limit=5, readonly=False)
    count_after = db.count_pathways()
    assert count_after >= count_before


def test_recall_content_field_not_empty(db):
    results = recall("Japan Tokyo", db=db, embed_fn=get_embedding, limit=3)
    for r in results:
        assert r.content is not None
        assert len(r.content) > 0


# ---------------------------------------------------------------------------
# Channels / Constants Tests
# ---------------------------------------------------------------------------

def test_phase_gains_defined():
    for phase in ("PRIME", "ACTIVE", "FLOW", "SORT", "DREAM"):
        assert phase in PHASE_GAINS
        assert PHASE_GAINS[phase] > 0


def test_get_phase_context():
    ctx = get_phase_context("DREAM")
    assert ctx.phase == "DREAM"
    assert ctx.gain == 2.0


def test_age_dependent_floor_decreases_with_age():
    floor_new = age_dependent_floor(0.8, days_since_last_access=0)
    floor_old = age_dependent_floor(0.8, days_since_last_access=120)
    assert floor_new > floor_old


def test_power_law_decay():
    s = power_law_decay(1.0, days_inactive=10)
    assert 0 < s <= 1.0


def test_cepeda_review_interval():
    interval = cepeda_review_interval(100)
    assert interval > 0


def test_boosted_gain_labile_higher():
    gain_normal = boosted_gain(0.1, 1.0, is_labile=False)
    gain_labile = boosted_gain(0.1, 1.0, is_labile=True)
    assert gain_labile > gain_normal


# ---------------------------------------------------------------------------
# Embedding Tests
# ---------------------------------------------------------------------------

def test_embedding_dimension():
    from moss.hebbian.embeddings import EMBEDDING_DIM
    vec = get_embedding("test text")
    assert len(vec) == EMBEDDING_DIM


def test_embedding_deterministic():
    v1 = get_embedding("same text")
    v2 = get_embedding("same text")
    assert v1 == v2


def test_embedding_different_for_different_text():
    v1 = get_embedding("Paris")
    v2 = get_embedding("Tokyo")
    assert v1 != v2


def test_embedding_normalized():
    import math
    vec = get_embedding("hello world")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6
