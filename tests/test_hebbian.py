"""Tests for Hebbian Recall module."""

import pytest


def test_recall_result_dataclass():
    """Test RecallResult supports both attribute and dict access."""
    from moss.hebbian.recall import RecallResult
    from datetime import datetime

    r = RecallResult(
        memory_id="test-1",
        content="Test memory content",
        summary=None,
        score=0.85,
        source="direct",
        tier="core",
        created_at=datetime.now(),
    )
    assert r.score == 0.85
    assert r["score"] == 0.85
    assert r.get("score") == 0.85
    assert r.get("nonexistent", "default") == "default"


def test_recall_result_fields():
    """Test RecallResult has all expected fields."""
    from moss.hebbian.recall import RecallResult
    from datetime import datetime

    r = RecallResult(
        memory_id="m-1",
        content="content",
        summary="summary",
        score=0.5,
        source="activated",
        tier="working",
        created_at=datetime.now(),
        pathway_strength=0.7,
        integrity_ok=True,
        is_stale=False,
        estimated_tokens=150,
    )
    assert r.pathway_strength == 0.7
    assert r.integrity_ok is True
    assert r.estimated_tokens == 150
