"""Tests for Moss multi-channel RRF module."""
import pytest
from moss.rrf.reciprocal_rank_fusion import (
    rrf_fuse, normalize_scores, RRFChannel
)


def test_rrf_basic_fusion():
    channels = [
        RRFChannel(name="vector", results=["a", "b", "c"], weight=1.0),
        RRFChannel(name="bm25",   results=["b", "a", "d"], weight=1.0),
    ]
    fused = rrf_fuse(channels, k=60)
    ids = [item[0] for item in fused]
    # b and a both appear in both channels — should rank high
    assert ids[0] in ("a", "b")
    assert ids[1] in ("a", "b")


def test_rrf_weighted_channels():
    """Channel with higher weight should dominate."""
    channels = [
        RRFChannel(name="strong", results=["x", "y", "z"], weight=5.0),
        RRFChannel(name="weak",   results=["z", "y", "x"], weight=0.1),
    ]
    fused = rrf_fuse(channels, k=60)
    ids = [item[0] for item in fused]
    assert ids[0] == "x"  # strong channel tops


def test_rrf_deduplication():
    channels = [
        RRFChannel(name="a", results=["x", "x", "y"], weight=1.0),
    ]
    fused = rrf_fuse(channels, k=60)
    ids = [item[0] for item in fused]
    assert len(ids) == len(set(ids))


def test_normalize_scores_basic():
    scores = {"a": 10.0, "b": 5.0, "c": 0.0}
    normed = normalize_scores(scores)
    assert abs(normed["a"] - 1.0) < 1e-9
    assert abs(normed["c"] - 0.0) < 1e-9
    assert 0.0 < normed["b"] < 1.0


def test_normalize_scores_all_equal():
    scores = {"a": 3.0, "b": 3.0}
    normed = normalize_scores(scores)
    # All equal — should all be 0 or all 1 (implementation-defined)
    assert all(v >= 0.0 for v in normed.values())


def test_rrf_empty_channels():
    fused = rrf_fuse([], k=60)
    assert fused == []


def test_rrf_single_channel():
    channels = [RRFChannel(name="only", results=["a", "b", "c"], weight=1.0)]
    fused = rrf_fuse(channels, k=60)
    ids = [item[0] for item in fused]
    assert ids[0] == "a"
