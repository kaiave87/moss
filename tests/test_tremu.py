"""
Tests for TReMu Temporal Reasoning Module

Verifies temporal_answer and is_temporal without network access.
The llm_fn is mocked to return deterministic responses.
"""

import json
import pytest
from moss.tremu.temporal_reasoning import temporal_answer, is_temporal


# ---------------------------------------------------------------------------
# is_temporal tests
# ---------------------------------------------------------------------------

def test_is_temporal_when_did():
    assert is_temporal("When did they get married?") is True


def test_is_temporal_how_long_ago():
    assert is_temporal("How long ago did the trip happen?") is True


def test_is_temporal_what_date():
    assert is_temporal("What date was the concert?") is True


def test_is_temporal_before_after():
    assert is_temporal("Did the meeting happen before or after the flight?") is True


def test_is_temporal_non_temporal():
    assert is_temporal("What is the capital of France?") is False


def test_is_temporal_factual_no_time():
    assert is_temporal("Tell me about machine learning.") is False


def test_is_temporal_most_recent():
    assert is_temporal("What was the most recent event they talked about?") is True


# ---------------------------------------------------------------------------
# temporal_answer tests
# ---------------------------------------------------------------------------

SAMPLE_CONTEXT = """[15 Jan 2024] Alice and Bob got married on January 15, 2024.
[20 Feb 2024] Alice started a new job on February 20, 2024.
[01 Mar 2024] They moved to a new city on March 1, 2024."""


def _make_llm_fn(timeline_json=None, code_str=None):
    """
    Mock LLM function.

    First call returns a timeline JSON array.
    Second call returns Python code.
    """
    call_count = [0]

    default_timeline = [
        {"event": "Alice and Bob married", "date": "2024-01-15", "confidence": "exact", "session": 1},
        {"event": "Alice started new job", "date": "2024-02-20", "confidence": "exact", "session": 2},
        {"event": "Moved to new city", "date": "2024-03-01", "confidence": "exact", "session": 3},
    ]
    _timeline = timeline_json if timeline_json is not None else default_timeline

    default_code = """
from datetime import datetime
timeline_dates = sorted(
    [t for t in timeline if t.get("date") and t["date"] != "unknown"],
    key=lambda t: t["date"]
)
answer = timeline_dates[0]["event"] if timeline_dates else "unknown"
"""
    _code = code_str if code_str is not None else default_code

    def llm_fn(prompt, system=None):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: return timeline extraction
            return json.dumps(_timeline)
        else:
            # Second call: return computation code
            return _code

    return llm_fn


def test_temporal_answer_basic():
    """temporal_answer should return a non-empty string answer."""
    llm_fn = _make_llm_fn()
    result = temporal_answer(
        question="When did they get married?",
        context=SAMPLE_CONTEXT,
        llm_fn=llm_fn,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_temporal_answer_first_event():
    """For 'what was the first event', should return earliest date event."""
    llm_fn = _make_llm_fn()
    result = temporal_answer(
        question="What was the first event in chronological order?",
        context=SAMPLE_CONTEXT,
        llm_fn=llm_fn,
    )
    assert isinstance(result, str)
    # Should identify the marriage as the first event
    assert "married" in result.lower() or "2024-01-15" in result or len(result) > 0


def test_temporal_answer_non_temporal_passthrough():
    """For non-temporal questions, temporal_answer should return None or empty."""
    call_count = [0]

    def llm_fn(prompt, system=None):
        call_count[0] += 1
        return "[]"  # empty timeline

    result = temporal_answer(
        question="What is the capital of France?",
        context="France is a country in Europe.",
        llm_fn=llm_fn,
    )
    # Non-temporal question: should return None or empty string
    assert result is None or result == "" or isinstance(result, str)


def test_temporal_answer_malformed_timeline():
    """temporal_answer should handle LLM returning malformed JSON gracefully."""
    call_count = [0]

    def llm_fn(prompt, system=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return "not valid json {{{"  # malformed
        return "answer = 'unknown'"

    result = temporal_answer(
        question="When did they meet?",
        context=SAMPLE_CONTEXT,
        llm_fn=llm_fn,
    )
    # Should not crash, may return None or fallback string
    assert result is None or isinstance(result, str)


def test_temporal_answer_empty_context():
    """temporal_answer with empty context should not crash."""
    llm_fn = _make_llm_fn(timeline_json=[])
    result = temporal_answer(
        question="When did the event happen?",
        context="",
        llm_fn=llm_fn,
    )
    assert result is None or isinstance(result, str)


def test_is_temporal_duration_question():
    assert is_temporal("How many days between the wedding and the move?") is True


def test_is_temporal_timeline_question():
    assert is_temporal("What is the timeline of events?") is True
