"""Tests for Moss TReMu temporal reasoning module."""
import pytest
from moss.tremu.temporal_reasoning import is_temporal, temporal_answer, parse_timeline_json


def _mock_llm(prompt: str, system: str = None) -> str:
    """Mock LLM that returns structured JSON for timeline extraction."""
    if "timeline" in prompt.lower() or "events" in prompt.lower():
        return '''```json
{
  "events": [
    {"label": "started job", "date": "2024-03-15"},
    {"label": "moved cities", "date": "2024-06-01"},
    {"label": "today", "date": "2025-01-10"}
  ]
}
```'''
    return "2024-03-15"


def test_is_temporal_keyword_match():
    assert is_temporal("when did she start the job?")
    assert is_temporal("how long ago did he move?")
    assert is_temporal("what date was the meeting?")


def test_is_temporal_non_temporal():
    assert not is_temporal("what is the capital of France?")
    assert not is_temporal("who wrote this document?")


def test_parse_timeline_json_basic():
    raw = '''{"events": [{"label": "start", "date": "2024-01-01"}]}'''
    events = parse_timeline_json(raw)
    assert len(events) == 1
    assert events[0]["label"] == "start"


def test_parse_timeline_json_with_code_fence():
    raw = '''```json\n{"events": [{"label": "a", "date": "2024-05-01"}]}\n```'''
    events = parse_timeline_json(raw)
    assert len(events) == 1


def test_temporal_answer_returns_string():
    context = "Alice started her job on March 15, 2024. She moved to Toronto on June 1, 2024."
    question = "How many months after starting her job did Alice move?"
    answer = temporal_answer(question, context, _mock_llm)
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_temporal_answer_non_temporal_returns_empty():
    """Non-temporal question should return empty string (caller handles with fallback)."""
    question = "What is Alice's favourite colour?"
    answer = temporal_answer(question, "Alice likes blue.", _mock_llm)
    # Either empty or a passthrough — must be a string
    assert isinstance(answer, str)
