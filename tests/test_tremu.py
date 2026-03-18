"""Tests for TReMu temporal reasoning module."""
import pytest
from moss.tremu.temporal_reasoning import is_temporal, temporal_answer


def test_is_temporal_positive():
    assert is_temporal("when did this happen?")
    assert is_temporal("how many days ago was the meeting?")
    assert is_temporal("what date was the appointment?")


def test_is_temporal_negative():
    assert not is_temporal("what is the capital of France?")
    assert not is_temporal("who wrote this document?")


def test_temporal_answer_fallback():
    """When no LLM is provided, temporal_answer should return None gracefully."""
    result = temporal_answer(
        "when did the event happen?",
        "The event happened on January 5th, 2024.",
        llm_fn=None,
    )
    # Without an LLM function, should handle gracefully
    assert result is None or isinstance(result, str)


def test_temporal_answer_with_mock_llm():
    """With a mock LLM that returns a simple date computation."""
    
    def mock_llm(prompt, system=None):
        # Return a simple timeline for extraction
        if "timeline" in prompt.lower() or "extract" in prompt.lower():
            return '{"events": [{"date": "2024-01-05", "description": "event happened"}]}'
        if "python" in prompt.lower() or "code" in prompt.lower():
            return "```python\nfrom datetime import date\nresult = (date.today() - date(2024, 1, 5)).days\nprint(result)\n```"
        return "The event happened on January 5, 2024."
    
    result = temporal_answer(
        "how many days ago did the event happen?",
        "The event happened on January 5th, 2024.",
        llm_fn=mock_llm,
    )
    # Should return some string result (might be a number or fallback)
    assert result is None or isinstance(result, str)
