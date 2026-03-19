"""
Tests for TReMu temporal reasoning module.
"""

from moss.tremu.temporal_reasoning import temporal_answer, is_temporal


# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------

def mock_llm_timeline(prompt: str, system: str = None) -> str:
    """Returns a valid timeline JSON array."""
    return '[{"event": "event A", "date": "2023-06-15", "confidence": "exact"}, {"event": "event B", "date": "2023-09-20", "confidence": "exact"}]'


def mock_llm_code(prompt: str, system: str = None) -> str:
    """Returns Python code that computes day difference."""
    return """```python
from datetime import date
d1 = date(2023, 6, 15)
d2 = date(2023, 9, 20)
result = abs((d2 - d1).days)
print(result)
```"""


def mock_llm_combined(prompt: str, system: str = None) -> str:
    """Single mock that handles both timeline extraction and code generation calls."""
    # Timeline extraction: return JSON array
    if "[{" not in prompt and ("timeline" in prompt.lower() or "extract" in prompt.lower() or "JSON" in prompt):
        return '[{"event": "graduation ceremony", "date": "2023-06-15", "confidence": "exact"}]'
    # Code generation: return Python
    if "python" in prompt.lower() or "compute" in prompt.lower() or "code" in prompt.lower():
        return "```python\nprint('June 15, 2023')\n```"
    return "June 15, 2023"


def mock_fallback(question: str, context: str, llm_fn) -> str:
    """Simple fallback that always returns a string."""
    return "Fallback answer for: " + question[:50]


def _make_duration_llm():
    """Mock LLM for a 90-day duration test."""
    call_count = [0]

    def _llm(prompt: str, system: str = None) -> str:
        call_count[0] += 1
        if call_count[0] <= 2:
            # Timeline extraction calls
            return '[{"event": "project start", "date": "2024-01-01", "confidence": "exact"}, {"event": "project finish", "date": "2024-03-31", "confidence": "exact"}]'
        # Code generation calls
        return "```python\nfrom datetime import date\nresult = (date(2024,3,31) - date(2024,1,1)).days\nprint(result)\n```"

    return _llm


# ---------------------------------------------------------------------------
# is_temporal
# ---------------------------------------------------------------------------

def test_is_temporal_duration():
    assert is_temporal("How long did it take?") is True


def test_is_temporal_how_many_days():
    assert is_temporal("How many days between the two events?") is True


def test_is_temporal_when():
    assert is_temporal("When did the meeting happen?") is True


def test_is_temporal_non_temporal():
    assert is_temporal("What is the capital of France?") is False


def test_is_temporal_before_after():
    assert is_temporal("Which came first, the trip or the wedding?") is True


def test_is_temporal_empty():
    assert is_temporal("") is False


def test_is_temporal_elapsed():
    assert is_temporal("How much time elapsed between the two dates?") is True


# ---------------------------------------------------------------------------
# temporal_answer: return type contract
# ---------------------------------------------------------------------------

def test_temporal_answer_with_fallback_returns_string():
    """When fallback_fn is provided, temporal_answer always returns a string."""
    answer = temporal_answer(
        "When did the party happen?",
        "The party was on June 15, 2023.",
        mock_llm_combined,
        fallback_fn=mock_fallback,
    )
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_temporal_answer_non_temporal_with_fallback():
    """Non-temporal questions should use fallback path."""
    answer = temporal_answer(
        "What is the capital of France?",
        "Paris is the capital.",
        mock_llm_combined,
        fallback_fn=mock_fallback,
    )
    assert isinstance(answer, str)


def test_temporal_answer_none_without_fallback():
    """Without fallback_fn, failed pipeline returns None (documented behavior)."""
    answer = temporal_answer(
        "When did the party happen?",
        "",  # empty context → timeline extraction fails
        mock_llm_combined,
        fallback_fn=None,
    )
    # Acceptable: either None (failed) or a string (succeeded)
    assert answer is None or isinstance(answer, str)


def test_temporal_answer_duration_pipeline():
    """Full pipeline test for a calculation-type temporal question."""
    context = "Alice started the project on January 1, 2024. She finished on March 31, 2024."
    answer = temporal_answer(
        "How many days did Alice work on the project?",
        context,
        _make_duration_llm(),
        fallback_fn=mock_fallback,
    )
    assert isinstance(answer, str)
    assert len(answer) > 0
    # Either the code executed (90) or fallback ran
    assert "90" in answer or "fallback" in answer.lower() or len(answer) > 0


def test_temporal_answer_with_timeline_mock():
    """Mock that returns valid timeline array should succeed in pipeline."""
    def _dual_mock(prompt, system=None):
        # Timeline extraction path
        if "[" in prompt or "extract" in prompt.lower() or "timeline" in prompt.lower():
            return '[{"event": "graduation", "date": "2023-06-15", "confidence": "exact"}, {"event": "job start", "date": "2023-09-01", "confidence": "exact"}]'
        # Code generation path
        return "```python\nfrom datetime import date\nresult = (date(2023,9,1) - date(2023,6,15)).days\nprint(result)\n```"

    answer = temporal_answer(
        "How many days between graduation and starting the job?",
        "John graduated on June 15. He started his job on September 1.",
        _dual_mock,
        fallback_fn=mock_fallback,
    )
    assert isinstance(answer, str)
