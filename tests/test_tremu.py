"""Tests for TReMu temporal reasoning module."""

from moss.tremu import is_temporal, extract_timeline
from moss.tremu.temporal_reasoning import (
    _parse_timeline_json,
    _extract_code,
    execute_temporal_code,
)


def test_is_temporal_strong_keywords():
    assert is_temporal("When did Alice start her new job?")
    assert is_temporal("How long ago did they move?")
    assert is_temporal("What date was the meeting?")


def test_is_temporal_negative():
    assert not is_temporal("What is Alice's favorite color?")
    assert not is_temporal("Tell me about the project")


def test_is_temporal_weak_signals():
    assert is_temporal("Did it happen before or after the move?")
    assert is_temporal("What was the first event in the sequence?")


def test_parse_timeline_json_valid():
    raw = '[{"event": "started job", "date": "2023-05-15", "confidence": "exact", "session": 1}]'
    result = _parse_timeline_json(raw)
    assert len(result) == 1
    assert result[0]["date"] == "2023-05-15"


def test_parse_timeline_json_with_fences():
    raw = '```json\n[{"event": "test", "date": "2023-01-01", "confidence": "inferred", "session": null}]\n```'
    result = _parse_timeline_json(raw)
    assert len(result) == 1


def test_parse_timeline_json_empty():
    assert _parse_timeline_json("") == []
    assert _parse_timeline_json("no json here") == []


def test_extract_code_plain():
    code = 'answer = "May 2023"'
    assert _extract_code(code) == code


def test_extract_code_fenced():
    raw = '```python\nanswer = "May 2023"\n```'
    assert _extract_code(raw) == 'answer = "May 2023"'


def test_execute_temporal_code_simple():
    code = """
dates = [e["date"] for e in timeline if e["date"] != "unknown"]
dates.sort()
answer = dates[0] if dates else "unknown"
"""
    timeline = [
        {"event": "A", "date": "2023-05-15", "confidence": "exact", "session": 1},
        {"event": "B", "date": "2023-03-01", "confidence": "exact", "session": 2},
    ]
    result = execute_temporal_code(code, timeline)
    assert result == "2023-03-01"


def test_execute_temporal_code_duration():
    code = """
from datetime import datetime, timedelta
d1 = datetime.strptime(timeline[0]["date"], "%Y-%m-%d")
d2 = datetime.strptime(timeline[1]["date"], "%Y-%m-%d")
diff = abs((d2 - d1).days)
answer = f"{diff} days"
"""
    timeline = [
        {"event": "start", "date": "2023-01-10", "confidence": "exact", "session": 1},
        {"event": "end", "date": "2023-01-20", "confidence": "exact", "session": 2},
    ]
    result = execute_temporal_code(code, timeline)
    assert result == "10 days"


def test_execute_temporal_code_sandbox_blocks_imports():
    code = 'import os\nanswer = os.getcwd()'
    timeline = []
    result = execute_temporal_code(code, timeline)
    assert result is None  # Should fail due to import stripping
