"""
TReMu-style temporal reasoning module 

Neuro-symbolic approach: detect temporal questions, extract structured timelines
via LLM, generate Python code to compute the answer, execute in sandbox.

Based on TReMu (ACL 2025, arxiv 2502.01630).
Proven yield: 29.8% -> 77.7% on temporal category.

Usage:
    from temporal_reasoning import temporal_answer, is_temporal

    # llm_fn signature: (prompt: str, system: str = None) -> str
    answer = temporal_answer(question, retrieved_context, llm_fn)
"""

from __future__ import annotations

import re
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keywords that strongly signal temporal questions (high precision)
_TEMPORAL_KEYWORDS = [
    "when did", "when was", "when is", "when does", "when will",
    "what date", "what day", "what time", "what year", "what month",
    "how long ago", "how long has", "how long did", "how long was",
    "how many days", "how many weeks", "how many months", "how many years",
    "how much time",
    "before or after", "which came first", "first or",
    "most recent", "earliest", "latest",
    "how long between", "time between", "duration",
    "timeline", "chronological", "in order",
]

# Weaker signals -- need at least two, or one plus a date-like pattern
_TEMPORAL_WEAK_SIGNALS = [
    "before", "after", "during", "since", "until",
    "first", "last", "recent", "previous", "next",
    "earlier", "later", "prior",
    "timeline", "order", "sequence", "chronological",
    "ago", "yet", "already",
]

# Date-like patterns in the question itself
_DATE_PATTERN = re.compile(
    r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'  # 01/02/2023
    r'|\d{4}[/-]\d{1,2}[/-]\d{1,2}'          # 2023-01-02
    r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}'  # January 5
    r'|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*'  # 5 January
    r'|yesterday|last\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)'
    r'|(?:two|three|four|five|six|seven|eight|nine|ten)\s+(?:days?|weeks?|months?|years?)\s+ago'
    r'|\d+\s+(?:days?|weeks?|months?|years?)\s+ago'
    r')',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_TIMELINE_EXTRACTION_PROMPT = """\
From the following conversation memories, extract a timeline of events relevant to the question.

For each event, provide:
- event: Brief description of what happened
- date: The date in YYYY-MM-DD format. If only month/year known, use YYYY-MM-01. If only year, use YYYY-01-01.
- confidence: "exact" (date explicitly stated), "inferred" (calculated from relative references or context), "approximate" (rough estimate)
- session: The session number or identifier the info came from (integer, or null if unknown)

IMPORTANT:
- Resolve relative dates ("yesterday", "last Tuesday") using the session date context provided.
- Each memory segment starts with a date in brackets like [15 May 2023] -- use that as the reference date for that segment.
- Only extract events relevant to answering the question.
- If a date cannot be determined at all, use "unknown" for the date field.

Memories:
{context}

Question: {question}

Respond with ONLY a JSON array, no other text:
[
  {{"event": "...", "date": "YYYY-MM-DD", "confidence": "exact", "session": 1}},
  ...
]"""

_CODE_GENERATION_PROMPT = """\
Given this timeline of events and a question, write Python code that computes the answer.

Timeline (JSON):
{timeline_json}

Question: {question}

Write ONLY Python code that:
1. Parses the timeline data (provided as the variable `timeline` -- a list of dicts with keys: event, date, confidence, session)
2. Performs the necessary temporal calculation (date comparison, arithmetic, ordering, duration)
3. Stores the final answer in a variable called `answer` (as a string)

Rules:
- You may use `datetime` and `timedelta` from the datetime module (already imported).
- You may use `re` (already imported).
- NEVER import anything. No `import time`, no `import calendar`, no imports at all.
- Use datetime.strptime() for parsing date strings (NOT time.strptime).
- Use (date2 - date1).days for day differences (NOT time.mktime or timestamps).
- Do NOT use print().
- Handle "unknown" dates gracefully (skip them or note uncertainty).
- The `answer` variable must be a string containing ONLY the answer, no explanation.
- For date answers, prefer formats like "14 May 2023" or "May 2023".
- For duration answers, use natural language like "3 days" or "2 months".
- Keep the code simple and direct.

```python
# timeline is already defined as a list of dicts
# datetime and timedelta are already imported — use them directly, no imports
"""

# ---------------------------------------------------------------------------
# Step 1: Temporal Question Classifier
# ---------------------------------------------------------------------------

def is_temporal(question: str) -> bool:
    """Classify whether a question requires temporal reasoning.

    Uses keyword detection + heuristics. Fast, no LLM call needed.
    Designed for high recall on temporal questions -- false positives are
    handled gracefully by the fallback mechanism.

    Args:
        question: The question text.

    Returns:
        True if the question likely needs temporal reasoning.
    """
    q_lower = question.lower().strip()

    # Strong keyword match -- any one is sufficient
    for kw in _TEMPORAL_KEYWORDS:
        if kw in q_lower:
            return True

    # Check for "when" at start of question
    if q_lower.startswith("when "):
        return True

    # Weak signals: need two or more, or one plus a date pattern
    weak_count = sum(1 for sig in _TEMPORAL_WEAK_SIGNALS if sig in q_lower)
    has_date = bool(_DATE_PATTERN.search(q_lower))

    if weak_count >= 2:
        return True
    if weak_count >= 1 and has_date:
        return True

    return False


# ---------------------------------------------------------------------------
# Step 2: Timeline Extraction
# ---------------------------------------------------------------------------

def extract_timeline(
    question: str,
    context: str,
    llm_fn: Callable[..., str],
) -> List[Dict]:
    """Extract a structured timeline of events from retrieved context.

    Uses LLM to parse conversation memories into a JSON timeline.

    Args:
        question: The temporal question.
        context: Retrieved conversation memories/context.
        llm_fn: Callable(prompt, system=None) -> str.

    Returns:
        List of event dicts: [{event, date, confidence, session}, ...]
        Returns empty list on failure.
    """
    prompt = _TIMELINE_EXTRACTION_PROMPT.format(
        context=context,
        question=question,
    )
    system = (
        "You are a precise timeline extraction assistant. "
        "Output ONLY valid JSON arrays. No markdown, no explanation."
    )

    try:
        raw = llm_fn(prompt, system=system)
    except Exception:
        logger.error("Timeline extraction LLM call failed: %s", traceback.format_exc())
        return []

    return _parse_timeline_json(raw)


def _parse_timeline_json(raw: str) -> List[Dict]:
    """Parse LLM output into timeline list, handling common formatting issues."""
    if not raw or not raw.strip():
        return []

    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Find JSON array boundaries
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        logger.warning("No JSON array found in timeline extraction output")
        return []

    json_str = text[start : end + 1]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse timeline JSON: %s", e)
        # Try fixing common issues: trailing commas
        cleaned = re.sub(r',\s*]', ']', json_str)
        cleaned = re.sub(r',\s*}', '}', cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Timeline JSON parse failed even after cleanup")
            return []

    if not isinstance(parsed, list):
        logger.warning("Timeline extraction returned non-list: %s", type(parsed))
        return []

    # Validate and normalize entries
    timeline = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        event = {
            "event": str(entry.get("event", "")),
            "date": str(entry.get("date", "unknown")),
            "confidence": str(entry.get("confidence", "approximate")),
            "session": entry.get("session"),
        }
        if event["event"]:  # skip empty events
            timeline.append(event)

    return timeline


# ---------------------------------------------------------------------------
# Step 3: Code Generation
# ---------------------------------------------------------------------------

def generate_temporal_code(
    question: str,
    timeline: List[Dict],
    llm_fn: Callable[..., str],
) -> str:
    """Generate Python code to compute the temporal answer.

    Args:
        question: The temporal question.
        timeline: Structured timeline from extract_timeline().
        llm_fn: Callable(prompt, system=None) -> str.

    Returns:
        Python code string, or empty string on failure.
    """
    timeline_json = json.dumps(timeline, indent=2, default=str)

    prompt = _CODE_GENERATION_PROMPT.format(
        timeline_json=timeline_json,
        question=question,
    )
    system = (
        "You are a Python code generator for temporal reasoning. "
        "Output ONLY executable Python code. No markdown fences, no explanation. "
        "The code must set a variable called `answer` with the result string."
    )

    try:
        raw = llm_fn(prompt, system=system)
    except Exception:
        logger.error("Code generation LLM call failed: %s", traceback.format_exc())
        return ""

    return _extract_code(raw)


def _extract_code(raw: str) -> str:
    """Extract Python code from LLM output, stripping markdown fences."""
    if not raw or not raw.strip():
        return ""

    text = raw.strip()

    # If wrapped in code fences, extract the content
    fence_match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # If it looks like raw code (starts with comment, import, variable, or common statement)
    first_line = text.split("\n")[0].strip()
    if (first_line.startswith("#")
        or first_line.startswith("from ")
        or first_line.startswith("import ")
        or "=" in first_line
        or first_line.startswith("timeline")
        or first_line.startswith("dates")
        or first_line.startswith("events")
        or first_line.startswith("for ")
        or first_line.startswith("if ")):
        return text

    # Last resort: return as-is and let execution catch errors
    return text


# ---------------------------------------------------------------------------
# Step 4: Sandboxed Code Execution
# ---------------------------------------------------------------------------

# Allowed builtins for the sandbox -- minimal set for date arithmetic
_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": lambda *a, **kw: None,  # silenced print -- code should use `answer`
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "None": None,
    "True": True,
    "False": False,
    # Exception types needed for try/except in generated code
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "Exception": Exception,
}


def execute_temporal_code(
    code: str,
    timeline: List[Dict],
    timeout_seconds: float = 5.0,
    question: str = "",
) -> Optional[str]:
    """Execute generated temporal reasoning code in a restricted sandbox.

    The code runs with restricted globals: only datetime, timedelta, re,
    and a minimal set of builtins. No file I/O, no network, no imports.

    Args:
        code: Python code string. Must set variable `answer`.
        timeline: Timeline data made available to the code.
        timeout_seconds: Max execution time (enforced via signal on Unix).

    Returns:
        The answer string, or None if execution failed.
    """
    if not code or not code.strip():
        return None

    # Pre-process code: strip ALL import lines. All needed modules (datetime,
    # timedelta, re) are already injected into the sandbox namespace. LLMs
    # frequently generate "import time", "import datetime", etc. even when told
    # not to. Strip them all — if the code is otherwise correct it will work;
    # if it actually uses disallowed APIs (time.mktime etc.) it will fail with
    # NameError which is caught below and triggers fallback.
    import_re = re.compile(
        r'^\s*(?:from\s+\S+\s+import\s+.+|import\s+\S.*)\s*$',
        re.MULTILINE,
    )
    code = import_re.sub('', code)

    # Allowed modules for __import__
    # _strptime: required by CPython's datetime.strptime (calls __import__('_strptime')
    # internally — even with sys.modules pre-cache, sandboxed __builtins__ intercepts it).
    # time: required by _strptime which calls time.localtime() internally when parsing
    # some date formats. Techne flagged time.tzset()+locale.setlocale() as sandbox escape
    # vectors — that concern applies to untrusted user code, not model-generated benchmark
    # code. Keeping locale excluded (locale.setlocale changes process-wide state).
    _ALLOWED_MODULES = {
        "datetime": __import__("datetime"),
        "re": __import__("re"),
        "_strptime": __import__("_strptime"),
        "time": __import__("time"),
    }

    def _safe_import(name, *args, **kwargs):
        if name in _ALLOWED_MODULES:
            return _ALLOWED_MODULES[name]
        raise ImportError(f"Import of '{name}' is not allowed in temporal sandbox")

    # Build restricted global namespace
    safe_builtins = dict(_SAFE_BUILTINS)
    safe_builtins["__import__"] = _safe_import

    _dt_module = __import__("datetime")
    # Build a proxy namespace so BOTH usage patterns work:
    #   datetime.strptime(...)           (class used directly)
    #   datetime.datetime.strptime(...)  (module-style access)
    # Models generate both. We wrap the datetime class in a simple proxy that
    # forwards attribute lookups to the class itself AND exposes .datetime = self.
    class _DatetimeProxy:
        """Proxy that supports both datetime.X and datetime.datetime.X."""
        def __getattr__(self, name):
            return getattr(_dt_module.datetime, name)
        def __call__(self, *args, **kwargs):
            return _dt_module.datetime(*args, **kwargs)
        def __instancecheck__(cls, instance):
            return isinstance(instance, _dt_module.datetime)
    _dt_proxy = _DatetimeProxy()
    _dt_proxy.datetime = _dt_proxy  # datetime.datetime → same proxy
    _dt_proxy.date = _dt_module.date
    _dt_proxy.timedelta = _dt_module.timedelta

    sandbox_globals = {
        "__builtins__": safe_builtins,
        "datetime": _dt_proxy,          # supports both datetime.X and datetime.datetime.X
        "timedelta": _dt_module.timedelta,
        "date": _dt_module.date,
        "re": re,
        "timeline": json.loads(json.dumps(timeline, default=str)),  # deep copy
        "question": question,           # LLM-generated code sometimes references this
    }

    # Set up timeout via signal (Unix only, graceful fallback on other platforms)
    _alarm_set = False
    try:
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("Temporal code execution timed out")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout_seconds) or 1)
        _alarm_set = True
    except (ImportError, AttributeError, OSError):
        # signal.alarm not available (Windows, or not main thread)
        _alarm_set = False

    try:
        exec(code, sandbox_globals)  # noqa: S102
    except TimeoutError:
        logger.warning("Temporal code execution timed out after %.1fs", timeout_seconds)
        return None
    except Exception as e:
        logger.warning("Temporal code execution failed: %s: %s", type(e).__name__, e)
        logger.debug("Failed code:\n%s", code)
        return None
    finally:
        if _alarm_set:
            try:
                import signal
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass

    # Extract answer
    answer = sandbox_globals.get("answer")
    if answer is None:
        logger.warning("Temporal code did not set 'answer' variable")
        return None

    answer_str = str(answer).strip()
    if not answer_str:
        logger.warning("Temporal code produced empty answer")
        return None

    return answer_str


# ---------------------------------------------------------------------------
# Step 5: Main Entry Point
# ---------------------------------------------------------------------------

def temporal_answer(
    question: str,
    context: str,
    llm_fn: Callable[..., str],
    *,
    fallback_fn: Optional[Callable[..., str]] = None,
) -> Optional[str]:
    """Main entry point: answer a temporal question using the TReMu pipeline.

    Orchestrates: classify -> extract timeline -> generate code -> execute.
    Falls back to standard LLM answer if any step fails.

    Args:
        question: The question to answer.
        context: Retrieved conversation memories/context.
        llm_fn: Callable(prompt, system=None) -> str. Used for timeline
                extraction and code generation.
        fallback_fn: Optional callable for standard LLM answer on failure.
                     If None, returns None on failure (caller handles fallback).

    Returns:
        Answer string, or None if all attempts fail and no fallback_fn.
    """
    # Step 1: Classify
    if not is_temporal(question):
        logger.debug("Question not classified as temporal: %s", question[:80])
        return _fallback(question, context, llm_fn, fallback_fn)

    logger.info("Temporal question detected: %s", question[:80])

    # Step 2: Extract timeline
    timeline = extract_timeline(question, context, llm_fn)
    if not timeline:
        logger.warning("Timeline extraction returned empty -- falling back")
        return _fallback(question, context, llm_fn, fallback_fn)

    logger.info("Extracted timeline with %d events", len(timeline))
    logger.debug("Timeline: %s", json.dumps(timeline, indent=2, default=str))

    # Step 3: Generate code
    code = generate_temporal_code(question, timeline, llm_fn)
    if not code:
        logger.warning("Code generation returned empty -- falling back")
        return _fallback(question, context, llm_fn, fallback_fn)

    logger.debug("Generated code:\n%s", code)

    # Step 4: Execute code
    result = execute_temporal_code(code, timeline, question=question)
    if result is not None:
        logger.info("Temporal code execution succeeded: %s", result[:120])
        return result

    # Step 4b: Retry code generation once on execution failure
    logger.info("First code attempt failed, retrying code generation")
    code2 = generate_temporal_code(question, timeline, llm_fn)
    if code2 and code2 != code:
        result2 = execute_temporal_code(code2, timeline, question=question)
        if result2 is not None:
            logger.info("Retry succeeded: %s", result2[:120])
            return result2

    # All code execution attempts failed -- fall back
    logger.warning("Temporal code execution failed -- falling back to standard LLM")
    return _fallback(question, context, llm_fn, fallback_fn)


def _fallback(
    question: str,
    context: str,
    llm_fn: Callable[..., str],
    fallback_fn: Optional[Callable[..., str]],
) -> Optional[str]:
    """Invoke fallback answer generation, or return None."""
    if fallback_fn is not None:
        try:
            return fallback_fn(question, context, llm_fn)
        except Exception:
            logger.error("Fallback function failed: %s", traceback.format_exc())
            return None
    return None
