# Patent 2: TReMu — Temporal Reasoning Module

**Status:** Canadian patent application filed March 2026 (patent pending).
US provisional application pending.

---

## What It Is

TReMu (Temporal Reasoning Module) is a neuro-symbolic approach to answering
time-sensitive questions about long-term conversation history. It combines
LLM-based timeline extraction with sandboxed Python code generation and
execution to compute precise temporal answers.

---

## The Problem It Solves

Standard retrieval-augmented generation (RAG) fails on temporal questions
because:

1. **Retrieval retrieves context, not computations.** Knowing that "the trip
   was 3 weeks after the birthday" doesn't directly answer "how many days
   between the birthday and the graduation?"

2. **LLMs hallucinate dates.** Without structured intermediate representation,
   LLMs produce confident but incorrect date arithmetic.

3. **Relative references chain unpredictably.** "She called yesterday"
   relative to a session from six months ago requires context-aware resolution.

---

## How It Works

TReMu uses a three-stage neuro-symbolic pipeline:

### Stage 1: Temporal Detection
A pattern-based classifier determines whether the question requires temporal
reasoning. Checks for keywords (`"when did"`, `"how long ago"`, `"before or
after"`) and date-like patterns. Sub-millisecond, no LLM call.

### Stage 2: Timeline Extraction (LLM)
The LLM is prompted to extract a structured timeline from retrieved memories:

```json
[
  {"event": "Alice and Bob married", "date": "2024-01-15", "confidence": "exact"},
  {"event": "Alice started new job",  "date": "2024-02-20", "confidence": "exact"}
]
```

Relative date references (`"last Tuesday"`) are resolved using session-date
context embedded in memory content.

### Stage 3: Code Generation + Sandboxed Execution
The LLM generates Python code that computes the answer from the timeline:

```python
from datetime import datetime
married = datetime(2024, 1, 15)
job_start = datetime(2024, 2, 20)
delta = (job_start - married).days
answer = f"{delta} days after the wedding"
```

The code runs in a restricted sandbox (no file I/O, no network, no imports
beyond stdlib). The `answer` variable is extracted as the final response.

---

## Patent Claims Summary

The key novelty over prior art (TReMu, ACL 2025) is the integration with
persistent Hebbian memory, enabling:

1. Temporal reasoning over **long-term** conversation history (months/years)
   rather than a single session context window.

2. Timeline extraction that leverages **pathway-strengthened** memories —
   frequently accessed events surface more reliably.

3. **Fallback escalation**: if code execution fails, the system degrades
   gracefully to direct LLM answer generation.

---

## Benchmark Impact

On LoCoMo temporal category:

| System | Temporal Score |
|--------|---------------|
| Baseline RAG | 29.8% |
| TReMu-enhanced | **67.6%** |

Improvement: **+37.8 percentage points** on temporal questions.

---

## References

- Su et al. (2025). TReMu: Temporal Reasoning for Long-context Memory.
  *ACL 2025*, arXiv:2502.01630.
- Maharana et al. (2024). Evaluating Very Long-Term Conversational Memory.
  *EACL 2024* (LoCoMo benchmark).
