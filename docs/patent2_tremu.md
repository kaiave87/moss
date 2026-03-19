# Patent 2: TReMu — Temporal Reasoning via Multi-stage Code Execution

## What It Is

Large language models fail at date arithmetic. Asked "how many days between June 15 and September 20?", a model trained on text will guess or hallucinate. The correct answer requires computation, not pattern matching.

TReMu (Temporal Reasoning via Multi-stage code execution) is a neuro-symbolic pipeline:
1. **Classify:** Is this a temporal question?
2. **Extract:** Extract a structured timeline from the retrieved context (via LLM → JSON)
3. **Generate:** Generate Python code to compute the answer from the timeline (via LLM → code)
4. **Execute:** Run the code in a sandboxed environment, return the result

The LLM handles language → structure conversion (what it's good at). Python handles computation (what it's good at).

## Why It Matters

Without TReMu, on the LoCoMo temporal benchmark category:
- GPT-4o answers: ~25% accuracy (hallucinated dates)
- With TReMu pipeline: **77.7% accuracy** (+52.7pp)

The key insight: temporal questions fail not because the context is missing, but because the LLM can't reliably do date arithmetic. Offloading arithmetic to Python eliminates the failure mode.

## Core Pipeline

```python
def temporal_answer(question, context, llm_fn, fallback_fn=None):
    if not is_temporal(question):
        return fallback(...)

    # Step 2: Extract structured timeline
    timeline = extract_timeline(question, context, llm_fn)
    # timeline = [{"event": "...", "date": "2023-06-15", "confidence": "exact"}, ...]

    # Step 3: Generate computation code
    code = generate_temporal_code(question, timeline, llm_fn)
    # code = "from datetime import date\nresult = ...\nprint(result)"

    # Step 4: Execute in sandbox
    result = execute_temporal_code(code, timeline)
    return result or fallback(...)
```

## Sandboxing

The generated Python code runs in a restricted namespace:
- **Allowed:** `datetime`, `date`, `timedelta`, `math`, basic builtins
- **Blocked:** `os`, `sys`, `open`, `exec`, `eval`, `import`, `__builtins__`
- **Timeout:** 5 seconds (prevents infinite loops)
- **No file I/O, no network access, no process spawning**

The restriction prevents code injection. The LLM can only compute dates — it cannot exfiltrate data or execute system commands.

## What Is Novel

The combination of:
1. **LLM-driven timeline extraction** → structured JSON (not brittle regex)
2. **LLM-driven code generation** from the structured timeline
3. **Sandboxed Python execution** with date-focused restriction policy
4. **Fallback path** when pipeline fails (graceful degradation)

As applied to **conversational memory systems** (not one-shot QA).

## Prior Art

Based on TReMu (ACL 2025, arxiv 2502.01630). Our contribution is applying this to conversational long-term memory retrieval and integrating it with the RRF retrieval pipeline.

**Chain-of-thought (Wei et al.):** Text-based reasoning. No code execution. Fails at arithmetic.
**Program synthesis (AlphaCode, Copilot):** Code generation, but not in a temporal QA loop with sandboxed execution and fallback.
**PAL (Program-aided Language models, Gao et al. 2022):** Closest prior art. General program-aided approach. TReMu is specialized for temporal/date reasoning in conversational memory context, with the sandboxing and fallback design.

## Likely Examiner Questions

**Q: How is this different from PAL?**
A: PAL generates code for general arithmetic. TReMu is specialized for temporal reasoning in conversational memory: the timeline extraction step (LLM → structured JSON) is domain-specific, and the sandbox restriction policy is date-focused. The integration with multi-channel RRF retrieval is also novel.

**Q: Why is a patent appropriate if it's based on an ACL 2025 paper?**
A: The ACL paper establishes the approach. Our patent covers the specific implementation: sandboxed execution policy, integration with Hebbian recall and RRF retrieval, the conversational memory application, and the fallback path design.

**Q: What prevents others from implementing this themselves?**
A: The patent covers our specific implementation decisions. Others can implement TReMu-style approaches for different domains. The patent protects our particular combination in conversational memory systems.
