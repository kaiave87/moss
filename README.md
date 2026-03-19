# Moss — Neuroplastic AI Memory System

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Patent Pending](https://img.shields.io/badge/patent-pending-orange.svg)](NOTICE)

**Three Canadian patent applications filed March 2026 (patent pending).
US provisional applications pending.**

Moss is a neuroplastic AI memory system implementing three patented algorithms:

1. **Hebbian Recall** — Memory that learns from its own retrieval patterns
2. **TReMu** — Neuro-symbolic temporal reasoning for time-sensitive questions
3. **Multi-Channel RRF** — Reciprocal Rank Fusion across six retrieval channels

---

## Modules

### `moss.hebbian` — Hebbian Neuroplastic Memory

Implements Hebb's postulate ("neurons that fire together wire together") for AI
memory retrieval. Memories that are co-retrieved form persistent pathways that
strengthen over time, enabling associative recall beyond simple vector similarity.

Key features:
- **Spreading activation** — graph traversal from seed memories to related ones
- **Four-channel strengthening** — event-driven, rapid consolidation, sleep
  consolidation, and spacing-effect channels
- **Reconsolidation lability** — retrieved pathways enter a plasticity window
- **Lateral inhibition** — near-duplicate suppression for diverse results

### `moss.tremu` — Temporal Reasoning Module

Neuro-symbolic approach to temporal questions. Detects questions that require
timeline reasoning, extracts structured event timelines via an LLM, then
generates and executes Python code to compute the answer precisely.

Based on TReMu (ACL 2025, arXiv:2502.01630). Proven improvement from 29.8%
to 77.7% on temporal question categories.

### `moss.rrf` — Multi-Channel Reciprocal Rank Fusion

Combines BM25 lexical search, dense vector search, graph traversal, and
cross-encoder reranking into a single ranked list using Reciprocal Rank Fusion
(Cormack et al. 2009).

RRF formula: `score(d) = Σ w_s * (1 / (k + rank_s(d)))` where k=60.

---

## Installation

```bash
pip install -e .
```

Or from PyPI:

```bash
pip install moss-memory
```

---

## Quick Examples

### Hebbian Recall

```python
from moss.hebbian.db import MockDB
from moss.hebbian.embeddings import get_embedding
from moss.hebbian.recall import recall

# Set up in-memory store (swap MockDB for your production backend)
db = MockDB()
db.store_memory("Paris is the capital of France.", embedding=get_embedding("Paris is the capital of France."))
db.store_memory("The Eiffel Tower is in Paris.", embedding=get_embedding("The Eiffel Tower is in Paris."))

# Recall — memories retrieved together strengthen their pathways
results = recall("Tell me about Paris", db=db, embed_fn=get_embedding, limit=5)
for r in results:
    print(f"{r.score:.3f} [{r.source}] {r.content}")
```

### TReMu Temporal Reasoning

```python
from moss.tremu import temporal_answer, is_temporal

# Your LLM function (any callable that takes a prompt string)
def my_llm(prompt, system=None):
    # ... call your LLM here ...
    return response_text

context = """
[15 Jan 2024] Alice and Bob got married.
[20 Feb 2024] Alice started a new job.
[01 Mar 2024] They moved to a new city.
"""

question = "How many days passed between the wedding and the move?"

if is_temporal(question):
    answer = temporal_answer(question, context, my_llm)
    print(answer)  # "45 days"
```

### Multi-Channel RRF

```python
from moss.rrf.db import MockDB
from moss.rrf.bm25_rrf import build_bm25_index, hybrid_search

documents = [
    {"id": "1", "content": "Paris is the capital of France.", "title": "Paris"},
    {"id": "2", "content": "Berlin is the capital of Germany.", "title": "Berlin"},
    {"id": "3", "content": "Machine learning uses statistical models.", "title": "ML"},
]

# Build BM25 index
bm25 = build_bm25_index(documents)

# Simulate vector results (use your own embedding model in production)
vector_results = [{"id": "3", "content": documents[2]["content"], "score": 0.9}]

# Fuse BM25 + vector results via RRF
fused = hybrid_search("machine learning statistics", vector_results, bm25, top_k=3)
for r in fused:
    print(f"rrf={r['rrf_score']:.4f} | {r.get('title', r['id'])}")
```

---

## Benchmark Results

Evaluated on the LoCoMo long-term conversation memory benchmark (1,986 questions):

| Category | Score |
|----------|-------|
| **Overall** | **76.6%** |
| Adversarial | 95.0% |
| Common-sense | 88.5% |
| Multi-hop | 74.3% |
| Temporal | 67.6% |
| Single-hop | 59.6% |

Benchmark details: `benchmarks/locomo_results.json`

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

Copyright 2026 Lichen Research Inc.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## Patent Notice

See [NOTICE](NOTICE) for patent information.
