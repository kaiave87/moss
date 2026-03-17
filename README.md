# Moss

**Neuroplastic memory retrieval for conversational AI systems.**

Moss is the open-source reference implementation of three patented techniques for long-term conversational memory:

1. **Hebbian Recall** — Spreading activation across a co-activation pathway graph. Memories that are retrieved together develop stronger associative links, surfacing contextually coherent clusters beyond the direct similarity horizon.

2. **TReMu** — Temporal Reasoning via Multi-stage code execution. Classifies temporal questions, extracts structured timelines via LLM, generates Python to compute answers, and executes in a sandboxed environment.

3. **Multi-Channel RRF** — Weighted Reciprocal Rank Fusion across heterogeneous retrieval channels (lexical BM25, dense vector, graph traversal, temporal, cross-encoder reranker). Per-channel weights are adapted based on query category.

## Why Moss?

Static memory stores what you tell it. Neuroplastic memory learns from what it stores.

Moss addresses the core failure modes of retrieval-augmented memory:
- **Recency bias** — flat vector search buries older relevant memories
- **Temporal blindness** — LLMs hallucinate date arithmetic; TReMu computes it
- **Single-channel brittleness** — keyword queries miss semantic matches and vice versa; RRF fuses both

## Installation

```bash
pip install moss-memory
# or from source:
pip install -e .
```

**Dependencies:** `rank_bm25` (for BM25 channel). All other modules are stdlib-only.

## Quick Start

### Hebbian Recall

```python
from moss.hebbian.hebbian_recall import HebbianMemoryStore

# Plug in any embedding function
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embed = lambda text: model.encode(text).tolist()

store = HebbianMemoryStore(embed_func=embed, spreading_depth=2)

# Add memories
store.add("m1", "Alice went hiking last Saturday")
store.add("m2", "Alice loves the outdoors")
store.add("m3", "Bob stayed home last weekend")

# Co-activate m1 and m2 to wire them together
store.strengthen_pathway("m1", "m2")

# Recall — m2 surfaces via Hebbian pathway even without direct query match
results = store.recall("What did Alice do on the weekend?", limit=5)
for r in results:
    print(f"{r.memory_id}: score={r.score:.3f} source={r.source} boost={r.hebbian_boost:.3f}")
```

### Multi-Channel RRF

```python
from moss.rrf.reciprocal_rank_fusion import fuse, RRFConfig

# Each channel returns (doc_id, score) pairs
lexical  = [("doc1", 0.9), ("doc3", 0.7)]
semantic = [("doc2", 0.95), ("doc1", 0.8)]
temporal = [("doc1", 0.85), ("doc3", 0.6)]

config = RRFConfig(lexical_weight=1.0, semantic_weight=1.0, temporal_weight=2.0)
fused = fuse(
    channels={"lexical": lexical, "semantic": semantic, "temporal": temporal},
    config=config,
    top_k=5,
)
for r in fused:
    print(r.doc_id, f"{r.rrf_score:.4f}")
```

### TReMu (Temporal Reasoning)

```python
from moss.tremu.temporal_reasoning import temporal_answer, is_temporal

# Provide any LLM function: (prompt, system=None) -> str
def my_llm(prompt, system=None):
    ...  # call OpenAI, Ollama, Anthropic, etc.

question = "How many days passed between Alice's hiking trip and Bob's party?"
context = """
[15 March 2025] Alice went hiking.
[22 March 2025] Bob threw a party and Alice attended.
"""

if is_temporal(question):
    answer = temporal_answer(question, context, my_llm)
    print(answer)  # "7 days"
```

## Architecture

```
moss/
├── hebbian/
│   └── hebbian_recall.py    # HebbianMemoryStore, spreading activation
├── rrf/
│   ├── bm25_rrf.py          # BM25Index for lexical channel
│   └── reciprocal_rank_fusion.py  # fuse(), RRFConfig, FusedResult
└── tremu/
    └── temporal_reasoning.py  # temporal_answer(), is_temporal()
```

## Patents

The techniques in this repository are covered by patent applications filed with the Canadian Intellectual Property Office (CIPO) on March 17, 2026:

- **Patent 1**: Hebbian Memory with Spreading Activation for Conversational AI
- **Patent 2**: TReMu — Temporal Reasoning via Multi-stage Code Execution
- **Patent 3**: Multi-Channel Reciprocal Rank Fusion with Adaptive Weighting

This code is published as a reference implementation to substantiate the patent claims and to enable the research community to reproduce and build upon this work.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{moss2026,
  author = {Lichen Research Inc.},
  title  = {Moss: Neuroplastic Memory Retrieval},
  year   = {2026},
  url    = {https://github.com/kaiavery87/moss},
}
```
