# Patent 1: Hebbian Recall — Neuroplastic AI Memory

**Status:** Canadian patent application filed March 2026 (patent pending).
US provisional application pending.

---

## What It Is

Hebbian Recall is a memory retrieval system for AI agents that goes beyond
static vector similarity search. Instead of simply returning documents most
similar to a query, it learns from retrieval patterns and strengthens
associations between memories that are frequently retrieved together.

The name derives from Hebb's postulate (1949): *"Neurons that fire together
wire together."* Applied to AI memory: memories retrieved together form
learned pathways that improve future retrieval.

---

## How It Maps to the Patent

The patent claims cover three interconnected innovations:

### 1. Co-Activation Pathway Graph
A graph of `memory_pathways` grows dynamically as memories are co-retrieved.
Each edge stores a `strength` value that increases with co-activation frequency
and decays with disuse (temporal homeostasis). This is analogous to long-term
potentiation (LTP) in biological neural networks.

### 2. Spreading Activation Retrieval
Given a set of seed memories from vector search, the system performs graph
traversal through the pathway graph. Activation energy spreads from seeds to
connected memories, decayed by:
- Hop distance (configurable decay factor)
- Pathway strength threshold
- Query intent (intent-guided traversal modulation)

This surfaces memories that are *related* (graph-connected) rather than merely
*similar* (vector-close) to the query.

### 3. Four-Channel Strengthening
Pathway strength is updated through four channels inspired by neuroscience:
1. **Event-driven (STDP)** — co-retrieval strengthening, immediate
2. **Rapid consolidation (E-LTP → L-LTP)** — phase-aware passive boost
3. **Sleep consolidation (SWR + STC)** — overnight replay and capture
4. **Spacing effect (Cepeda ridgeline)** — age-dependent review intervals

---

## Key Algorithm

```python
# Simplified Hebbian recall pipeline
embedding = embed_fn(query)                     # 1. Embed query
seeds = db.search_by_vector(embedding, k=5)    # 2. Find seed memories
activated = spreading_activation(              # 3. Graph traversal
    seed_ids=[s.id for s in seeds],
    db=db, depth=2, decay=0.5,
)
results = merge_and_rank(seeds, activated)     # 4. Merge + rank
strengthen_batch(result_ids, db, boost=0.1)    # 5. Reinforce pathways
```

The pathway graph is persistent across sessions. Over time, high-quality
associations accumulate strength and low-quality ones decay, creating a
self-organising memory topology.

---

## LoCoMo Benchmark Results

On the LoCoMo long-term conversation memory benchmark (1,986 questions):

| Category | Score |
|----------|-------|
| Overall | **76.6%** |
| Adversarial | 95.0% |
| Common-sense | 88.5% |
| Multi-hop | 74.3% |
| Temporal | 67.6% |
| Single-hop | 59.6% |

---

## References

- Hebb, D. O. (1949). *The Organization of Behavior.* Wiley.
- Frey & Morris (1997). Synaptic tagging and long-term potentiation. *Nature.*
- Nader (2000). Memory traces unbound: reconsolidation. *Nature.*
- Cepeda et al. (2006). Distributed practice in verbal recall tasks. *Psych Bull.*
- Remme & Wadman (2012). Homeostatic scaling of excitability. *PLOS Comp Bio.*
