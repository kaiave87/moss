# Patent 1: Hebbian Recall with Spreading Activation

## What It Is

Traditional memory retrieval treats each stored item independently: you embed a query, compute cosine similarity against all items, and return the top-k. There is no memory of which items were previously retrieved together.

Hebbian Recall adds a pathway graph on top of vector search. When two memories are retrieved together (co-activated), the pathway between them is strengthened. Future queries can then traverse these pathways via spreading activation, surfacing memories beyond the direct similarity horizon.

**The Hebbian principle:** "Neurons that fire together wire together." In our system: memories that retrieve together strengthen together.

## Core Algorithm

```
1. Embed query → retrieve top-k direct candidates (vector similarity)
2. For each candidate: traverse pathway graph up to `spreading_depth` hops
3. Activated memories accumulate hebbian_boost ∝ pathway_strength × decay^depth
4. Final ranking = vector_score + hebbian_boost
5. After retrieval: strengthen_pathway(recalled_a, recalled_b) for all pairs
```

**Pathway update rule:**
```python
strength += learning_rate * (1 - strength)   # asymptotic approach to 1.0
strength *= (1 - decay_rate)                 # periodic decay
```

## What Is Novel

Prior work uses static vector similarity (FAISS, pgvector) or explicit knowledge graphs (structured triples). Moss introduces:

1. **Co-activation learning:** Pathways form and strengthen automatically from retrieval patterns — no manual annotation.
2. **Spreading activation with decay:** Signal propagates multi-hop through the pathway graph with exponential decay, preventing noise explosion.
3. **Asymptotic update rule:** Pathways approach 1.0 asymptotically, never exceeding it. Old pathways decay unless reinforced.
4. **Retrieval-induced learning:** The retrieval act itself is the training signal. No separate training phase.

## Implementation

See `moss/hebbian/hebbian_recall.py` → `HebbianMemoryStore`.

Key classes:
- `HebbianMemoryStore`: the full store with add/recall/strengthen/decay
- `MemoryEntry`: a stored memory with content + embedding
- `MemoryPathway`: a directed pathway with strength + activation_count

## Prior Art Landscape

- **FAISS / pgvector:** Pure vector similarity. No pathway learning. No spreading activation.
- **Knowledge graphs (Wikidata, ConceptNet):** Static, manually annotated. Not derived from retrieval patterns.
- **Memory-augmented neural networks (NTM, DNC):** Write/read heads update weights globally. Not per-item pathway learning.
- **Hopfield networks (modern: Ramsauer et al. 2020):** Energy minimization, not graph traversal. Different mechanism.
- **Associative retrieval (TF-IDF, BM25):** Lexical matching. No vector similarity, no pathway graph.

Our combination — vector store + Hebbian pathway graph derived from retrieval co-activation + spreading activation at query time — is not present in prior art.

## Likely Examiner Questions

**Q: Isn't this just a knowledge graph?**
A: No. Knowledge graphs are statically constructed from structured data. Our pathways emerge dynamically from retrieval patterns — they're not programmed in, they're learned from use.

**Q: How is this different from collaborative filtering?**
A: Collaborative filtering recommends items based on co-preference across users. We strengthen pathways based on co-retrieval within a single conversation memory system, with no user preference signal.

**Q: Doesn't LangChain's memory module do something similar?**
A: LangChain's memory stores conversation history. It does not build a pathway graph, does not apply spreading activation at retrieval time, and does not learn from co-retrieval patterns.
