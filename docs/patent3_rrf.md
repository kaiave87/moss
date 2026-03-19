# Patent 3: Multi-Channel RRF with Intent-Guided Weighting

## What It Is

Any single retrieval signal fails in specific conditions:
- Vector similarity misses exact keyword matches ("what did John say about BERT?")
- BM25 misses paraphrases ("what was discussed about the language model?")
- Temporal search misses non-temporal facts
- Graph search misses isolated memories with no linked context

Multi-Channel RRF (Reciprocal Rank Fusion) fuses up to 6 retrieval channels into a single ranked list. The key innovation is **intent-guided weighting**: different question types use different channel weights, computed automatically from query classification.

## The Six Channels

| Channel | Method | Strength | Weakness |
|---------|--------|----------|----------|
| Lexical (BM25) | Keyword overlap | Exact terms, named entities | Paraphrases |
| Semantic (vector) | Cosine similarity | Paraphrases, semantics | Rare terms |
| Graph | Pathway traversal | Co-referenced entities | Isolated memories |
| Temporal | Date-range filter | Time-bounded queries | Non-temporal |
| Entity-scoped | Entity + vector | Specific-person queries | Multi-hop |
| Cross-encoder | Neural reranker | Precise relevance | Slow, post-processing |

## RRF Formula

Reciprocal Rank Fusion (Cormack et al. 2009):

```
score(doc) = Σ_channel  w_channel / (k + rank_channel(doc))
```

Where:
- `k = 60` (empirically optimal, reduces influence of rank-1 dominance)
- `w_channel` = per-channel weight (intent-guided)
- `rank_channel(doc)` = doc's rank within that channel (1-indexed, ∞ if not present)

## Intent-Guided Weighting (MAGMA Pattern)

Query classification determines which channels dominate:

```python
CAT_ROUTER = {
    'multi-hop':    {'recall': 30, 'temporal_w': 0.5, 'entity_w': 1.5, 'bm25_w': 1.5, 'escoped_w': 1.5},
    'temporal':     {'recall': 35, 'temporal_w': 3.0, 'entity_w': 1.0, 'bm25_w': 2.0, 'escoped_w': 1.0},
    'single-hop':   {'recall': 22, 'temporal_w': 0.3, 'entity_w': 1.5, 'bm25_w': 1.2, 'escoped_w': 2.0},
    'common-sense': {'recall': 25, 'temporal_w': 0.5, 'entity_w': 1.5, 'bm25_w': 0.5, 'escoped_w': 1.5},
    'adversarial':  {'recall': 20, 'temporal_w': 0.3, 'entity_w': 0.5, 'bm25_w': 0.3, 'escoped_w': 0.5},
}
```

Temporal questions: `temporal_w=3.0` (date-range filtering dominates)  
Multi-hop questions: balanced weights, high recall (need chain-of-fact)  
Adversarial questions: low weights everywhere (avoid false retrieval)  

## Benchmark Results

On LoCoMo-10 benchmark (1986 questions, GPT-4o judge):

| Category | Score |
|----------|-------|
| Adversarial | 96.0% |
| Multi-hop | 74.3% |
| Temporal | 72.9% |
| Common-sense | 69.8% |
| Single-hop | 59.6% |
| **Overall** | **76.6%** |

## What Is Novel

1. **Six-channel fusion** (not two): Adding graph, temporal, and entity-scoped channels alongside the standard vector+BM25 pair
2. **Intent-guided weight routing**: Weights computed from query classification, not fixed globally
3. **Per-category recall limits**: Different question types need different retrieval breadth
4. **Integration with Hebbian recall**: Graph channel uses the Hebbian pathway graph (Patent 1)

## Prior Art

**Standard RAG (Lewis et al. 2020):** Single dense retrieval channel. No fusion, no intent routing.
**BM25 + vector hybrid (many implementations):** Two channels, equal weights. No intent classification.
**RAG-Fusion:** Re-ranks multiple generated queries, not multiple retrieval channels.
**ColBERT / PLAID:** Late interaction retrieval. Single channel, no fusion.

## Likely Examiner Questions

**Q: RRF has been around since 2009. What's novel?**
A: RRF is a known technique for combining ranked lists. Our contribution is: (1) extending it to 6 heterogeneous channels including a Hebbian pathway graph, (2) routing channel weights based on automated query intent classification, and (3) applying it in a conversational long-term memory context.

**Q: How is intent-guided weighting different from standard query expansion?**
A: Query expansion adds terms to the query. Intent-guided weighting changes the relative importance of different retrieval signals. A temporal question needs date-range filtering to dominate; an adversarial question needs all signals suppressed to avoid hallucinated retrieval.

**Q: Is there prior art for query-dependent retrieval fusion weights?**
A: Learning-to-rank methods learn weights from labeled data. Our system derives weights from rule-based query classification (no training required) applied to a fixed channel vocabulary. The specific combination for conversational memory is novel.
