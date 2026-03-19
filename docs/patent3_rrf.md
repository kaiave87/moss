# Patent 3: Multi-Channel Reciprocal Rank Fusion

**Status:** Canadian patent application filed March 2026 (patent pending).
US provisional application pending.

---

## What It Is

Multi-Channel Reciprocal Rank Fusion (RRF) is a hybrid retrieval system that
combines results from multiple independent retrieval channels — full-text search,
vector similarity, graph traversal, and cross-encoder reranking — into a single
ranked result list using the Reciprocal Rank Fusion algorithm.

The patent covers the multi-channel architecture, the weighted fusion formula,
and the integration of graph-based retrieval as a first-class channel alongside
lexical and semantic channels.

---

## The Fusion Formula

The core RRF formula (Cormack, Clarke & Buettcher 2009):

```
RRF(d) = Σ_s [ w_s * (1 / (k + rank_s(d))) ]
```

Where:
- `d` = document
- `s` = retrieval source (channel)
- `rank_s(d)` = rank of document `d` in source `s` (1-indexed)
- `k` = smoothing constant (default 60, empirically optimal)
- `w_s` = per-channel weight

The `k=60` constant dampens the influence of extreme rank differences.
A document ranked 1st contributes `1/61 ≈ 0.016` per channel; one ranked
100th contributes `1/160 ≈ 0.006`. Documents appearing across multiple
channels accumulate scores additively.

---

## Why Multiple Channels

Different retrieval modalities have complementary strengths:

| Channel | Strength | Weakness |
|---------|----------|----------|
| Full-text (BM25) | Exact lexical match, named entities, dates | Misses paraphrase |
| Vector similarity | Semantic understanding, paraphrase | Misses exact terms |
| Graph traversal | Associative links, multi-hop reasoning | Sparse for new topics |
| Cross-encoder | Full query-document attention, most accurate | Slow, cannot rank all |

RRF fusion exploits complementarity: a document must be relevant in multiple
ways to rank highly. This substantially outperforms any single channel alone,
particularly on multi-hop and temporal query categories.

---

## Six-Channel Configuration

The production configuration uses six retrieval channels:

1. **BM25 lexical search** — keyword overlap with Okapi BM25 scoring
2. **Dense vector search** — cosine similarity of 4096-dimensional embeddings
3. **Graph traversal** — entity co-occurrence links from the memory graph
4. **Cross-encoder reranking** — query-document joint attention model
5. **Rule retrieval** — boosted retrieval of extracted procedural knowledge
6. **Wisdom nodes** — synthesis nodes surfaced via entity bridge queries

---

## LoCoMo Benchmark Results

The multi-channel RRF architecture was validated on the LoCoMo benchmark
(1,986 questions across 45 long-term conversations):

| Category | Score |
|----------|-------|
| Overall | **76.6%** |
| Adversarial | 95.0% |
| Common-sense | 88.5% |
| Multi-hop | 74.3% |
| Temporal | 67.6% |
| Single-hop | 59.6% |

The BM25 + vector RRF fusion was particularly impactful for temporal queries
(date mentions, named entities) that embedding similarity alone under-retrieves.

---

## References

- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank
  fusion outperforms Condorcet and individual relevance feedback methods.
  *SIGIR 2009.*
- Nogueira & Cho (2019). Passage re-ranking with BERT. *arXiv:1901.04085.*
- Robertson & Zaragoza (2009). The probabilistic relevance framework: BM25
  and beyond. *Foundations and Trends in IR.*
