"""
Hebbian Recall — Graph-Enhanced Memory Retrieval

Combines vector similarity search with spreading activation through the
co-activation pathway graph to find both SIMILAR memories (vector search)
and RELATED memories (graph traversal).

The recall pipeline:
    1. Classify query intent (query_understanding)
    2. Rules-first retrieval (extracted_rule content_type)
    3. Vector search for seed memories with lateral inhibition
    4. Entity-anchor multi-hop retrieval (HippoRAG-inspired)
    5. Wisdom node retrieval via entity-memory links
    6. Spreading activation through pathway graph
    7. Semantic lateral inhibition (Jaccard overlap suppression)
    8. Score normalisation, temporal decay, recency boost
    9. ACT-R activation boost (Anderson & Schooler 1991)
    10. Near-duplicate collapse
    11. MMR diversity reranking (Carbonell & Goldstein 1998)
    12. Token budget enforcement
    13. Hebbian reinforcement of co-retrieved pathways
    14. Reconsolidation lability (Nader 2000)
    15. Access tracking

Patent pending — Lichen Research Inc., Canadian application filed March 2026.

Usage:
    from moss.hebbian.recall import recall

    db = MockDB()
    embed_fn = get_embedding  # from moss.hebbian.embeddings

    results = recall("what did we discuss about Paris?", db=db, embed_fn=embed_fn)
    for r in results:
        print(r.score, r.content)
"""

import hashlib
import logging
import math
import re
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable defaults (production: replace with a config/parameter store)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "sm_dampening": 0.4,           # Session Mining score dampening factor
    "diversity_pool_multiplier": 4, # Candidate pool = limit * this
    "content_type_cap": 2,         # Max results per content_type in seeds
    "mmr_lambda": 0.7,             # MMR relevance vs diversity tradeoff
    "mmr_top_k": 10,               # Apply MMR to top K results
    "learning_rate": 0.1,          # Hebbian co-retrieval learning rate
    "labile_hours": 6,             # Reconsolidation lability window (hours)
    "labile_cap_per_epoch": 50,    # Max labile edges per epoch
}


@dataclass
class RecallResult:
    """A memory recall result with provenance."""
    memory_id: str
    content: str
    summary: Optional[str]
    score: float
    source: str          # 'direct', 'activated', 'rule', 'wisdom', 'entity_anchor'
    tier: str
    created_at: Optional[datetime]
    pathway_strength: Optional[float] = None
    integrity_ok: Optional[bool] = None
    is_stale: bool = False
    estimated_tokens: int = 0

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _verify_content_integrity(memory_id: str, content: str, stored_hash: Optional[str]) -> bool:
    """Verify content has not been tampered with (MD5 hash check).
    Returns True if OK, False if tampered, True if no hash stored (legacy data)."""
    if not stored_hash:
        return True
    computed = hashlib.md5(content.encode()).hexdigest()
    if computed != stored_hash:
        logger.warning(f"Integrity mismatch for memory {memory_id}")
        return False
    return True


def _extract_entities(text: str) -> list:
    """Extract named entities from text for multi-anchor retrieval.

    Uses spaCy if available, falls back to capitalized phrase heuristic.
    Returns deduplicated entity strings, max 4.
    """
    entities = []
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:512])
        entities = list({ent.text for ent in doc.ents
                         if ent.label_ in ('PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'EVENT', 'PRODUCT', 'FAC')})
    except Exception:
        cap = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        years = re.findall(r'\b(?:19|20)\d{2}\b', text)
        entities = list(dict.fromkeys(cap + years))
    return [e for e in entities if len(e) > 2][:4]


def recall(
    query: str,
    db,
    embed_fn: Callable[[str], List[float]],
    limit: int = 10,
    min_pathway_strength: float = 0.2,
    spreading_depth: int = 2,
    spreading_decay: float = 0.5,
    include_activated: bool = True,
    debug: bool = False,
    readonly: bool = False,
    exclude_archived: bool = True,
    exclude_noise: bool = True,
    max_tokens: Optional[int] = None,
    multi_hop: bool = False,
) -> List[RecallResult]:
    """
    Retrieve memories using hybrid vector + graph approach.

    Args:
        query: Natural language query.
        db: Database backend (MockDB or production equivalent).
        embed_fn: Embedding function: (str) -> List[float].
        limit: Max results to return.
        min_pathway_strength: Only traverse pathways stronger than this.
        spreading_depth: How many hops to spread (1-2 recommended).
        spreading_decay: Activation decays by this factor per hop.
        include_activated: Include memories found via pathway traversal.
        debug: Print debug information.
        readonly: Skip Hebbian pathway strengthening on recall.
        exclude_archived: Exclude archived-tier memories.
        exclude_noise: Exclude session_snapshot and self_observation content.
        max_tokens: Cap total estimated tokens across returned results.
        multi_hop: Enable entity-anchor multi-hop retrieval (HippoRAG-inspired).

    Returns:
        List of RecallResult, sorted by score descending.
    """
    from moss.hebbian.spreading_activation import spreading_activation
    from moss.hebbian.pathway_strengthening import strengthen_batch

    results: List[RecallResult] = []

    # ── Step 0: Intent classification ─────────────────────────────────────
    query_intent = None
    _rule_boost = 1.5
    _intent_params = None
    _qa_entities = []
    _qa_temporal = None
    try:
        from moss.hebbian.query_understanding import QueryAnalyzer
        _analyzer = QueryAnalyzer(use_llm=False)
        _intent_params = _analyzer.get_intent_params(query)
        query_intent = _intent_params.intent.value
        if _intent_params.confidence >= 0.2:
            spreading_depth = _intent_params.spreading_depth_mod
            spreading_decay = _intent_params.spreading_decay_mod
            _rule_boost = _intent_params.rule_boost

        # Extract entities + temporal info for downstream boosts
        _qa_entities = _analyzer._rule_based_entities(query)
        _qa_temporal = _analyzer._detect_temporal(query)

        if debug:
            print(f"Intent: {query_intent} (conf={_intent_params.confidence})"
                  f" depth={spreading_depth}, decay={spreading_decay}")
            if _qa_entities:
                print(f"  Entities: {[e.text for e in _qa_entities]}")
            if _qa_temporal:
                print(f"  Temporal: {_qa_temporal.raw_text} ({_qa_temporal.temporal_type.value})")
    except Exception as e:
        if debug:
            print(f"Intent classification failed (non-fatal): {e}")

    try:
        # ── Step 1: Embed query ────────────────────────────────────────────
        embedding = embed_fn(query)

        # ── Step 1.5: Rule retrieval (extracted_rule content_type) ─────────
        seed_ids = []
        seed_scores: Dict[str, float] = {}
        rule_ids = set()

        rule_candidates = db.search_by_content_type(
            "extracted_rule", embedding, similarity_threshold=0.5, limit=3
        )
        for row in rule_candidates:
            mem_id = row["id"]
            similarity = row.get("similarity", 0.0)
            rule_ids.add(mem_id)
            integrity = _verify_content_integrity(mem_id, row.get("content", ""), row.get("content_hash"))
            boosted_score = similarity * _rule_boost
            created_at = _parse_dt(row.get("created_at"))
            results.append(RecallResult(
                memory_id=mem_id,
                content=(row.get("content") or "")[:1500],
                summary=row.get("summary"),
                score=boosted_score,
                source="rule",
                tier=row.get("tier", "active"),
                created_at=created_at,
                integrity_ok=integrity,
                estimated_tokens=row.get("estimated_tokens", 0),
            ))
            seed_ids.append(mem_id)
            seed_scores[mem_id] = boosted_score

        if debug:
            print(f"Rules: {len(rule_candidates)} found")

        # ── Step 2: Vector search for seeds with lateral inhibition ────────
        # Biologically inspired: lateral inhibition prevents any single
        # content_type from dominating recall, analogous to surround
        # suppression in visual cortex.
        _sm_damp = DEFAULTS["sm_dampening"]
        _ct_cap = DEFAULTS["content_type_cap"]
        if _intent_params and hasattr(_intent_params, 'diversity_mod') and _intent_params.diversity_mod != 1.0:
            _ct_cap = max(1, int(_ct_cap * _intent_params.diversity_mod))
        _pool_size = max(5, 5 * DEFAULTS["diversity_pool_multiplier"])

        vector_candidates = db.search_by_vector(
            embedding,
            limit=_pool_size,
            exclude_ids=list(rule_ids),
            tier_filter=exclude_archived,
            noise_filter=exclude_noise,
        )

        # Extract query terms for keyword matching boost
        _query_terms = set(re.findall(r'\b[A-Za-z0-9][\w.-]*\b', query.lower()))

        # Score candidates with importance blending, keyword boost,
        # entity overlap, temporal relevance, and SM dampening
        scored_candidates = []
        for row in vector_candidates:
            mem_id = row["id"]
            content = row.get("content", "")
            similarity = row.get("similarity", 0.0)
            is_stale = bool(row.get("is_stale", False))
            importance = float(row.get("importance_score", 0.5))
            content_type = row.get("content_type", "general")
            created_at_raw = row.get("created_at")

            # Stale dampening
            effective_score = similarity * 0.8 if is_stale else similarity

            # Importance blending: 85% similarity + 15% importance
            effective_score = 0.85 * effective_score + 0.15 * importance

            # Keyword match boost — specific terms in both query and content
            if content and _query_terms:
                _content_lower = content[:500].lower()
                _matched = sum(1 for t in _query_terms if len(t) >= 2 and t in _content_lower)
                if _matched > 0:
                    _kw_boost = min(0.10, _matched * 0.025)  # 2.5% per match, max 10%
                    effective_score *= (1.0 + _kw_boost)

            # Entity overlap boost — proper noun/entity matches
            if _qa_entities and content:
                _entity_names = {e.text.lower() for e in _qa_entities}
                _content_lower_ent = content[:500].lower()
                _entity_hits = sum(1 for e in _entity_names if e in _content_lower_ent)
                if _entity_hits > 0:
                    effective_score *= (1.0 + 0.05 * _entity_hits)  # +5% per entity hit

            # Temporal relevance boost — memories created in query's time range
            if _qa_temporal and hasattr(_qa_temporal, 'start_date') and _qa_temporal.start_date and created_at_raw:
                try:
                    _mem_date = _parse_dt(created_at_raw)
                    if _mem_date:
                        _mem_date_naive = _mem_date.replace(tzinfo=None)
                        _t_start = _qa_temporal.start_date.replace(tzinfo=None)
                        _t_end = (_qa_temporal.end_date or datetime.now()).replace(tzinfo=None)
                        if _t_start <= _mem_date_naive <= _t_end:
                            effective_score *= 1.15  # 15% boost for temporal match
                except (ValueError, TypeError, AttributeError):
                    pass

            # Session Mining dampening: generic descriptions with high cosine to everything
            is_sm = bool(content and content.startswith('[Session Mining]'))
            if is_sm:
                effective_score *= _sm_damp

            scored_candidates.append({
                'row': row,
                'content_type': content_type,
                'effective_score': effective_score,
                'is_sm': is_sm,
            })

        # Greedy diversity selection with progressive penalty
        ct_counts: Dict[str, int] = {}
        seeds = []
        scored_candidates.sort(key=lambda c: c['effective_score'], reverse=True)
        for cand in scored_candidates:
            if len(seeds) >= 5:
                break
            ct = cand['content_type']
            ct_count = ct_counts.get(ct, 0)
            if ct_count >= _ct_cap:
                penalty = 0.5 ** (ct_count - _ct_cap + 1)
                cand['effective_score'] *= penalty
            seeds.append(cand)
            ct_counts[ct] = ct_count + 1

        seeds.sort(key=lambda c: c['effective_score'], reverse=True)
        seeds = seeds[:5]

        if debug:
            sm_in_seeds = sum(1 for s in seeds if s['is_sm'])
            print(f"Seeds: {len(seeds)} selected (SM={sm_in_seeds}/5, ct_cap={_ct_cap})")

        for cand in seeds:
            row = cand['row']
            mem_id = row["id"]
            effective_score = cand['effective_score']
            is_stale = bool(row.get("is_stale", False))
            integrity = _verify_content_integrity(mem_id, row.get("content", ""), row.get("content_hash"))
            created_at = _parse_dt(row.get("created_at"))

            results.append(RecallResult(
                memory_id=mem_id,
                content=(row.get("content") or "")[:1500],
                summary=row.get("summary"),
                score=effective_score,
                source="direct",
                tier=row.get("tier", "active"),
                created_at=created_at,
                integrity_ok=integrity,
                is_stale=is_stale,
                estimated_tokens=row.get("estimated_tokens", 0),
            ))
            seed_ids.append(mem_id)
            seed_scores[mem_id] = effective_score

        # ── Step 2.5: Entity-anchor multi-hop retrieval (HippoRAG-inspired) ─
        # Extract named entities, search for each as an anchor, then
        # spreading activation from those anchors finds chained memories.
        if multi_hop and not readonly:
            _entities = _extract_entities(query)
            if debug:
                print(f"Multi-hop: extracted entities={_entities}")
            for _entity in _entities[:3]:
                try:
                    _entity_emb = embed_fn(_entity)
                    entity_hits = db.search_by_vector(
                        _entity_emb, limit=3, exclude_ids=seed_ids
                    )
                    for erow in entity_hits:
                        esim = erow.get("similarity", 0.0)
                        if esim < 0.45:
                            continue
                        _eid = erow["id"]
                        if _eid not in seed_scores:
                            _escore = esim * 0.85  # slight discount vs direct query match
                            seed_ids.append(_eid)
                            seed_scores[_eid] = _escore
                            results.append(RecallResult(
                                memory_id=_eid,
                                content=(erow.get("content") or "")[:1500],
                                summary=erow.get("summary"),
                                score=_escore,
                                source='entity_anchor',
                                tier=erow.get("tier", "active"),
                                created_at=_parse_dt(erow.get("created_at")),
                                integrity_ok=_verify_content_integrity(
                                    _eid, erow.get("content", ""), erow.get("content_hash")),
                                is_stale=bool(erow.get("is_stale", False)),
                                estimated_tokens=erow.get("estimated_tokens", 0),
                            ))
                except Exception as _e:
                    if debug:
                        print(f"Entity anchor failed for '{_entity}': {_e}")

        # ── Step 2.7: Wisdom node retrieval via entity-memory links ────────
        # Wisdom nodes are synthesis memories linked to seed memories'
        # entities. If a seed shares entities with a wisdom node, that
        # principle is likely relevant.
        if hasattr(db, 'get_linked_wisdom_nodes') and seed_ids:
            try:
                wisdom_rows = db.get_linked_wisdom_nodes(seed_ids, limit=3)
                if debug:
                    print(f"Wisdom nodes: {len(wisdom_rows)} found")
                best_seed = max(seed_scores.values()) if seed_scores else 0.1
                for wrow in wisdom_rows:
                    wid = wrow["id"]
                    link_score = float(wrow.get("link_score", 0.5))
                    boosted_score = max(best_seed * 0.5, best_seed * link_score)
                    integrity = _verify_content_integrity(wid, wrow.get("content", ""), wrow.get("content_hash"))
                    results.append(RecallResult(
                        memory_id=wid,
                        content=(wrow.get("content") or "")[:1500],
                        summary=wrow.get("summary"),
                        score=boosted_score,
                        source='wisdom',
                        tier=wrow.get("tier", "active"),
                        created_at=_parse_dt(wrow.get("created_at")),
                        integrity_ok=integrity,
                    ))
                    seed_ids.append(wid)
                    seed_scores[wid] = boosted_score
            except Exception as e:
                if debug:
                    print(f"Wisdom retrieval error (non-fatal): {e}")

        # ── Step 3: Spreading activation ─────────────────────────────────
        if include_activated and seed_ids:
            _temporal_boost = 0.0
            if query_intent in ("temporal", "exploratory"):
                _temporal_boost = 0.6

            activated = spreading_activation(
                seed_ids=seed_ids,
                db=db,
                seed_scores=seed_scores,
                depth=spreading_depth,
                decay=spreading_decay,
                min_activation=0.005,
                min_pathway_strength=min_pathway_strength,
                intent=query_intent,
                temporal_boost=_temporal_boost,
            )

            # Cap activated scores at the max DIRECT seed score.
            # Rules get an artificial boost that would otherwise propagate.
            _max_direct_score = max(
                (s for mid, s in seed_scores.items()
                 if any(r.memory_id == mid and r.source == 'direct' for r in results)),
                default=0.5
            )

            activated_ids = [
                (mid, min(score, _max_direct_score))
                for mid, score in activated.items()
                if mid not in seed_ids and score > 0.005
            ]

            if debug:
                print(f"Spreading activation: {len(activated_ids)} additional memories")

            for mem_id, act_score in activated_ids:
                row = db.get_memory(mem_id)
                if row is None:
                    continue
                if exclude_archived and row.get("tier") == "archived":
                    continue
                if exclude_noise and row.get("content_type") in ("session_snapshot", "self_observation"):
                    continue
                integrity = _verify_content_integrity(mem_id, row.get("content", ""), row.get("content_hash"))
                created_at = _parse_dt(row.get("created_at"))

                results.append(RecallResult(
                    memory_id=mem_id,
                    content=(row.get("content") or "")[:1500],
                    summary=row.get("summary"),
                    score=act_score,
                    source="activated",
                    tier=row.get("tier", "active"),
                    created_at=created_at,
                    pathway_strength=act_score,
                    integrity_ok=integrity,
                    estimated_tokens=row.get("estimated_tokens", 0),
                ))

        # ── Step 3.5: Semantic lateral inhibition ─────────────────────────
        # Suppress near-duplicate memories (Jaccard word overlap > 65%).
        # Biologically: lateral inhibitory interneurons prevent clusters
        # of highly similar representations from dominating recall.
        if results:
            _results_sorted = sorted(results, key=lambda r: r.score, reverse=True)
            _selected_wordsets = []
            _inhibited = 0
            for r in _results_sorted:
                _words = set(re.findall(r'\b[a-z]{3,}\b', (r.content or "").lower()))
                if not _words:
                    _selected_wordsets.append(_words)
                    continue
                max_overlap = 0.0
                for _sel in _selected_wordsets:
                    if _sel:
                        _inter = len(_words & _sel)
                        _union = len(_words | _sel)
                        if _union > 0:
                            max_overlap = max(max_overlap, _inter / _union)
                if max_overlap > 0.65:
                    r.score = round(r.score * (1.0 - (max_overlap - 0.65) * 2.0), 4)
                    _inhibited += 1
                else:
                    _selected_wordsets.append(_words)

            if debug and _inhibited:
                print(f"Lateral inhibition: dampened {_inhibited} near-duplicates")
            results.sort(key=lambda r: r.score, reverse=True)

        # ── Step 3.9: Score normalisation ──────────────────────────────────
        best_direct = max(
            (r.score for r in results if r.source in ("direct", "rule")), default=0
        )
        if best_direct > 0:
            for r in results:
                if r.source in ("activated", "graph") and r.score > best_direct:
                    r.score = best_direct * 0.95

        # SM dampening on activated/graph results (seeds already dampened)
        if _sm_damp < 1.0:
            _sm_dampened_ids = {c['row']['id'] for c in seeds if c['is_sm']}
            for r in results:
                if (r.content and r.content.startswith('[Session Mining]')
                        and r.memory_id not in _sm_dampened_ids):
                    r.score *= _sm_damp

        # ── Step 4: Temporal decay + recency boost ─────────────────────────
        now = datetime.now(timezone.utc)
        _recency_w = _intent_params.recency_weight if _intent_params else 0.5
        decay_rate = 0.005 * (0.5 + _recency_w)

        for r in results:
            if r.created_at:
                created = r.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age = now - created
                days_old = age.days
                hours_old = age.total_seconds() / 3600

                # Multiplicative decay (gradual long-term fade)
                decay_factor = (1 - decay_rate) ** days_old
                r.score = r.score * decay_factor

                # Additive recency boost — recent memories get a meaningful
                # advantage on ALL queries.
                # <24h: up to +0.3 (strong boost, 12h half-life)
                # 24-72h: up to +0.15 (moderate boost, 24h half-life)
                # >72h: no boost (rely on existing decay + Hebbian strength)
                if hours_old < 24:
                    r.score += 0.3 * math.exp(-hours_old / 12)
                elif hours_old < 72:
                    r.score += 0.15 * math.exp(-(hours_old - 24) / 24)

        # ── Step 4.4: ACT-R activation boost (Anderson & Schooler 1991) ───
        # Power-law memory scoring: B_i = ln(sum(t_j^(-d)))
        # Memories accessed multiple times recently get higher activation.
        if hasattr(db, 'get_access_log'):
            try:
                _mem_ids = [r.memory_id for r in results if r.memory_id]
                _access_log = db.get_access_log(_mem_ids)
                for r in results:
                    accesses = _access_log.get(r.memory_id, [])
                    if accesses:
                        total = sum(t ** (-0.5) for t in accesses[:50])
                        actr_val = math.log(total) if total > 0 else 0.0
                        actr_boost = max(0, min(0.15, (actr_val + 2) / 33.3))
                        r.score += actr_boost
            except Exception:
                pass  # ACT-R is additive — failure degrades gracefully

        # ── Step 4.5: Near-duplicate collapse ──────────────────────────────
        seen_hashes = set()
        seen_summaries = set()
        deduped = []
        for r in results:
            content_key = hashlib.md5(r.content.encode()).hexdigest()[:12] if r.content else None
            if content_key and content_key in seen_hashes:
                continue
            summary_key = (r.summary or "")[:60].lower().strip()
            if summary_key and len(summary_key) > 20 and summary_key in seen_summaries:
                continue
            if content_key:
                seen_hashes.add(content_key)
            if summary_key and len(summary_key) > 20:
                seen_summaries.add(summary_key)
            deduped.append(r)

        if debug and len(deduped) < len(results):
            print(f"Dedup: {len(results)} -> {len(deduped)}")
        results = deduped

        # ── Step 4.6: State-dependent recall (Godden & Baddeley 1975) ─────
        # If a cognitive state tracker is available, boost memories that
        # were previously active during the same state. Up to 20% bonus.
        # Integration point: implement get_state_affinity() in your backend.
        # Stub: no-op in the open-source version.

        # ── Step 4.7: MMR diversity reranking (Carbonell & Goldstein 1998) ─
        # Maximal Marginal Relevance penalizes redundancy among top results.
        # MMR(d) = lambda * relevance(d) - (1-lambda) * max_sim(d, selected)
        _mmr_lambda = DEFAULTS["mmr_lambda"]
        _mmr_top_k = DEFAULTS["mmr_top_k"]

        if len(results) > 3 and embedding is not None:
            try:
                import numpy as np

                _mmr_candidates = results[:min(len(results), _mmr_top_k * 2)]
                _mmr_ids = [r.memory_id for r in _mmr_candidates]

                # Get embeddings for candidates
                _emb_map = {}
                for r in _mmr_candidates:
                    mem = db.get_memory(r.memory_id)
                    if mem and mem.get("embedding"):
                        emb = mem["embedding"]
                        if isinstance(emb, str):
                            import json
                            emb = json.loads(emb)
                        _emb_map[r.memory_id] = np.array(emb, dtype=np.float32)

                if len(_emb_map) >= 3:
                    def _cosine(a, b):
                        a, b = np.asarray(a), np.asarray(b)
                        dot = np.dot(a, b)
                        na, nb = np.linalg.norm(a), np.linalg.norm(b)
                        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

                    _max_score = max(r.score for r in _mmr_candidates) or 1.0
                    selected = []
                    remaining = list(_mmr_candidates)

                    for _ in range(min(_mmr_top_k, len(remaining))):
                        best_mmr = -1
                        best_idx = 0
                        for i, r in enumerate(remaining):
                            if r.memory_id not in _emb_map:
                                continue
                            relevance = r.score / _max_score
                            max_sim = 0.0
                            for s in selected:
                                if s.memory_id in _emb_map:
                                    sim = _cosine(_emb_map[r.memory_id], _emb_map[s.memory_id])
                                    max_sim = max(max_sim, sim)
                            mmr = _mmr_lambda * relevance - (1 - _mmr_lambda) * max_sim
                            if mmr > best_mmr:
                                best_mmr = mmr
                                best_idx = i
                        if best_idx < len(remaining):
                            selected.append(remaining.pop(best_idx))

                    results = selected + [r for r in results if r not in selected]
                    if debug:
                        print(f"MMR rerank: top {len(selected)} diversified (lambda={_mmr_lambda})")
            except Exception as e:
                if debug:
                    print(f"MMR rerank failed (non-fatal): {e}")

        # ── Step 4.8: Foresight filter ─────────────────────────────────────
        # Boost active foresight memories, demote expired ones.
        # Integration point: implement foresight_valid_from/until in your DB.
        if hasattr(db, 'get_foresight_windows'):
            try:
                foresight_ids = [r.memory_id for r in results]
                foresight_data = db.get_foresight_windows(foresight_ids)
                if foresight_data:
                    id_to_result = {r.memory_id: r for r in results}
                    for mid, valid_from, valid_until in foresight_data:
                        r = id_to_result.get(mid)
                        if not r:
                            continue
                        if valid_until.tzinfo is None:
                            valid_until = valid_until.replace(tzinfo=timezone.utc)
                        if valid_from and valid_from.tzinfo is None:
                            valid_from = valid_from.replace(tzinfo=timezone.utc)
                        if valid_until < now:
                            r.score *= 0.3  # Expired foresight
                        elif valid_from and valid_from <= now <= valid_until:
                            r.score *= 1.5  # Active foresight boost
            except Exception as e:
                if debug:
                    print(f"Foresight filter failed: {e}")

        # Sort by score, limit results
        results.sort(key=lambda r: r.score, reverse=True)

        # ── Step 5: Token budget enforcement ───────────────────────────────
        if max_tokens is not None:
            budgeted = []
            token_sum = 0
            for r in results:
                tok = max(1, int(len(r.content) / 3.5))
                if token_sum + tok > max_tokens and budgeted:
                    break
                budgeted.append(r)
                token_sum += tok
            if debug:
                print(f"Token budget: {token_sum}/{max_tokens} tokens, "
                      f"{len(budgeted)}/{len(results)} results")
            results = budgeted

        # ── Step 6: Hebbian reinforcement ──────────────────────────────────
        # Memories retrieved together wire together (Hebb 1949).
        final = results[:limit]
        if final and len(final) >= 2 and not readonly:
            result_ids = [r.memory_id for r in final[:5]]
            try:
                strengthen_batch(result_ids, db, boost=DEFAULTS["learning_rate"], max_pairs=5)
            except Exception:
                pass

        # ── Step 6.5: Reconsolidation lability (Nader 2000) ────────────────
        # Edges traversed during recall become labile — plasticity window
        # opens. Labile edges get 1.5x boost in rapid consolidation channel.
        # If lability expires without re-consolidation, apply 0.95 decay.
        # This is a novel contribution: no published AI system does this.
        if final and not readonly and hasattr(db, 'mark_labile'):
            try:
                labile_ids = [r.memory_id for r in final[:5]]
                if len(labile_ids) >= 2:
                    db.mark_labile(
                        labile_ids,
                        hours=DEFAULTS["labile_hours"],
                        cap=DEFAULTS["labile_cap_per_epoch"],
                    )
            except Exception:
                pass  # Non-blocking — reconsolidation is enhancement

        # ── Step 7: Access tracking ────────────────────────────────────────
        if final:
            try:
                db.update_access([r.memory_id for r in final])
            except Exception:
                pass

        return final

    except Exception as e:
        if debug:
            print(f"Recall error: {e}")
        return results


def format_results(results: List[RecallResult]) -> str:
    """Format results for display."""
    markers = {
        "direct": "->", "activated": "~>", "wisdom": "=>",
        "graph": "<>", "rule": "!>", "entity_anchor": ">>",
    }
    lines = []
    for r in results:
        marker = markers.get(r.source, "->")
        text = r.summary[:80] if r.summary else r.content[:80]
        lines.append(f"{r.score:.3f} {marker} [{r.tier}] {text}...")
    return "\n".join(lines)


def _parse_dt(ts) -> Optional[datetime]:
    """Parse timestamp from string or return as-is if already datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return datetime.fromisoformat(str(ts).rstrip("Z"))
    except Exception:
        return None
