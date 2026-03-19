"""
Hebbian Recall — Graph-Enhanced Memory Retrieval

Combines vector similarity search with spreading activation through the
co-activation pathway graph to find both SIMILAR memories (vector search)
and RELATED memories (graph traversal).

The recall pipeline:
    1. Classify query intent (query_understanding)
    2. Vector search for seed memories (embeddings)
    3. Spreading activation through pathway graph (spreading_activation)
    4. Score normalisation, decay, deduplication
    5. Hebbian reinforcement of co-retrieved pathways (strengthen_batch)

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
import re
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RecallResult:
    """A memory recall result with provenance."""
    memory_id: str
    content: str
    summary: Optional[str]
    score: float
    source: str          # 'direct', 'activated', 'rule', 'wisdom'
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
    """Verify content has not been tampered with (MD5 hash check)."""
    if not stored_hash:
        return True
    computed = hashlib.md5(content.encode()).hexdigest()
    if computed != stored_hash:
        logger.warning(f"Integrity mismatch for memory {memory_id}")
        return False
    return True


def recall(
    query: str,
    db: any,
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

    Returns:
        List of RecallResult, sorted by score descending.
    """
    from moss.hebbian.spreading_activation import spreading_activation
    from moss.hebbian.pathway_strengthening import strengthen_batch

    results: List[RecallResult] = []

    # ── Step 0: Intent classification ─────────────────────────────────────
    query_intent = None
    _rule_boost = 1.5
    try:
        from moss.hebbian.query_understanding import QueryAnalyzer
        _analyzer = QueryAnalyzer(use_llm=False)
        _intent_params = _analyzer.get_intent_params(query)
        query_intent = _intent_params.intent.value
        if _intent_params.confidence >= 0.2:
            spreading_depth = _intent_params.spreading_depth_mod
            spreading_decay = _intent_params.spreading_decay_mod
            _rule_boost = _intent_params.rule_boost
        if debug:
            print(f"Intent: {query_intent} (conf={_intent_params.confidence})")
    except Exception as e:
        _intent_params = None
        if debug:
            print(f"Intent classification failed (non-fatal): {e}")

    try:
        # ── Step 1: Embed query ────────────────────────────────────────────
        embedding = embed_fn(query)

        # ── Step 1.5: Rule retrieval (extracted_rule content_type) ─────────
        seed_ids = []
        seed_scores = {}
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

        # ── Step 2: Vector search for seeds ───────────────────────────────
        vector_candidates = db.search_by_vector(
            embedding,
            limit=25,
            exclude_ids=list(rule_ids),
            tier_filter=exclude_archived,
            noise_filter=exclude_noise,
        )

        # Lateral inhibition: cap per content_type (diversity)
        ct_counts: Dict[str, int] = {}
        ct_cap = 2
        if _intent_params and _intent_params.diversity_mod != 1.0:
            ct_cap = max(1, int(ct_cap * _intent_params.diversity_mod))

        seeds_selected = []
        for row in sorted(vector_candidates, key=lambda r: r.get("similarity", 0), reverse=True):
            if len(seeds_selected) >= 5:
                break
            ct = row.get("content_type", "general")
            if ct_counts.get(ct, 0) >= ct_cap:
                continue
            ct_counts[ct] = ct_counts.get(ct, 0) + 1
            seeds_selected.append(row)

        for row in seeds_selected:
            mem_id = row["id"]
            similarity = row.get("similarity", 0.0)
            is_stale = bool(row.get("is_stale", False))
            eff_score = similarity * 0.8 if is_stale else similarity
            integrity = _verify_content_integrity(mem_id, row.get("content", ""), row.get("content_hash"))
            created_at = _parse_dt(row.get("created_at"))

            results.append(RecallResult(
                memory_id=mem_id,
                content=(row.get("content") or "")[:1500],
                summary=row.get("summary"),
                score=eff_score,
                source="direct",
                tier=row.get("tier", "active"),
                created_at=created_at,
                integrity_ok=integrity,
                is_stale=is_stale,
                estimated_tokens=row.get("estimated_tokens", 0),
            ))
            seed_ids.append(mem_id)
            seed_scores[mem_id] = eff_score

        if debug:
            print(f"Seeds: {len(seeds_selected)} selected")

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

            activated_ids = [
                (mid, score)
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
        # Suppress near-duplicate memories (Jaccard overlap > 65%)
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

        # ── Step 4: Score normalisation ───────────────────────────────────
        best_direct = max(
            (r.score for r in results if r.source in ("direct", "rule")), default=0
        )
        if best_direct > 0:
            for r in results:
                if r.source in ("activated",) and r.score > best_direct:
                    r.score = best_direct * 0.95

        # ── Step 4.5: Temporal decay ──────────────────────────────────────
        now = datetime.now(timezone.utc)
        _recency_w = _intent_params.recency_weight if _intent_params else 0.5
        decay_rate = 0.005 * (0.5 + _recency_w)

        for r in results:
            if r.created_at:
                created = r.created_at
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                days_old = max(0, (now - created).days)
                decay_factor = (1 - decay_rate) ** days_old
                r.score = r.score * decay_factor

        # ── Step 4.6: Near-duplicate collapse ─────────────────────────────
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

        results = deduped
        results.sort(key=lambda r: r.score, reverse=True)

        # ── Step 5: Token budget ──────────────────────────────────────────
        if max_tokens is not None:
            budgeted = []
            token_sum = 0
            for r in results:
                tok = max(1, int(len(r.content) / 3.5))
                if token_sum + tok > max_tokens and budgeted:
                    break
                budgeted.append(r)
                token_sum += tok
            results = budgeted

        # ── Step 6: Hebbian reinforcement ─────────────────────────────────
        final = results[:limit]
        if final and len(final) >= 2 and not readonly:
            result_ids = [r.memory_id for r in final[:5]]
            try:
                strengthen_batch(result_ids, db, boost=0.1, max_pairs=5)
            except Exception:
                pass

        # ── Step 7: Access tracking ───────────────────────────────────────
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
    markers = {"direct": "->", "activated": "~>", "wisdom": "=>", "rule": "!>"}
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
