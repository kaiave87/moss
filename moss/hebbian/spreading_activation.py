"""
Spreading Activation for Hebbian Memory

Implements neural-inspired spreading activation through the memory pathway
graph. When seed memories are activated, their co-activation energy
propagates to connected memories, weighted by pathway strength and decayed
by hop distance.

Like neural priming: activating one concept partially activates related
concepts through learned associations.

Patent pending — Lichen Research Inc., Canadian application filed March 2026.

References:
    Collins & Loftus (1975) — Spreading activation theory of semantic processing
    Anderson (1983) — ACT-R spreading activation
    Hasselmo (1999) — Neuromodulation and cortical function
"""

import logging
import math
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def spreading_activation(
    seed_ids: List[str],
    db: any,
    seed_scores: Optional[Dict[str, float]] = None,
    depth: int = 2,
    decay: float = 0.5,
    min_activation: float = 0.005,
    min_pathway_strength: float = 0.0,
    intent: Optional[str] = None,
    temporal_boost: float = 0.0,
) -> Dict[str, float]:
    """
    Spreading activation from seed memories through the pathway graph.

    Activation propagates along co-activation pathways, decaying by the
    `decay` factor at each hop. The algorithm is:

        activation(neighbor) += seed_activation * pathway_strength * decay^hop

    Intent-guided traversal modulates the exploration strategy:
        - EXPLORATORY: wider spread (lower decay, lower strength threshold)
        - FACTUAL/NAVIGATIONAL: tighter focus (higher decay, higher threshold)
        - DEBUG: deeper traversal (+1 depth, moderate decay)

    Args:
        seed_ids: Memory IDs that were directly retrieved (query seeds).
        db: Database backend with get_pathways(ids, min_strength) method.
        seed_scores: Optional dict of seed_id -> initial activation score.
                     Defaults to 1.0 for all seeds.
        depth: Number of hops to spread (1 = direct connections only).
        decay: Activation decay factor per hop (0.5 = half per hop).
        min_activation: Prune activations below this threshold.
        min_pathway_strength: Only traverse pathways above this strength.
        intent: Query intent string for modulation (factual, exploratory, etc.)
        temporal_boost: Boost factor for temporally proximate memories (0.0 = off).

    Returns:
        Dict mapping memory_id -> activation_score for all activated memories
        (includes seeds and activated neighbors).
    """
    if not seed_ids:
        return {}

    # ── Intent-guided traversal modulation ──────────────────────────────────
    # Different intents require different graph exploration strategies.
    # Profiles: (decay_multiplier, strength_threshold_multiplier, depth_delta)
    if intent:
        _intent = intent.lower()
        _profiles = {
            "exploratory":  (0.7, 0.6, 1),   # Wide: low decay, low bar, +1 depth
            "debug":        (0.85, 0.8, 1),   # Deep: moderate settings, +1 depth
            "procedural":   (0.9, 0.9, 0),    # Moderate: slightly wider
            "creative":     (0.75, 0.5, 1),   # Widest: low decay, very low bar
            "factual":      (1.2, 1.3, 0),    # Tight: high decay, high bar
            "navigational": (1.3, 1.5, -1),   # Tightest: focused, fewer hops
        }
        if _intent in _profiles:
            _dm, _sm, _da = _profiles[_intent]
            decay = min(0.95, decay * _dm)
            min_pathway_strength = max(0.0, min_pathway_strength * _sm)
            depth = max(1, depth + _da)

    # ── Initialise activation levels ────────────────────────────────────────
    if seed_scores is None:
        seed_scores = {sid: 1.0 for sid in seed_ids}

    activation: Dict[str, float] = dict(seed_scores)
    frontier = list(seed_ids)

    # ── BFS spreading activation ─────────────────────────────────────────────
    try:
        for hop in range(depth):
            if not frontier:
                break

            hop_decay = decay ** (hop + 1)
            pathways = db.get_pathways(frontier, min_strength=min_pathway_strength)

            new_frontier = []
            for pw in pathways:
                source = pw["source_memory"]
                target = pw["target_memory"]
                strength = pw.get("strength", 0.0)
                ptype = pw.get("pathway_type", "coactivation")

                # Determine which endpoint is the frontier seed
                if source in activation:
                    seed_id = source
                    neighbor_id = target
                elif target in activation:
                    seed_id = target
                    neighbor_id = source
                else:
                    continue

                seed_act = activation[seed_id]

                # Causal edge directional boost (forward=1.3x, reverse=0.8x)
                causal_mult = 1.0
                if ptype == "caused":
                    if seed_id == source:
                        causal_mult = 1.3
                    else:
                        causal_mult = 0.8

                # Temporal proximity boost (optional)
                temporal_mult = 1.0
                if temporal_boost > 0.0:
                    src_ts = pw.get("source_created")
                    tgt_ts = pw.get("target_created")
                    if src_ts and tgt_ts:
                        try:
                            delta_days = abs((_parse_date(src_ts) - _parse_date(tgt_ts)).days)
                            proximity = math.exp(-delta_days / 30.0)
                            temporal_mult = 1.0 + temporal_boost * proximity
                        except Exception:
                            pass

                neighbor_act = seed_act * strength * hop_decay * causal_mult * temporal_mult

                if neighbor_act < min_activation:
                    continue

                if neighbor_id in activation:
                    activation[neighbor_id] = min(1.0, activation[neighbor_id] + neighbor_act)
                else:
                    activation[neighbor_id] = neighbor_act
                    new_frontier.append(neighbor_id)

            frontier = new_frontier

    except Exception as e:
        logger.debug(f"spreading_activation failed: {e}")
        return seed_scores

    logger.debug(
        f"Spreading activation: {len(seed_ids)} seeds -> {len(activation)} activated"
    )
    return activation


def _parse_date(ts: str):
    """Parse ISO timestamp string to datetime."""
    from datetime import datetime
    # Handle both naive and timezone-aware
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts[:26], fmt[:len(ts[:26])])
        except ValueError:
            continue
    return datetime.fromisoformat(ts[:19])
