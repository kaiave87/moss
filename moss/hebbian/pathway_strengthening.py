"""
Hebbian Pathway Strengthening and Weakening

Implements synchronous pathway operations extracted from the core
Hebbian learning algorithm.

Core principle: Neurons that fire together wire together (Hebb 1949).
When memories are co-retrieved, the pathways between them are
strengthened. When retrieval is unhelpful, pathways are weakened
(anti-Hebbian correction).

These functions operate on the MockDB interface defined in moss.hebbian.db.
Swap `db` for your production backend to wire into a real database.

Patent pending — Lichen Research Inc., Canadian application filed March 2026.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def strengthen_pathway_sync(
    source_id: str,
    target_id: str,
    db: any,
    boost: float = 0.1,
) -> bool:
    """
    Synchronously strengthen a co-activation pathway between two memories.

    Called after successful co-retrieval to reinforce the association.
    Implements the Hebbian learning rule: repeated co-activation increases
    pathway strength, capped at 1.0 to prevent runaway potentiation.

    The pathway is created if it does not exist (initial strength = boost).
    Pruned pathways (soft-deleted) are resurrected on strengthen.

    Args:
        source_id: Memory ID of the source node.
        target_id: Memory ID of the target node.
        db: Database backend with strengthen_pathway(src, tgt, boost) method.
        boost: Strength increase (default 0.1).

    Returns:
        True if the pathway was created or strengthened, False otherwise.
    """
    if source_id == target_id:
        return False

    try:
        return db.strengthen_pathway(source_id, target_id, boost=boost)
    except Exception as e:
        logger.debug(f"strengthen_pathway_sync failed: {e}")
        return False


def weaken_pathway_sync(
    source_id: str,
    target_id: str,
    db: any,
    penalty: float = 0.05,
    labile_only: bool = False,
) -> bool:
    """
    Synchronous anti-Hebbian pathway weakening.

    Mirror of strengthen_pathway_sync() with negative delta.
    Used for outcome-weighted pathway correction — when a retrieved
    memory was not helpful, reduce the strength of the pathways that
    led to its retrieval.

    The default penalty (0.05) is deliberately conservative: 6x smaller
    than the default strengthen boost (0.1). This asymmetry prevents
    spurious negative signals from erasing well-established associations.

    Args:
        source_id: Memory ID of the source node.
        target_id: Memory ID of the target node.
        db: Database backend with weaken_pathway(src, tgt, penalty) method.
        penalty: Strength decrease (default 0.05).
        labile_only: If True, only weaken pathways in reconsolidation window.

    Returns:
        True if the pathway was found and weakened, False otherwise.
    """
    if source_id == target_id:
        return False

    try:
        return db.weaken_pathway(source_id, target_id, penalty=penalty)
    except Exception as e:
        logger.debug(f"weaken_pathway_sync failed: {e}")
        return False


def strengthen_batch(
    memory_ids: list,
    db: any,
    boost: float = 0.1,
    max_pairs: int = 10,
) -> int:
    """
    Strengthen all pairwise pathways among a set of co-retrieved memories.

    Called after a successful recall to reinforce the full co-activation
    graph. For N memories, creates/strengthens up to N*(N-1)/2 pathways,
    capped at max_pairs for efficiency.

    Args:
        memory_ids: List of memory IDs that were co-retrieved.
        db: Database backend.
        boost: Strength increase per pair (default 0.1).
        max_pairs: Maximum number of pathway operations (default 10).

    Returns:
        Number of pathways strengthened.
    """
    strengthened = 0
    for i, mem_a in enumerate(memory_ids):
        for mem_b in memory_ids[i + 1:]:
            if strengthen_pathway_sync(mem_a, mem_b, db, boost=boost):
                strengthened += 1
            if strengthened >= max_pairs:
                return strengthened
    return strengthened
