"""
Hebbian Recall — Graph-enhanced memory retrieval with spreading activation.

Combines dense vector similarity search with Hebbian spreading activation
across a memory pathway graph. Memories that co-activate frequently develop
stronger associative links, which are then exploited at retrieval time to
surface related memories beyond the direct similarity horizon.

Core algorithm:
    1. Embed the query via any embedding function (plug-in)
    2. Retrieve top-k candidates by cosine similarity (direct recall)
    3. For each candidate, traverse the pathway graph up to `spreading_depth`
       hops, decaying signal strength by `spreading_decay` per hop
    4. Activated memories accumulate a hebbian_boost proportional to their
       pathway strength and traversal depth
    5. Final ranking = vector_score + hebbian_boost

Hebbian update rule (call `strengthen_pathway` after retrievals):
    strength += learning_rate * (1 - strength)   # approach 1.0 asymptotically
    strength *= (1 - decay_rate)                 # periodic decay toward 0

This produces a pathway graph where frequently co-accessed memories develop
strong links, biasing future retrievals toward contextually coherent clusters.

Usage:
    from moss.hebbian.hebbian_recall import HebbianMemoryStore, RecallResult

    store = HebbianMemoryStore(embed_func=my_embed_fn)

    # Add memories
    store.add("m1", "Alice went hiking last Saturday")
    store.add("m2", "Alice loves the outdoors")
    store.add("m3", "Bob stayed home last weekend")

    # Manually strengthen a co-activation pathway
    store.strengthen_pathway("m1", "m2")

    # Recall — activates m1 by similarity, m2 via Hebbian pathway
    results = store.recall("What did Alice do on the weekend?", limit=5)
    for r in results:
        print(r.memory_id, r.score, r.source, r.hebbian_boost)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single memory with its embedding and metadata."""
    memory_id: str
    content: str
    embedding: List[float]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class RecallResult:
    """A retrieved memory with provenance and scoring breakdown."""
    memory_id: str
    content: str
    score: float                        # Final blended score
    vector_score: float                 # Raw cosine similarity
    hebbian_boost: float                # Added via spreading activation
    source: str                         # 'direct' | 'activated' | 'both'
    pathway_strength: Optional[float]   # Strength of activating pathway (if source='activated')
    created_at: Optional[datetime] = None

    # Dict-style access for drop-in compatibility
    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass
class MemoryPathway:
    """Associative link between two memories."""
    source_id: str
    target_id: str
    strength: float = 0.01              # [0, 1] — grows via Hebbian updates
    activation_count: int = 0
    last_activated: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Core store
# ---------------------------------------------------------------------------

class HebbianMemoryStore:
    """
    In-memory implementation of Hebbian recall.

    Plug-in architecture: provide any `embed_func` that maps str -> List[float].
    For production use, replace the in-memory index with a vector database
    (pgvector, Faiss, Qdrant, etc.) while keeping the spreading activation logic.
    """

    def __init__(
        self,
        embed_func: Optional[Callable[[str], List[float]]] = None,
        learning_rate: float = 0.1,
        decay_rate: float = 0.01,
        spreading_depth: int = 2,
        spreading_decay: float = 0.5,
        min_pathway_strength: float = 0.05,
    ):
        """
        Args:
            embed_func: Function mapping text -> embedding vector.
                        If None, a simple bag-of-words fallback is used.
            learning_rate: Hebbian update rate (0 < lr < 1). Higher = faster
                           strengthening of co-activated pairs.
            decay_rate: Pathway decay rate per maintenance cycle (0 < dr < 1).
                        Keeps the graph sparse by decaying unused pathways.
            spreading_depth: Max hops for spreading activation traversal.
            spreading_decay: Signal decay per hop (0 < sd < 1). At depth=2
                             with decay=0.5: 1.0 → 0.5 → 0.25.
            min_pathway_strength: Pathways below this threshold are ignored
                                  during retrieval (pruned over time by decay).
        """
        self.embed_func = embed_func or _bow_embed
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.spreading_depth = spreading_depth
        self.spreading_decay = spreading_decay
        self.min_pathway_strength = min_pathway_strength

        # Memory index
        self._memories: Dict[str, MemoryEntry] = {}

        # Pathway graph: (source_id, target_id) -> MemoryPathway
        # Stored bidirectionally for undirected spreading activation
        self._pathways: Dict[Tuple[str, str], MemoryPathway] = {}

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def add(self, memory_id: str, content: str) -> MemoryEntry:
        """Add a memory. Computes and stores its embedding."""
        embedding = self.embed_func(content)
        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            embedding=embedding,
        )
        self._memories[memory_id] = entry
        logger.debug("Added memory %s (%d-dim embedding)", memory_id, len(embedding))
        return entry

    def remove(self, memory_id: str) -> bool:
        """Remove a memory and all its pathways."""
        if memory_id not in self._memories:
            return False
        del self._memories[memory_id]
        # Remove all pathways involving this memory
        stale = [k for k in self._pathways if memory_id in k]
        for k in stale:
            del self._pathways[k]
        return True

    # ------------------------------------------------------------------
    # Pathway management (Hebbian updates)
    # ------------------------------------------------------------------

    def strengthen_pathway(self, id_a: str, id_b: str) -> float:
        """
        Apply Hebbian update to the pathway between id_a and id_b.

        Call this after any retrieval event where both memories were returned
        together — "neurons that fire together, wire together."

        Returns the new pathway strength.
        """
        if id_a not in self._memories or id_b not in self._memories:
            return 0.0
        key = _pathway_key(id_a, id_b)
        pw = self._pathways.get(key)
        if pw is None:
            pw = MemoryPathway(source_id=id_a, target_id=id_b)
            self._pathways[key] = pw

        # Hebbian update: asymptotic approach to 1.0
        pw.strength += self.learning_rate * (1.0 - pw.strength)
        pw.strength = min(pw.strength, 1.0)
        pw.activation_count += 1
        pw.last_activated = datetime.now(timezone.utc)
        return pw.strength

    def decay_pathways(self) -> int:
        """
        Apply decay to all pathways. Call periodically (e.g., nightly maintenance).

        Returns the number of pathways pruned (strength dropped below minimum).
        """
        pruned = 0
        stale_keys = []
        for key, pw in self._pathways.items():
            pw.strength *= (1.0 - self.decay_rate)
            if pw.strength < self.min_pathway_strength:
                stale_keys.append(key)
                pruned += 1
        for key in stale_keys:
            del self._pathways[key]
        logger.debug("Pathway decay: pruned %d pathways, %d remain", pruned, len(self._pathways))
        return pruned

    def get_neighbors(self, memory_id: str) -> List[Tuple[str, float]]:
        """Return (neighbor_id, strength) pairs for all pathways from memory_id."""
        neighbors = []
        for (a, b), pw in self._pathways.items():
            if a == memory_id:
                neighbors.append((b, pw.strength))
            elif b == memory_id:
                neighbors.append((a, pw.strength))
        return sorted(neighbors, key=lambda x: -x[1])

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        limit: int = 10,
        min_vector_score: float = 0.0,
        include_activated: bool = True,
        auto_strengthen: bool = False,
        debug: bool = False,
    ) -> List[RecallResult]:
        """
        Retrieve memories using hybrid vector + Hebbian spreading activation.

        Args:
            query: Natural language query.
            limit: Maximum results to return.
            min_vector_score: Discard candidates with cosine similarity below this.
            include_activated: If True, follow Hebbian pathways from direct matches
                               to surface associated memories.
            auto_strengthen: If True, automatically strengthen pathways between
                             all returned memories (trains the graph).
            debug: Log detailed scoring information.

        Returns:
            List of RecallResult sorted by final score (descending).
        """
        if not self._memories:
            return []

        query_embedding = self.embed_func(query)

        # --- Stage 1: Direct vector recall ---
        scored: Dict[str, RecallResult] = {}

        for mem_id, entry in self._memories.items():
            sim = _cosine_similarity(query_embedding, entry.embedding)
            if sim < min_vector_score:
                continue
            scored[mem_id] = RecallResult(
                memory_id=mem_id,
                content=entry.content,
                score=sim,
                vector_score=sim,
                hebbian_boost=0.0,
                source="direct",
                pathway_strength=None,
                created_at=entry.created_at,
            )

        if debug:
            logger.debug("Direct recall: %d candidates above threshold", len(scored))

        # --- Stage 2: Spreading activation ---
        if include_activated and scored:
            self._spread_activation(scored, query_embedding, debug=debug)

        # --- Stage 3: Sort and cap ---
        results = sorted(scored.values(), key=lambda r: r.score, reverse=True)[:limit]

        # --- Stage 4: Optional Hebbian strengthening ---
        if auto_strengthen and len(results) > 1:
            ids = [r.memory_id for r in results]
            for i in range(len(ids)):
                for j in range(i + 1, min(i + 4, len(ids))):  # Strengthen within top-4 window
                    self.strengthen_pathway(ids[i], ids[j])

        # Update access counts
        for r in results:
            if r.memory_id in self._memories:
                self._memories[r.memory_id].access_count += 1
                self._memories[r.memory_id].last_accessed = datetime.now(timezone.utc)

        return results

    def _spread_activation(
        self,
        scored: Dict[str, RecallResult],
        query_embedding: List[float],
        debug: bool = False,
    ) -> None:
        """
        BFS spreading activation from direct recall candidates.

        For each direct match, traverse its pathway graph up to
        `spreading_depth` hops. Signal strength decays by `spreading_decay`
        per hop. Accumulated boost is added to final scores.

        Modifies `scored` in-place.
        """
        # Activation queue: (memory_id, signal_strength, pathway_strength)
        frontier: List[Tuple[str, float, float]] = [
            (mid, result.vector_score, 1.0)
            for mid, result in scored.items()
        ]

        for depth in range(self.spreading_depth):
            next_frontier: List[Tuple[str, float, float]] = []
            signal_decay = self.spreading_decay ** (depth + 1)

            for source_id, signal, _ in frontier:
                neighbors = self.get_neighbors(source_id)

                for neighbor_id, pw_strength in neighbors:
                    if pw_strength < self.min_pathway_strength:
                        continue
                    if neighbor_id not in self._memories:
                        continue

                    boost = signal * pw_strength * signal_decay

                    if neighbor_id in scored:
                        # Already a direct match — add boost to existing result
                        r = scored[neighbor_id]
                        r.hebbian_boost += boost
                        r.score = r.vector_score + r.hebbian_boost
                        r.source = "both"
                    else:
                        # New memory activated via pathway
                        entry = self._memories[neighbor_id]
                        vec_score = _cosine_similarity(query_embedding, entry.embedding)
                        existing = scored.get(neighbor_id)
                        if existing is None:
                            scored[neighbor_id] = RecallResult(
                                memory_id=neighbor_id,
                                content=entry.content,
                                score=vec_score + boost,
                                vector_score=vec_score,
                                hebbian_boost=boost,
                                source="activated",
                                pathway_strength=pw_strength,
                                created_at=entry.created_at,
                            )
                        else:
                            existing.hebbian_boost += boost
                            existing.score = existing.vector_score + existing.hebbian_boost
                            existing.pathway_strength = max(
                                existing.pathway_strength or 0.0, pw_strength
                            )

                    next_frontier.append((neighbor_id, signal * pw_strength, pw_strength))

            frontier = next_frontier
            if not frontier:
                break

        if debug:
            activated = sum(1 for r in scored.values() if r.source == "activated")
            boosted = sum(1 for r in scored.values() if r.source == "both")
            logger.debug(
                "Spreading activation depth=%d: %d newly activated, %d boosted",
                self.spreading_depth, activated, boosted,
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        """Return summary statistics about the memory store."""
        strong = sum(1 for pw in self._pathways.values() if pw.strength >= 0.5)
        moderate = sum(1 for pw in self._pathways.values() if 0.2 <= pw.strength < 0.5)
        return {
            "total_memories": len(self._memories),
            "total_pathways": len(self._pathways),
            "strong_pathways": strong,
            "moderate_pathways": moderate,
            "weak_pathways": len(self._pathways) - strong - moderate,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _pathway_key(id_a: str, id_b: str) -> Tuple[str, str]:
    """Canonical undirected key — always (smaller, larger) lexicographically."""
    return (id_a, id_b) if id_a < id_b else (id_b, id_a)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        # Truncate to shorter length (graceful mismatch handling)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _bow_embed(text: str) -> List[float]:
    """
    Minimal bag-of-words fallback embedding (no external dependencies).

    For testing only. Replace with a real embedding model in production
    (e.g., sentence-transformers, OpenAI embeddings, Ollama qwen3-embedding).
    """
    import re
    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    vocab: Dict[str, int] = {}
    for w in words:
        vocab[w] = vocab.get(w, 0) + 1
    if not vocab:
        return [0.0]
    # Normalize
    total = sum(vocab.values())
    all_words = sorted(vocab)
    return [vocab[w] / total for w in all_words]
