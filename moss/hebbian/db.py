"""
Mock Database Layer for Hebbian Memory

Implements the storage interface expected by Hebbian recall and pathway
strengthening using SQLite in-memory as the backing store. This enables
tests to run without any external database dependencies while preserving
the SQL schemas that are central to the patent's algorithmic claims.

The SQL schemas here document the data structures described in:
    Canadian Patent Application — Hebbian Recall System
    Lichen Research Inc., filed March 2026.

For production, replace this layer with a PostgreSQL backend using
the same schema (unified_memory, memory_pathways tables).
"""

import hashlib
import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Schema (documented for patent reference)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- unified_memory: primary memory storage
-- Patent claim: content addressed by semantic embedding + pathway graph
CREATE TABLE IF NOT EXISTS unified_memory (
    id          TEXT PRIMARY KEY,          -- UUID string
    content     TEXT NOT NULL,
    summary     TEXT,
    tier        TEXT DEFAULT 'active',     -- active, archived, hardwired
    memory_type TEXT DEFAULT 'episodic',   -- episodic, semantic, procedural
    content_type TEXT DEFAULT 'general',
    strength    REAL DEFAULT 0.5,          -- Hebbian strength [0, 1]
    access_count INTEGER DEFAULT 0,
    last_accessed TEXT,                    -- ISO timestamp
    created_at  TEXT,
    content_hash TEXT,                     -- MD5 for integrity verification
    embedding   TEXT,                      -- JSON-encoded float list
    estimated_tokens INTEGER DEFAULT 0,
    is_stale    INTEGER DEFAULT 0          -- boolean
);

-- memory_pathways: Hebbian co-activation graph
-- Patent claim: pathway strength encodes learned associations
--               via repeated co-activation (Hebb 1949)
CREATE TABLE IF NOT EXISTS memory_pathways (
    id              TEXT PRIMARY KEY,
    source_memory   TEXT NOT NULL REFERENCES unified_memory(id),
    target_memory   TEXT NOT NULL REFERENCES unified_memory(id),
    strength        REAL DEFAULT 0.1,      -- Co-activation strength [0, 1]
    coactivation_count INTEGER DEFAULT 1,
    last_coactivation TEXT,                -- ISO timestamp
    created_at      TEXT,
    pruned_at       TEXT,                  -- Soft-delete timestamp
    labile          INTEGER DEFAULT 0,     -- Reconsolidation lability flag
    labile_until    TEXT,                  -- Lability window expiry
    pathway_type    TEXT DEFAULT 'coactivation',
    UNIQUE(source_memory, target_memory)
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


class MockDB:
    """
    In-memory SQLite-backed Hebbian memory database.

    Implements the interface required by moss.hebbian.recall and
    moss.hebbian.pathway_strengthening.

    Usage:
        db = MockDB()
        mem_id = db.store_memory("Paris is the capital of France.")
        results = db.search_by_vector(query_vec, limit=5)
        db.strengthen_pathway(id_a, id_b, boost=0.1)
    """

    def __init__(self):
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ── Memory Storage ──────────────────────────────────────────────────────

    def store_memory(
        self,
        content: str,
        summary: Optional[str] = None,
        tier: str = "active",
        memory_type: str = "episodic",
        content_type: str = "general",
        strength: float = 0.5,
        embedding: Optional[List[float]] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """Store a memory and return its ID."""
        mem_id = memory_id or str(uuid.uuid4())
        if embedding is None:
            from moss.hebbian.embeddings import get_embedding
            embedding = get_embedding(content)
        emb_json = json.dumps(embedding) if embedding is not None else None
        est_tokens = max(1, len(content) // 4)
        chash = _content_hash(content)

        self._conn.execute(
            """
            INSERT INTO unified_memory
              (id, content, summary, tier, memory_type, content_type,
               strength, embedding, content_hash, estimated_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (mem_id, content, summary, tier, memory_type, content_type,
             strength, emb_json, chash, est_tokens, _now_iso()),
        )
        self._conn.commit()
        return mem_id

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single memory by ID."""
        row = self._conn.execute(
            "SELECT * FROM unified_memory WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("embedding"):
            d["embedding"] = json.loads(d["embedding"])
        return d

    def update_access(self, memory_ids: List[str]) -> None:
        """Increment access count and last_accessed for the given IDs."""
        now = _now_iso()
        for mid in memory_ids:
            self._conn.execute(
                """
                UPDATE unified_memory
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
                """,
                (now, mid),
            )
        self._conn.commit()

    # ── Search ───────────────────────────────────────────────────────────────

    def search_by_vector(
        self,
        query_vec: List[float],
        limit: int = 10,
        exclude_ids: Optional[List[str]] = None,
        tier_filter: bool = True,
        noise_filter: bool = True,
        content_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search using cosine similarity.

        Returns memories sorted by similarity descending. Only returns
        memories that have stored embeddings.
        """
        rows = self._conn.execute(
            "SELECT * FROM unified_memory WHERE embedding IS NOT NULL"
        ).fetchall()

        exclude = set(exclude_ids or [])
        scored = []
        for row in rows:
            d = dict(row)
            if d["id"] in exclude:
                continue
            if tier_filter and d.get("tier") == "archived":
                continue
            if noise_filter and d.get("content_type") in ("session_snapshot", "self_observation"):
                continue
            if content_type and d.get("content_type") != content_type:
                continue
            try:
                emb = json.loads(d["embedding"])
                sim = _cosine_similarity(query_vec, emb)
            except Exception:
                continue
            d["similarity"] = sim
            if d.get("embedding"):
                d["embedding"] = json.loads(d["embedding"]) if isinstance(d["embedding"], str) else d["embedding"]
            scored.append(d)

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:limit]

    def search_by_content_type(
        self,
        content_type: str,
        query_vec: List[float],
        similarity_threshold: float = 0.5,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Search for memories of a specific content_type by vector similarity."""
        return self.search_by_vector(
            query_vec,
            limit=limit,
            content_type=content_type,
        )

    # ── Pathway Management ───────────────────────────────────────────────────

    def strengthen_pathway(
        self,
        source_id: str,
        target_id: str,
        boost: float = 0.1,
    ) -> bool:
        """
        Create or strengthen a co-activation pathway.

        Implements the core Hebbian learning rule:
            "Neurons that fire together wire together."

        Returns True if pathway was created or updated.
        """
        if source_id == target_id:
            return False

        # Check if pathway exists in either direction
        row = self._conn.execute(
            """
            SELECT id, strength FROM memory_pathways
            WHERE (source_memory = ? AND target_memory = ?)
               OR (source_memory = ? AND target_memory = ?)
            """,
            (source_id, target_id, target_id, source_id),
        ).fetchone()

        now = _now_iso()

        if row:
            new_strength = min(1.0, row["strength"] + boost)
            self._conn.execute(
                """
                UPDATE memory_pathways
                SET strength = ?,
                    coactivation_count = coactivation_count + 1,
                    last_coactivation = ?,
                    pruned_at = NULL
                WHERE id = ?
                """,
                (new_strength, now, row["id"]),
            )
        else:
            pw_id = str(uuid.uuid4())
            self._conn.execute(
                """
                INSERT INTO memory_pathways
                  (id, source_memory, target_memory, strength, coactivation_count,
                   last_coactivation, created_at)
                VALUES (?, ?, ?, ?, 1, ?, ?)
                """,
                (pw_id, source_id, target_id, min(1.0, boost), now, now),
            )

        self._conn.commit()
        return True

    def weaken_pathway(
        self,
        source_id: str,
        target_id: str,
        penalty: float = 0.05,
    ) -> bool:
        """Apply anti-Hebbian weakening to a pathway."""
        row = self._conn.execute(
            """
            SELECT id, strength FROM memory_pathways
            WHERE (source_memory = ? AND target_memory = ?)
               OR (source_memory = ? AND target_memory = ?)
            AND pruned_at IS NULL
            """,
            (source_id, target_id, target_id, source_id),
        ).fetchone()

        if not row:
            return False

        new_strength = max(0.01, row["strength"] - penalty)
        self._conn.execute(
            "UPDATE memory_pathways SET strength = ? WHERE id = ?",
            (new_strength, row["id"]),
        )
        self._conn.commit()
        return True

    def get_pathways(
        self,
        memory_ids: List[str],
        min_strength: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Return all active pathways connected to any of the given memory IDs."""
        if not memory_ids:
            return []

        placeholders = ",".join("?" * len(memory_ids))
        rows = self._conn.execute(
            f"""
            SELECT * FROM memory_pathways
            WHERE (source_memory IN ({placeholders})
                   OR target_memory IN ({placeholders}))
              AND pruned_at IS NULL
              AND strength >= ?
            """,
            (*memory_ids, *memory_ids, min_strength),
        ).fetchall()

        return [dict(r) for r in rows]

    def count_pathways(self) -> int:
        """Return total active (not pruned) pathway count."""
        row = self._conn.execute(
            "SELECT COUNT(*) as n FROM memory_pathways WHERE pruned_at IS NULL"
        ).fetchone()
        return row["n"]

    # ── Helpers ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
