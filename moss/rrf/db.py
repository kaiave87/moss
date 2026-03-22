"""Mock database layer for Moss RRF module.

In production, this connects to PostgreSQL with pgvector.
For open-source usage, provide your own get_connection() implementation.

Schema (unified_memory table):
    CREATE TABLE unified_memory (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        summary TEXT,
        embedding vector(4096),  -- pgvector
        content_type TEXT DEFAULT 'observation',
        importance REAL DEFAULT 0.5,
        tier TEXT DEFAULT 'working',
        conversation_id TEXT,
        session_id TEXT,
        created_at TIMESTAMPTZ DEFAULT now(),
        last_accessed TIMESTAMPTZ,
        access_count INTEGER DEFAULT 0,
        archived BOOLEAN DEFAULT false,
        content_hash TEXT
    );

    CREATE INDEX idx_um_embedding ON unified_memory
        USING hnsw (embedding vector_cosine_ops);
"""


def get_connection():
    """Return a database connection.

    Override this with your actual database connection.
    Example with psycopg2:
        import psycopg2
        return psycopg2.connect(dbname='moss', user='moss', port=5432)
    """
    raise NotImplementedError(
        "Provide a database connection. See db.py for schema requirements."
    )
