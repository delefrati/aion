"""PostgreSQL-backed persistence for Standard mode."""
from __future__ import annotations

import asyncpg


class PostgresStore:
    """PostgreSQL persistence using asyncpg."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def init(self) -> None:
        self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id BIGSERIAL PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conv
                ON messages(conversation_id)
                """
            )

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def save_message(self, conversation_id: str, role: str, content: str) -> None:
        assert self._pool is not None
        await self._pool.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3)",
            conversation_id, role, content,
        )

    async def get_conversation(self, conversation_id: str) -> list[dict]:
        assert self._pool is not None
        rows = await self._pool.fetch(
            "SELECT role, content FROM messages WHERE conversation_id = $1 ORDER BY id",
            conversation_id,
        )
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def list_conversations(self, limit: int = 50) -> list[dict]:
        assert self._pool is not None
        rows = await self._pool.fetch(
            """
            SELECT conversation_id, MAX(created_at) as last_at, COUNT(*) as msg_count
            FROM messages
            GROUP BY conversation_id
            ORDER BY last_at DESC
            LIMIT $1
            """,
            limit,
        )
        return [
            {"conversation_id": row["conversation_id"], "last_at": str(row["last_at"]), "message_count": row["msg_count"]}
            for row in rows
        ]
