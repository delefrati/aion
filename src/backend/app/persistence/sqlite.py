from __future__ import annotations

import os
import aiosqlite


class SqliteStore:
    """SQLite-backed persistence for Nano mode."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv ON messages(conversation_id)"
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def save_message(self, conversation_id: str, role: str, content: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )
        await self._db.commit()

    async def get_conversation(self, conversation_id: str) -> list[dict]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]

    async def list_conversations(self, limit: int = 50) -> list[dict]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT conversation_id, MAX(created_at) as last_at, COUNT(*) as msg_count
            FROM messages
            GROUP BY conversation_id
            ORDER BY last_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {"conversation_id": row[0], "last_at": row[1], "message_count": row[2]}
            for row in rows
        ]
