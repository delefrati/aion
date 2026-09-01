"""User document storage for RAG."""
from __future__ import annotations

import logging
from uuid import uuid4

import aiosqlite

logger = logging.getLogger("aion.user_docs")


class UserDocumentStore:
    """Persistent storage for user-uploaded documents."""

    def __init__(self, db_path: str):
        """Initialize user document store.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

    async def init(self) -> None:
        """Initialize database schema."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await db.commit()
        logger.info("User document store initialized")

    async def add_document(self, title: str, content: str) -> str:
        """Add a new user document.

        Args:
            title: Document title
            content: Document content (text)

        Returns:
            Document ID
        """
        doc_id = str(uuid4())
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO user_documents (id, title, content) VALUES (?, ?, ?)",
                (doc_id, title, content),
            )
            await db.commit()
        logger.info("Added user document %r (id=%s)", title, doc_id)
        return doc_id

    async def get_document(self, doc_id: str) -> dict | None:
        """Retrieve a document by ID.

        Returns:
            Dict with id, title, content, created_at, updated_at; or None if not found
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, title, content, created_at, updated_at FROM user_documents WHERE id = ?",
                (doc_id,),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_documents(self) -> list[dict]:
        """List all user documents (metadata only, no content).

        Returns:
            List of dicts with id, title, created_at, updated_at
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, title, created_at, updated_at FROM user_documents ORDER BY created_at DESC"
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a user document.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM user_documents WHERE id = ?", (doc_id,))
            await db.commit()
            deleted = cursor.rowcount > 0
        
        if deleted:
            logger.info("Deleted user document id=%s", doc_id)
        else:
            logger.warning("Document not found: id=%s", doc_id)
        return deleted

    async def close(self) -> None:
        """Close database connection."""
        # aiosqlite handles cleanup per-query; nothing to do here
        pass
