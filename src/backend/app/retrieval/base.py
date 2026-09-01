"""Abstract base class for retrieval backends."""
from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Sequence


@dataclass
class Document:
    """A document with content and optional metadata."""

    content: str
    source: str  # e.g., "training", "user_doc_123", "instruction_seed"
    score: float | None = None  # relevance score, if applicable


class BaseRetriever(ABC):
    """Abstract retriever interface for all backends (BM25, embeddings, etc)."""

    @abstractmethod
    async def index(self, documents: Sequence[Document], namespace: str = "default") -> None:
        """Index a batch of documents.

        Args:
            documents: List of Document objects to index
            namespace: Logical group (e.g., "training", "user", "instructions")
        """
        pass

    @abstractmethod
    async def search(
        self, query: str, top_k: int = 5, namespace: str | None = None
    ) -> list[Document]:
        """Retrieve top-k documents matching the query.

        Args:
            query: Search query
            top_k: Number of results to return
            namespace: Filter by namespace (None = search all)

        Returns:
            List of Document objects, ordered by relevance (highest first)
        """
        pass

    @abstractmethod
    async def clear(self, namespace: str | None = None) -> None:
        """Clear indexed documents.

        Args:
            namespace: If provided, clear only that namespace; else clear all
        """
        pass

    @abstractmethod
    async def remove(self, source: str, namespace: str | None = None) -> None:
        """Remove all documents matching a source id.

        Args:
            source: Document source identifier (e.g., "user_doc_<id>")
            namespace: If provided, only remove from that namespace; else all
        """
        pass
