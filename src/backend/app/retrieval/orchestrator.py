"""RAG orchestrator: manages retrieval and context augmentation."""
from __future__ import annotations

import logging
from typing import Sequence

from app.retrieval.base import BaseRetriever, Document
from app.retrieval.bm25 import BM25Retriever

logger = logging.getLogger("aion.rag")


class RAGOrchestrator:
    """Orchestrates retrieval-augmented generation for chat."""

    def __init__(self, retriever: BaseRetriever | None = None):
        """Initialize RAG orchestrator.

        Args:
            retriever: Retriever backend (default: BM25)
        """
        self.retriever = retriever or BM25Retriever()

    async def index_documents(
        self, documents: Sequence[Document], namespace: str = "default"
    ) -> None:
        """Index documents for retrieval.

        Args:
            documents: List of Document objects
            namespace: Logical group (e.g., "training", "user", "instructions")
        """
        logger.info("Indexing %d documents in namespace %r", len(documents), namespace)
        await self.retriever.index(documents, namespace)

    async def retrieve(
        self, query: str, top_k: int = 5, namespace: str | None = None
    ) -> list[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query (typically the user message)
            top_k: Number of results
            namespace: Filter by namespace (None = all)

        Returns:
            List of relevant documents, ordered by relevance
        """
        logger.debug("Retrieving top-%d docs for query %r", top_k, query[:80])
        docs = await self.retriever.search(query, top_k=top_k, namespace=namespace)
        logger.info("Retrieved %d documents", len(docs))
        return docs

    async def augment_prompt(
        self,
        user_message: str,
        top_k: int = 5,
        namespace: str | None = None,
        max_context_chars: int = 2000,
    ) -> tuple[str, list[Document]]:
        """Augment the user message with retrieved context.

        Args:
            user_message: Original user query
            top_k: Max number of documents to retrieve
            namespace: Filter by namespace
            max_context_chars: Max total context length (truncate if needed)

        Returns:
            Tuple of (augmented_prompt, retrieved_documents)
        """
        docs = await self.retrieve(user_message, top_k=top_k, namespace=namespace)

        if not docs:
            # No docs retrieved; return original message unchanged
            return user_message, []

        # Build context block
        context_lines = ["Retrieved context:"]
        total_chars = 0

        for i, doc in enumerate(docs, 1):
            source_label = f"[{doc.source}]"
            entry = f"{source_label} {doc.content[:200]}..."  # Truncate long docs
            
            if total_chars + len(entry) > max_context_chars:
                context_lines.append("(... truncated)")
                break
            
            context_lines.append(entry)
            total_chars += len(entry)

        context_block = "\n".join(context_lines)

        # Augment prompt: context first, then original query
        augmented = f"{context_block}\n\nQuestion: {user_message}"

        logger.debug("Augmented prompt length: %d chars", len(augmented))
        return augmented, docs

    async def clear(self, namespace: str | None = None) -> None:
        """Clear indexed documents.

        Args:
            namespace: If provided, clear only that namespace; else clear all
        """
        logger.info("Clearing namespace %r", namespace)
        await self.retriever.clear(namespace=namespace)

    async def remove(self, source: str, namespace: str | None = None) -> None:
        """Remove all indexed documents matching a source id.

        Args:
            source: Document source identifier (e.g., "user_doc_<id>")
            namespace: If provided, only remove from that namespace; else all
        """
        logger.info("Removing source %r from namespace %r", source, namespace)
        await self.retriever.remove(source, namespace=namespace)
