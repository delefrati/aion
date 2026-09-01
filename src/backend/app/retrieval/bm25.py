"""BM25 full-text retriever (keyword search without embeddings)."""
from __future__ import annotations

import re
from typing import Sequence
from collections import defaultdict

from app.retrieval.base import BaseRetriever, Document


class BM25Retriever(BaseRetriever):
    """BM25-based keyword retriever. Uses TF-IDF-like scoring, no ML needed."""

    def __init__(self, use_rank_bm25: bool = True):
        """Initialize BM25 retriever.

        Args:
            use_rank_bm25: Use rank_bm25 library if available (better performance),
                          else fallback to simple TF-based scoring
        """
        self.use_rank_bm25 = use_rank_bm25
        self._bm25 = None  # Will be lazily imported if use_rank_bm25=True
        
        # Storage: namespace -> list of (Document, tokenized_text)
        self._documents: dict[str, list[tuple[Document, list[str]]]] = defaultdict(list)
        self._corpus_by_namespace: dict[str, list[list[str]]] = defaultdict(list)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric."""
        text = text.lower()
        # Split on whitespace and punctuation; keep meaningful tokens
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    async def index(self, documents: Sequence[Document], namespace: str = "default") -> None:
        """Index documents for a given namespace."""
        if namespace not in self._documents:
            self._documents[namespace] = []
            self._corpus_by_namespace[namespace] = []

        # Tokenize and store
        for doc in documents:
            tokens = self._tokenize(doc.content)
            self._documents[namespace].append((doc, tokens))
            self._corpus_by_namespace[namespace].append(tokens)

        # If using rank_bm25, rebuild the index
        if self.use_rank_bm25:
            self._rebuild_bm25(namespace)

    def _rebuild_bm25(self, namespace: str) -> None:
        """Rebuild BM25 index for a namespace."""
        try:
            from rank_bm25 import BM25Okapi

            corpus = self._corpus_by_namespace[namespace]
            if corpus:
                self._bm25 = BM25Okapi(corpus)
        except ImportError:
            # Fallback: rank_bm25 not available, will use TF-based scoring
            pass

    # Below this BM25 score, a document is considered unrelated to the query and
    # must not be surfaced as "relevant context" (a low/irrelevant query would
    # otherwise still return whatever top-k docs happen to be indexed).
    MIN_RELEVANCE_SCORE = 0.1

    async def search(
        self, query: str, top_k: int = 5, namespace: str | None = None
    ) -> list[Document]:
        """Search for documents matching the query."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Determine which namespaces to search
        namespaces_to_search = [namespace] if namespace else self._documents.keys()
        
        scored_docs: list[tuple[Document, float]] = []

        for ns in namespaces_to_search:
            if ns not in self._documents:
                continue

            if self.use_rank_bm25 and self._bm25:
                # Use BM25 scoring
                scores = self._bm25.get_scores(query_tokens)
                for doc, score in zip(
                    [d for d, _ in self._documents[ns]], scores
                ):
                    scored_docs.append((doc, float(score)))
            else:
                # Fallback: simple TF scoring
                for doc, doc_tokens in self._documents[ns]:
                    score = sum(1 for token in query_tokens if token in doc_tokens)
                    scored_docs.append((doc, float(score)))

        # Sort by score descending, drop irrelevant matches, then return top-k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []
        for doc, score in scored_docs[:top_k]:
            if score < self.MIN_RELEVANCE_SCORE:
                continue
            doc_copy = Document(
                content=doc.content,
                source=doc.source,
                score=score,
            )
            result.append(doc_copy)

        return result

    async def clear(self, namespace: str | None = None) -> None:
        """Clear indexed documents."""
        if namespace:
            if namespace in self._documents:
                del self._documents[namespace]
            if namespace in self._corpus_by_namespace:
                del self._corpus_by_namespace[namespace]
        else:
            self._documents.clear()
            self._corpus_by_namespace.clear()
        self._bm25 = None

    async def remove(self, source: str, namespace: str | None = None) -> None:
        """Remove all documents matching a source id (e.g. after a delete)."""
        namespaces = [namespace] if namespace else list(self._documents.keys())
        for ns in namespaces:
            if ns not in self._documents:
                continue
            kept = [(d, t) for d, t in self._documents[ns] if d.source != source]
            if len(kept) == len(self._documents[ns]):
                continue
            self._documents[ns] = kept
            self._corpus_by_namespace[ns] = [t for _, t in kept]
            if self.use_rank_bm25:
                self._rebuild_bm25(ns)
