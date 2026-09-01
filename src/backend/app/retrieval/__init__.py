"""RAG retrieval module with pluggable backends (BM25 -> embeddings)."""
from app.retrieval.base import BaseRetriever, Document
from app.retrieval.bm25 import BM25Retriever

__all__ = ["BaseRetriever", "Document", "BM25Retriever"]
