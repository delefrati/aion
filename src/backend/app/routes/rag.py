"""RAG management endpoints: indexing, document upload, cache management."""
from __future__ import annotations

import logging
from typing import Sequence

from fastapi import APIRouter, Depends, Request, HTTPException
from pydantic import BaseModel

from app.retrieval.base import Document
from app.retrieval.orchestrator import RAGOrchestrator
from app.persistence.user_docs import UserDocumentStore

logger = logging.getLogger("aion.rag")

router = APIRouter(prefix="/rag", tags=["rag"])


class DocumentInput(BaseModel):
    """Schema for indexing documents."""

    content: str
    source: str  # e.g., "user_doc_123", "training", "instructions"


class IndexRequest(BaseModel):
    """Request to index a batch of documents."""

    documents: Sequence[DocumentInput]
    namespace: str = "default"  # Logical group (e.g., "training", "user", "instructions")


class IndexResponse(BaseModel):
    """Response from indexing."""

    indexed: int
    namespace: str


class ClearRequest(BaseModel):
    """Request to clear indexed documents."""

    namespace: str | None = None  # None = clear all


class ClearResponse(BaseModel):
    """Response from clearing."""

    cleared: str  # e.g., "all" or "namespace:default"


def get_rag(request: Request) -> RAGOrchestrator:
    """Get RAG orchestrator from app state."""
    rag = getattr(request.app.state, "rag", None)
    if not rag:
        raise RuntimeError("RAG orchestrator not initialized in app state")
    return rag


def get_user_docs(request: Request) -> UserDocumentStore:
    """Get user document store from app state."""
    user_docs = getattr(request.app.state, "user_docs", None)
    if not user_docs:
        raise RuntimeError("User document store not initialized in app state")
    return user_docs


@router.post("/index", response_model=IndexResponse)
async def index_documents(
    req: IndexRequest, rag: RAGOrchestrator = Depends(get_rag)
) -> IndexResponse:
    """Index a batch of documents for retrieval.

    Example:
        POST /rag/index
        {
            "documents": [
                {"content": "Machine learning is...", "source": "training"},
                {"content": "Neural networks...", "source": "training"}
            ],
            "namespace": "training"
        }
    """
    docs = [
        Document(content=d.content, source=d.source) for d in req.documents
    ]
    await rag.index_documents(docs, namespace=req.namespace)
    logger.info("Indexed %d documents in namespace %r", len(docs), req.namespace)
    return IndexResponse(indexed=len(docs), namespace=req.namespace)


@router.post("/search", response_model=list[dict])
async def search_documents(
    query: str, top_k: int = 5, namespace: str | None = None, rag: RAGOrchestrator = Depends(get_rag)
) -> list[dict]:
    """Search for relevant documents.

    Args:
        query: Search query
        top_k: Number of results
        namespace: Filter by namespace (None = all)

    Returns:
        List of documents with content, source, and relevance score
    """
    docs = await rag.retrieve(query, top_k=top_k, namespace=namespace)
    return [
        {
            "content": doc.content[:200],  # Truncate for response
            "source": doc.source,
            "score": doc.score,
        }
        for doc in docs
    ]


@router.post("/clear", response_model=ClearResponse)
async def clear_documents(
    req: ClearRequest, rag: RAGOrchestrator = Depends(get_rag)
) -> ClearResponse:
    """Clear indexed documents.

    Args:
        namespace: If provided, clear only that namespace; else clear all
    """
    await rag.clear(namespace=req.namespace)
    cleared_label = req.namespace or "all"
    logger.info("Cleared RAG namespace %r", cleared_label)
    return ClearResponse(cleared=cleared_label)


# ============== User Document Management ==============


class UserDocumentInput(BaseModel):
    """Input for adding a new document."""

    title: str
    content: str


class UserDocumentResponse(BaseModel):
    """Response for user document operations."""

    id: str
    title: str
    created_at: str | None = None
    updated_at: str | None = None


class UserDocumentDetailResponse(UserDocumentResponse):
    """Detailed response including content."""

    content: str


@router.post("/documents/add", response_model=UserDocumentResponse)
async def add_document(
    req: UserDocumentInput,
    rag: RAGOrchestrator = Depends(get_rag),
    user_docs: UserDocumentStore = Depends(get_user_docs),
) -> UserDocumentResponse:
    """Add a new document to the knowledge base and index it for RAG.

    Args:
        req: Document input with title and content

    Returns:
        Document metadata (id, title, created_at)
    """
    # Store in database
    doc_id = await user_docs.add_document(req.title, req.content)

    # Index for RAG
    doc = Document(content=req.content, source=f"user_doc_{doc_id}")
    await rag.index_documents([doc], namespace="user")

    logger.info("Added and indexed document %r (id=%s)", req.title, doc_id)
    return UserDocumentResponse(id=doc_id, title=req.title)


@router.get("/documents", response_model=list[UserDocumentResponse])
async def list_documents(
    user_docs: UserDocumentStore = Depends(get_user_docs),
) -> list[UserDocumentResponse]:
    """List all user documents."""
    docs = await user_docs.list_documents()
    return [
        UserDocumentResponse(
            id=d["id"],
            title=d["title"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
        )
        for d in docs
    ]


@router.get("/documents/{doc_id}", response_model=UserDocumentDetailResponse)
async def get_document(
    doc_id: str, user_docs: UserDocumentStore = Depends(get_user_docs)
) -> UserDocumentDetailResponse:
    """Get a specific user document (with content)."""
    doc = await user_docs.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return UserDocumentDetailResponse(
        id=doc["id"],
        title=doc["title"],
        content=doc["content"],
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    rag: RAGOrchestrator = Depends(get_rag),
    user_docs: UserDocumentStore = Depends(get_user_docs),
) -> dict:
    """Delete a user document and remove from RAG index.

    Args:
        doc_id: Document ID

    Returns:
        Success message
    """
    # Delete from database
    deleted = await user_docs.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    # Also remove from the live RAG index, or it stays retrievable/leakable
    # until the next server restart even though it's gone from the DB.
    await rag.remove(f"user_doc_{doc_id}", namespace="user")

    logger.info("Deleted document id=%s", doc_id)
    return {"message": f"Document {doc_id} deleted"}
