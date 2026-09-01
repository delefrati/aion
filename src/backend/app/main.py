from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logging.getLogger("aion").setLevel(logging.INFO)

from app.config import settings, AionMode
from app.routes.chat import router as chat_router
from app.routes.health import router as health_router
from app.routes.rag import router as rag_router
from app.retrieval.orchestrator import RAGOrchestrator
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.base import Document
from app.persistence.user_docs import UserDocumentStore


def _create_store():
    if settings.mode == AionMode.standard:
        from app.persistence.postgres import PostgresStore
        return PostgresStore(settings.database_url)
    else:
        from app.persistence.sqlite import SqliteStore
        return SqliteStore(settings.db_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize persistence store
    store = _create_store()
    await store.init()
    app.state.store = store

    # Initialize user document store
    db_path = getattr(settings, "db_path", "aion.db")
    user_docs = UserDocumentStore(db_path)
    await user_docs.init()
    app.state.user_docs = user_docs
    logger = logging.getLogger("aion.main")
    logger.info("User document store initialized at %s", db_path)

    # Initialize RAG orchestrator
    retriever = BM25Retriever()
    rag = RAGOrchestrator(retriever=retriever)
    app.state.rag = rag
    logger.info("RAG orchestrator initialized with BM25 retriever")

    # The BM25 index is in-memory only; reload persisted user documents so a
    # restart doesn't silently make the "user" namespace unsearchable.
    existing_docs = await user_docs.list_documents()
    if existing_docs:
        docs_to_index = []
        for meta in existing_docs:
            full_doc = await user_docs.get_document(meta["id"])
            if full_doc:
                docs_to_index.append(
                    Document(content=full_doc["content"], source=f"user_doc_{full_doc['id']}")
                )
        if docs_to_index:
            await rag.index_documents(docs_to_index, namespace="user")
            logger.info("Reloaded %d user documents into RAG index", len(docs_to_index))

    yield

    # Cleanup
    await store.close()
    app.state.store = None
    await user_docs.close()
    app.state.user_docs = None
    app.state.rag = None


app = FastAPI(
    title="AION",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(rag_router)

