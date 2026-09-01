from __future__ import annotations

import logging
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.persistence.store import Store
from app.providers import get_provider
from app.retrieval.orchestrator import RAGOrchestrator

logger = logging.getLogger("aion.chat")

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    use_rag: bool = False  # Enable retrieval-augmented generation
    rag_top_k: int | None = None  # Number of documents to retrieve when use_rag is set

    # Advanced generation overrides. Any field left unset (None) falls back to
    # the provider's configured default.
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None
    history_turns: int | None = None

    def gen_params(self) -> dict:
        """Non-None generation overrides, ready to forward to a provider."""
        fields = ("temperature", "top_k", "top_p", "repetition_penalty", "max_tokens", "history_turns")
        return {f: v for f in fields if (v := getattr(self, f)) is not None}


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    retrieved_sources: list[str] | None = None  # Sources used if RAG was enabled


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_rag(request: Request) -> RAGOrchestrator:
    return getattr(request.app.state, "rag", None)


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, store: Store = Depends(get_store), rag: RAGOrchestrator = Depends(get_rag)):
    conv_id = req.conversation_id or str(uuid.uuid4())
    logger.info("POST /chat conv=%s msg=%r use_rag=%s", conv_id[:8], req.message[:80], req.use_rag)
    history = await store.get_conversation(conv_id)
    await store.save_message(conv_id, "user", req.message)

    # Optionally augment with retrieved context
    user_message = req.message
    retrieved_sources = None

    if req.use_rag and rag:
        user_message, docs = await rag.augment_prompt(req.message, top_k=req.rag_top_k or 5)
        retrieved_sources = [f"{doc.source} (score: {doc.score:.2f})" for doc in docs]
        logger.info("RAG augmented prompt with %d sources", len(docs))

    provider = get_provider(settings.provider)
    params = req.gen_params()
    t0 = time.monotonic()
    logger.info("Generating response (provider=%s, history=%d turns, rag=%s, params=%s)...", settings.provider, len(history), req.use_rag, params)
    response = await provider.generate(user_message, history, params)
    elapsed = time.monotonic() - t0
    logger.info("Generated %d chars in %.1fs", len(response), elapsed)

    await store.save_message(conv_id, "assistant", response)
    return ChatResponse(
        conversation_id=conv_id,
        response=response,
        retrieved_sources=retrieved_sources,
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest, store: Store = Depends(get_store), rag: RAGOrchestrator = Depends(get_rag)):
    conv_id = req.conversation_id or str(uuid.uuid4())
    logger.info("POST /chat/stream conv=%s msg=%r use_rag=%s", conv_id[:8], req.message[:80], req.use_rag)
    history = await store.get_conversation(conv_id)
    await store.save_message(conv_id, "user", req.message)

    # Optionally augment with retrieved context
    user_message = req.message
    retrieved_docs = []

    if req.use_rag and rag:
        user_message, retrieved_docs = await rag.augment_prompt(req.message, top_k=req.rag_top_k or 5)
        logger.info("RAG augmented prompt with %d sources", len(retrieved_docs))

    provider = get_provider(settings.provider)
    params = req.gen_params()

    async def event_generator() -> AsyncIterator[dict]:
        t0 = time.monotonic()
        logger.info("Streaming response (provider=%s, history=%d turns, rag=%s, params=%s)...", settings.provider, len(history), req.use_rag, params)

        # Emit retrieved sources if RAG was used
        if retrieved_docs:
            sources = [f"{doc.source} (score: {doc.score:.2f})" for doc in retrieved_docs]
            yield {"event": "sources", "data": ";".join(sources)}

        full_response = []
        async for token in provider.stream(user_message, history, params):
            full_response.append(token)
            yield {"event": "token", "data": token}
        response_text = "".join(full_response)
        elapsed = time.monotonic() - t0
        logger.info("Streamed %d chars in %.1fs", len(response_text), elapsed)
        await store.save_message(conv_id, "assistant", response_text)
        yield {"event": "done", "data": conv_id}

    return EventSourceResponse(event_generator())
