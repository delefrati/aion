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

logger = logging.getLogger("aion.chat")

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    response: str


def get_store(request: Request) -> Store:
    return request.app.state.store


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, store: Store = Depends(get_store)):
    conv_id = req.conversation_id or str(uuid.uuid4())
    logger.info("POST /chat conv=%s msg=%r", conv_id[:8], req.message[:80])
    history = await store.get_conversation(conv_id)
    await store.save_message(conv_id, "user", req.message)

    provider = get_provider(settings.provider)
    t0 = time.monotonic()
    logger.info("Generating response (provider=%s, history=%d turns)...", settings.provider, len(history))
    response = await provider.generate(req.message, history)
    elapsed = time.monotonic() - t0
    logger.info("Generated %d chars in %.1fs", len(response), elapsed)

    await store.save_message(conv_id, "assistant", response)
    return ChatResponse(conversation_id=conv_id, response=response)


@router.post("/stream")
async def chat_stream(req: ChatRequest, store: Store = Depends(get_store)):
    conv_id = req.conversation_id or str(uuid.uuid4())
    logger.info("POST /chat/stream conv=%s msg=%r", conv_id[:8], req.message[:80])
    history = await store.get_conversation(conv_id)
    await store.save_message(conv_id, "user", req.message)

    provider = get_provider(settings.provider)

    async def event_generator() -> AsyncIterator[dict]:
        t0 = time.monotonic()
        logger.info("Streaming response (provider=%s, history=%d turns)...", settings.provider, len(history))
        full_response = []
        async for token in provider.stream(req.message, history):
            full_response.append(token)
            yield {"event": "token", "data": token}
        response_text = "".join(full_response)
        elapsed = time.monotonic() - t0
        logger.info("Streamed %d chars in %.1fs", len(response_text), elapsed)
        await store.save_message(conv_id, "assistant", response_text)
        yield {"event": "done", "data": conv_id}

    return EventSourceResponse(event_generator())
