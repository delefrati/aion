"""Shared provider base: common fallback + word-by-word streaming."""
from __future__ import annotations

from typing import AsyncIterator


class BaseProvider:
    """Base class for chat providers.

    Subclasses implement `generate`. The default `stream` re-emits the full
    response word-by-word (no true token streaming yet).
    """

    FALLBACK = "I'm not sure how to respond to that."

    async def generate(self, message: str, history: list[dict], params: dict | None = None) -> str:
        raise NotImplementedError

    async def stream(self, message: str, history: list[dict], params: dict | None = None) -> AsyncIterator[str]:
        response = await self.generate(message, history, params)
        for word in response.split(" "):
            yield word + " "
