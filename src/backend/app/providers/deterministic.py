from __future__ import annotations

from app.providers.base import BaseProvider


class DeterministicProvider(BaseProvider):
    """Rule/template-based response provider. No model needed."""

    GREETINGS = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"}

    async def generate(self, message: str, history: list[dict], params: dict | None = None) -> str:
        return self._respond(message, history)

    def _respond(self, message: str, history: list[dict]) -> str:
        clean = message.strip().lower().rstrip("!?.")

        if clean in self.GREETINGS:
            return "Hello! How can I help you today?"

        if clean in ("help", "what can you do"):
            return (
                "I'm AION, a local-first conversational assistant. "
                "I can chat with you, and as more capabilities come online, "
                "I'll be able to do much more."
            )

        if "?" in message:
            return (
                "That's an interesting question. "
                "I'm currently running in deterministic mode with limited capabilities. "
                "More advanced reasoning will be available once a local model is loaded."
            )

        return (
            "I hear you. I'm running in deterministic mode right now, "
            "so my responses are template-based. "
            "Once a local model is available, I'll be able to have richer conversations."
        )
