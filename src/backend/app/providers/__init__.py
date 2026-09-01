from __future__ import annotations

import logging
import time
from typing import Protocol, AsyncIterator

from app.config import AionProvider

logger = logging.getLogger("aion.provider")


class Provider(Protocol):
    async def generate(self, message: str, history: list[dict], params: dict | None = None) -> str: ...
    async def stream(self, message: str, history: list[dict], params: dict | None = None) -> AsyncIterator[str]: ...


_provider_instance: Provider | None = None


def get_provider(provider_type: AionProvider) -> Provider:
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    logger.info("Loading provider: %s ...", provider_type.value)
    t0 = time.monotonic()

    if provider_type == AionProvider.deterministic:
        from app.providers.deterministic import DeterministicProvider
        _provider_instance = DeterministicProvider()
    elif provider_type == AionProvider.local_mamba:
        from app.providers.local_mamba import LocalMambaProvider
        from app.config import settings
        _provider_instance = LocalMambaProvider(
            model_path=settings.mamba_model_path,
            tokenizer_path=settings.mamba_tokenizer_path,
            config_path=settings.mamba_config_path,
            device=settings.model_device,
        )
    elif provider_type == AionProvider.hf_local:
        from app.providers.hf_local import HFLocalProvider
        from app.config import settings
        _provider_instance = HFLocalProvider(
            model_id=settings.hf_model_id,
            device=settings.model_device,
            max_new_tokens=settings.hf_max_new_tokens,
            torch_dtype=settings.hf_torch_dtype,
        )
    else:
        raise NotImplementedError(f"Provider {provider_type.value} not yet available")

    elapsed = time.monotonic() - t0
    logger.info("Provider %s loaded in %.1fs", provider_type.value, elapsed)
    return _provider_instance


def reload_provider() -> None:
    """Drop cached provider so next get_provider() reloads from disk."""
    global _provider_instance
    _provider_instance = None
