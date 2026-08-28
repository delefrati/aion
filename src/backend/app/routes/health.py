from __future__ import annotations

from fastapi import APIRouter

from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready():
    return {
        "status": "ready",
        "mode": settings.mode.value,
        "provider": settings.provider.value,
    }


@router.post("/reload-model")
async def reload_model():
    from app.providers import reload_provider
    reload_provider()
    return {"status": "reloaded"}
