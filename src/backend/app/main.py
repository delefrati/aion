from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)
logging.getLogger("aion").setLevel(logging.INFO)

from app.config import settings, AionMode
from app.routes.chat import router as chat_router
from app.routes.health import router as health_router


def _create_store():
    if settings.mode == AionMode.standard:
        from app.persistence.postgres import PostgresStore
        return PostgresStore(settings.database_url)
    else:
        from app.persistence.sqlite import SqliteStore
        return SqliteStore(settings.db_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = _create_store()
    await store.init()
    app.state.store = store
    yield
    await store.close()
    app.state.store = None


app = FastAPI(
    title="AION",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(chat_router)
