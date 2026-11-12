"""Run the backend in-process: nano mode, the deterministic provider, SQLite in a temp dir.

Settings are read once, when app.config is first imported, so the environment is set here,
before any test imports the app.
"""
import os
import tempfile
from pathlib import Path

import pytest

_DB_DIR = Path(tempfile.mkdtemp(prefix="aion-backend-test-"))
os.environ.update(AION_MODE="nano", AION_PROVIDER="deterministic", AION_DB_PATH=str(_DB_DIR / "aion.db"))


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:  # runs the lifespan: store, user docs, RAG index
        yield c
