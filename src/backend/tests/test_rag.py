"""RAG and chat through the HTTP API, in-process. These were scripts against a live server
on localhost:8900 (test_rag_e2e.py, test_rag_complete.py); the same flows, as tests."""
from __future__ import annotations

DOCS = [
    {"content": "Machine learning is a subset of artificial intelligence that trains algorithms "
                "to learn from data.", "source": "ml"},
    {"content": "Neural networks are computing systems inspired by the biological neural "
                "networks in animal brains.", "source": "nn"},
    {"content": "Transformers are deep learning models based on attention mechanisms, widely "
                "used in NLP.", "source": "transformers"},
    {"content": "Django is a high-level Python web framework for rapid development.", "source": "django"},
]


def _sse(text: str) -> list[tuple[str, str]]:
    """(event, data) pairs of a server-sent-events body."""
    events, event = [], "message"
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            events.append((event, line[5:].strip()))
    return events


def test_index_then_search_ranks_the_matching_document(client):
    r = client.post("/rag/index", json={"documents": DOCS, "namespace": "test"})
    assert r.status_code == 200 and r.json() == {"indexed": 4, "namespace": "test"}

    r = client.post("/rag/search", params={"query": "neural networks brains", "top_k": 2, "namespace": "test"})
    assert r.status_code == 200
    hits = r.json()
    assert 1 <= len(hits) <= 2 and hits[0]["source"] == "nn"

    r = client.post("/rag/search", params={"query": "neural networks", "namespace": "elsewhere"})
    assert r.json() == []


def _names(sources) -> list[str]:
    """Source labels without their score: 'transformers (score: 4.00)' -> 'transformers'."""
    return [s.split(" (score")[0] for s in sources]


def test_chat_without_rag(client):
    r = client.post("/chat", json={"message": "What is machine learning?", "use_rag": False})
    assert r.status_code == 200
    body = r.json()
    assert body["response"] and body["conversation_id"] and not body.get("retrieved_sources")


def test_chat_with_rag_returns_its_sources(client):
    client.post("/rag/index", json={"documents": DOCS, "namespace": "test"})
    r = client.post("/chat", json={"message": "Tell me about transformers and attention", "use_rag": True})
    assert r.status_code == 200
    assert _names(r.json()["retrieved_sources"])[0] == "transformers"


def test_streaming_chat_with_rag(client):
    client.post("/rag/index", json={"documents": DOCS, "namespace": "test"})
    with client.stream("POST", "/chat/stream",
                       json={"message": "Tell me about transformers in deep learning", "use_rag": True}) as r:
        assert r.status_code == 200
        events = _sse(r.read().decode())
    kinds = [e for e, _ in events]
    assert "token" in kinds and kinds[-1] == "done"
    sources = [d for e, d in events if e == "sources"]
    assert sources and _names(sources[0].split(";"))[0] == "transformers"


def test_user_documents_add_list_get_search_delete(client):
    ids = []
    for title, doc in (("ML", DOCS[0]), ("Django", DOCS[3])):
        r = client.post("/rag/documents/add", json={"title": title, "content": doc["content"]})
        assert r.status_code == 200
        ids.append(r.json()["id"])

    listed = client.get("/rag/documents").json()
    assert {d["id"] for d in listed} >= set(ids)
    assert client.get(f"/rag/documents/{ids[1]}").json()["content"] == DOCS[3]["content"]

    # Added documents are searchable in the "user" namespace without a separate index call.
    hits = client.post("/rag/search", params={"query": "Python web framework", "namespace": "user"}).json()
    assert hits and hits[0]["source"] == f"user_doc_{ids[1]}"

    assert client.delete(f"/rag/documents/{ids[1]}").status_code == 200
    assert client.get(f"/rag/documents/{ids[1]}").status_code == 404
    assert all(d["id"] != ids[1] for d in client.get("/rag/documents").json())
