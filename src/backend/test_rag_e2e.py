#!/usr/bin/env python3
"""End-to-end test script for RAG functionality."""
import asyncio
import aiohttp
import json
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8900"
TEST_DOCUMENTS = [
    {
        "content": "Machine learning is a subset of artificial intelligence that focuses on training algorithms to learn from data.",
        "source": "test_doc_1"
    },
    {
        "content": "Neural networks are computing systems inspired by biological neural networks found in animal brains.",
        "source": "test_doc_2"
    },
    {
        "content": "Deep learning uses neural networks with multiple layers to learn hierarchical representations of data.",
        "source": "test_doc_3"
    },
    {
        "content": "Transformers are deep learning models based on attention mechanisms, widely used in NLP tasks.",
        "source": "test_doc_4"
    },
]


async def test_rag_flow():
    """Test the complete RAG flow: index, search, and chat with RAG."""
    async with aiohttp.ClientSession() as session:
        print("=" * 60)
        print("AION RAG End-to-End Test")
        print("=" * 60)

        # Test 1: Index test documents
        print("\n[1/4] Indexing test documents...")
        try:
            async with session.post(
                f"{BACKEND_URL}/rag/index",
                json={
                    "documents": TEST_DOCUMENTS,
                    "namespace": "test"
                }
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"✓ Indexed {result['indexed']} documents in namespace '{result['namespace']}'")
                else:
                    print(f"✗ Failed to index documents: {resp.status}")
                    return False
        except Exception as e:
            print(f"✗ Connection error: {e}")
            return False

        # Test 2: Search documents (without RAG)
        print("\n[2/4] Testing search functionality...")
        try:
            async with session.post(
                f"{BACKEND_URL}/rag/search",
                params={"query": "neural networks learning", "top_k": 2, "namespace": "test"}
            ) as resp:
                if resp.status == 200:
                    results = await resp.json()
                    print(f"✓ Search returned {len(results)} results:")
                    for i, doc in enumerate(results, 1):
                        print(f"  {i}. {doc['source']} (score: {doc['score']:.2f})")
                        print(f"     {doc['content'][:80]}...")
                else:
                    print(f"✗ Search failed: {resp.status}")
                    return False
        except Exception as e:
            print(f"✗ Search error: {e}")
            return False

        # Test 3: Chat without RAG
        print("\n[3/4] Testing chat without RAG...")
        try:
            async with session.post(
                f"{BACKEND_URL}/chat",
                json={
                    "message": "What is machine learning?",
                    "use_rag": False
                }
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"✓ Chat response (no RAG):")
                    print(f"  '{result['response'][:100]}...'")
                    print(f"  Sources: {result.get('retrieved_sources', 'None')}")
                else:
                    print(f"✗ Chat failed: {resp.status}")
                    return False
        except Exception as e:
            print(f"✗ Chat error: {e}")
            return False

        # Test 4: Chat with RAG (streaming)
        print("\n[4/4] Testing streaming chat with RAG...")
        try:
            async with session.post(
                f"{BACKEND_URL}/chat/stream",
                json={
                    "message": "Tell me about transformers in deep learning",
                    "use_rag": True
                }
            ) as resp:
                if resp.status == 200:
                    print("✓ Streaming chat with RAG started:")
                    
                    response_text = ""
                    sources = []
                    
                    # Read SSE events
                    async for line in resp.content:
                        line = line.decode().strip()
                        if line.startswith("event: "):
                            event_type = line[7:]
                        elif line.startswith("data: "):
                            data = line[6:]
                            if event_type == "token":
                                response_text += data
                            elif event_type == "sources":
                                sources = data.split(";")
                            elif event_type == "done":
                                conv_id = data
                    
                    print(f"  Response: '{response_text[:100]}...'")
                    if sources:
                        print(f"  Retrieved sources:")
                        for source in sources:
                            print(f"    - {source}")
                    else:
                        print("  ⚠ No sources retrieved (BM25 index might be empty)")
                else:
                    print(f"✗ Streaming chat failed: {resp.status}")
                    return False
        except Exception as e:
            print(f"✗ Streaming chat error: {e}")
            return False

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return True


if __name__ == "__main__":
    success = asyncio.run(test_rag_flow())
    exit(0 if success else 1)
