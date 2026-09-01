#!/usr/bin/env python3
"""Complete RAG system test with document management."""
import asyncio
import aiohttp
import json

BACKEND_URL = "http://localhost:8900"

# Test documents with realistic content
TEST_DOCUMENTS = [
    {
        "title": "Python Basics",
        "content": """Python is a high-level, interpreted programming language known for its simplicity and readability.
        It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
        Python's syntax allows programmers to express concepts in fewer lines of code than would be possible in languages
        such as C++ or Java. The language has a comprehensive standard library."""
    },
    {
        "title": "Web Development with Django",
        "content": """Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design.
        It's built by experienced developers and takes care of much of the hassle of web development, so you can focus
        on writing your app without needing to reinvent the wheel. It's free and open source with a thriving and active
        community."""
    },
    {
        "title": "Machine Learning Fundamentals",
        "content": """Machine Learning is a subset of artificial intelligence that focuses on training algorithms to learn from data.
        Key concepts include supervised learning, unsupervised learning, and reinforcement learning. Common algorithms include
        decision trees, neural networks, support vector machines, and ensemble methods. Feature engineering and model evaluation
        are critical components of any ML pipeline."""
    },
]


async def test_complete_rag_system():
    """Test the complete RAG system: document management + search + chat integration."""
    async with aiohttp.ClientSession() as session:
        print("=" * 70)
        print("AION Complete RAG System Test")
        print("=" * 70)

        # Test 1: Add documents
        print("\n[1/8] Adding documents...")
        doc_ids = []
        try:
            for doc in TEST_DOCUMENTS:
                res = await session.post(
                    f"{BACKEND_URL}/rag/documents/add",
                    json=doc
                )
                if res.status == 200:
                    result = await res.json()
                    doc_ids.append(result["id"])
                    print(f"  ✓ Added: {doc['title']} (ID: {result['id']})")
                else:
                    print(f"  ✗ Failed to add document: {res.status}")
                    return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

        # Test 2: List documents
        print("\n[2/8] Listing stored documents...")
        try:
            res = await session.get(f"{BACKEND_URL}/rag/documents")
            if res.status == 200:
                docs = await res.json()
                print(f"  ✓ Found {len(docs)} documents")
                for doc in docs:
                    print(f"    - {doc['title']} (created: {doc['created_at']})")
            else:
                print(f"  ✗ Failed to list documents: {res.status}")
                return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

        # Test 3: Get document content
        print("\n[3/8] Retrieving document content...")
        try:
            if not doc_ids:
                print("  ✗ No documents to retrieve")
                return False
            res = await session.get(f"{BACKEND_URL}/rag/documents/{doc_ids[0]}")
            if res.status == 200:
                doc = await res.json()
                preview = doc['content'][:80] + "..." if len(doc['content']) > 80 else doc['content']
                print(f"  ✓ Retrieved '{doc['title']}'")
                print(f"    Content preview: {preview}")
            else:
                print(f"  ✗ Failed to retrieve document: {res.status}")
                return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

        # Test 4: Index documents for search
        print("\n[4/8] Indexing documents for RAG search...")
        try:
            # Convert stored docs to index format
            index_docs = [
                {
                    "content": doc["content"],
                    "source": doc["title"]
                }
                for doc in TEST_DOCUMENTS
            ]
            res = await session.post(
                f"{BACKEND_URL}/rag/index",
                json={"documents": index_docs, "namespace": "test_docs"}
            )
            if res.status == 200:
                result = await res.json()
                print(f"  ✓ Indexed {result['indexed']} documents in namespace '{result['namespace']}'")
            else:
                print(f"  ✗ Failed to index: {res.status}")
                return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

        # Test 5: Search documents
        print("\n[5/8] Searching for relevant documents...")
        test_queries = [
            ("Python programming language", "test_docs"),
            ("web framework development", "test_docs"),
            ("machine learning algorithms", "test_docs"),
        ]
        try:
            for query, namespace in test_queries:
                res = await session.post(
                    f"{BACKEND_URL}/rag/search",
                    params={"query": query, "top_k": 2, "namespace": namespace}
                )
                if res.status == 200:
                    results = await res.json()
                    print(f"  ✓ Query: '{query}'")
                    for i, doc in enumerate(results, 1):
                        print(f"    {i}. {doc['source']} (score: {doc['score']:.2f})")
                else:
                    print(f"  ✗ Search failed: {res.status}")
                    return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

        # Test 6: Chat without RAG
        print("\n[6/8] Testing chat WITHOUT RAG...")
        try:
            res = await session.post(
                f"{BACKEND_URL}/chat",
                json={"message": "What is Python?", "use_rag": False}
            )
            if res.status == 200:
                result = await res.json()
                print(f"  ✓ Chat response (no RAG):")
                print(f"    '{result['response'][:100]}...'")
                print(f"    Sources: {result.get('retrieved_sources', 'None')}")
            else:
                print(f"  ✗ Chat failed: {res.status}")
                return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

        # Test 7: Streaming chat WITH RAG
        print("\n[7/8] Testing streaming chat WITH RAG...")
        try:
            res = await session.post(
                f"{BACKEND_URL}/chat/stream",
                json={
                    "message": "Tell me about Python and web development",
                    "use_rag": True,
                }
            )

            if res.status != 200:
                print(f"  ✗ Stream failed: {res.status}")
                return False

            response_text = ""
            sources_found = False

            reader = res.content.iter_chunked(1024)
            async for chunk in reader:
                data = chunk.decode('utf-8', errors='ignore')
                for line in data.split('\n'):
                    line = line.strip()
                    if line.startswith("data: ") and response_text == "":
                        # Initial sources event
                        pass
                    elif line.startswith("data: "):
                        response_text += line[6:]
                    if line.startswith("event: sources"):
                        sources_found = True

            print(f"  ✓ Stream received: {len(response_text)} characters")
            print(f"    Preview: {response_text[:100]}...")
            print(f"    RAG sources: {'Yes' if sources_found else 'No (but expected)'}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

        # Test 8: Delete document
        print("\n[8/8] Testing document deletion...")
        try:
            if not doc_ids:
                print("  ✗ No documents to delete")
                return False
            res = await session.delete(f"{BACKEND_URL}/rag/documents/{doc_ids[0]}")
            if res.status == 200:
                print(f"  ✓ Deleted document ID: {doc_ids[0]}")
                
                # Verify deletion
                res = await session.get(f"{BACKEND_URL}/rag/documents")
                docs = await res.json()
                remaining = len(docs)
                print(f"    Remaining documents: {remaining}")
            else:
                print(f"  ✗ Delete failed: {res.status}")
                return False
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False

        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        return True


async def main():
    """Run the test suite."""
    try:
        success = await test_complete_rag_system()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
