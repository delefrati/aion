"""CLI command to seed RAG index with training corpus."""
import asyncio
import sys
from pathlib import Path

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.orchestrator import RAGOrchestrator
from app.retrieval.corpus_loader import CorpusLoader


async def seed_rag_index(
    instruction_file: str | None = None,
    training_data_dir: str | None = None,
    max_training_docs: int = 100,
) -> None:
    """Seed the RAG index with training corpus and instruction data.

    Args:
        instruction_file: Path to instruction_seed.json
        training_data_dir: Path to training data directory
        max_training_docs: Max number of training docs to load
    """
    retriever = BM25Retriever()
    rag = RAGOrchestrator(retriever=retriever)

    # Load and index instruction dataset
    if instruction_file and Path(instruction_file).exists():
        print(f"Loading instruction dataset from {instruction_file}...")
        docs = []
        async for doc in CorpusLoader.load_instruction_dataset(instruction_file):
            docs.append(doc)

        if docs:
            await rag.index_documents(docs, namespace="instructions")
            print(f"✓ Indexed {len(docs)} instruction documents")
        else:
            print("✗ No instruction documents found")
    else:
        print("⊘ Skipping instructions (file not found or not provided)")

    # Load and index training corpus sample
    if training_data_dir and Path(training_data_dir).exists():
        print(f"Loading training corpus sample from {training_data_dir}...")
        docs = []
        async for doc in CorpusLoader.load_training_corpus_sample(
            training_data_dir,
            max_documents=max_training_docs,
        ):
            docs.append(doc)

        if docs:
            await rag.index_documents(docs, namespace="training")
            print(f"✓ Indexed {len(docs)} training documents")
        else:
            print("✗ No training documents found")
    else:
        print("⊘ Skipping training corpus (directory not found or not provided)")

    print("\nRAG index seeded successfully!")


if __name__ == "__main__":
    # Default paths (adjust as needed)
    instruction_file = "llm_lab/data/instruction_seed.json"
    training_data_dir = "../training-data"
    max_training_docs = 50

    # Allow override via CLI args
    if len(sys.argv) > 1:
        instruction_file = sys.argv[1]
    if len(sys.argv) > 2:
        training_data_dir = sys.argv[2]
    if len(sys.argv) > 3:
        max_training_docs = int(sys.argv[3])

    asyncio.run(seed_rag_index(instruction_file, training_data_dir, max_training_docs))
