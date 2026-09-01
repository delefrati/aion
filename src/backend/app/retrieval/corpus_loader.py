"""Training corpus loader for RAG seeding."""
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import AsyncIterator

from app.retrieval.base import Document

logger = logging.getLogger("aion.corpus")


class CorpusLoader:
    """Load and yield documents from training corpus."""

    @staticmethod
    async def load_instruction_dataset(
        instruction_json_path: str,
    ) -> AsyncIterator[Document]:
        """Load instruction/response pairs from instruction_seed.json.

        Yields Document objects with (instruction + response) as content.

        Args:
            instruction_json_path: Path to instruction_seed.json
        """
        path = Path(instruction_json_path)
        if not path.exists():
            logger.warning("Instruction file not found: %s", instruction_json_path)
            return

        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed to load instruction dataset: %s", e)
            return

        for i, item in enumerate(data):
            instruction = item.get("instruction", "")
            response = item.get("response", "")
            context = item.get("context", "")

            # Combine context + instruction + response for indexing
            parts = [p for p in [context, instruction, response] if p]
            content = "\n".join(parts)

            yield Document(
                content=content,
                source=f"instruction_{i}",
            )

    @staticmethod
    async def load_training_corpus_sample(
        train_data_dir: str,
        max_documents: int = 100,
        max_chars_per_doc: int = 500,
    ) -> AsyncIterator[Document]:
        """Load sample documents from training corpus (SlimPajama/Wikipedia).

        Note: This is a simple implementation that reads text files.
        For production, use a more efficient streaming approach.

        Args:
            train_data_dir: Directory containing training data files
            max_documents: Maximum number of documents to load
            max_chars_per_doc: Truncate documents to this length
        """
        data_path = Path(train_data_dir)
        if not data_path.exists():
            logger.warning("Training data directory not found: %s", train_data_dir)
            return

        doc_count = 0
        
        # Look for common training data formats
        for pattern in ["*.txt", "*.jsonl", "train.bin", "val.bin"]:
            if doc_count >= max_documents:
                break

            for file_path in sorted(data_path.glob(pattern)):
                if doc_count >= max_documents:
                    break

                if file_path.name.endswith(".bin"):
                    logger.debug("Skipping binary file: %s", file_path)
                    continue

                try:
                    if file_path.suffix == ".jsonl":
                        with open(file_path) as f:
                            for line_num, line in enumerate(f):
                                if doc_count >= max_documents:
                                    break
                                try:
                                    obj = json.loads(line.strip())
                                    text = obj.get("text", "")
                                    if text:
                                        yield Document(
                                            content=text[:max_chars_per_doc],
                                            source=f"training_{file_path.stem}_{line_num}",
                                        )
                                        doc_count += 1
                                except json.JSONDecodeError:
                                    continue
                    else:
                        # Text file
                        with open(file_path, encoding="utf-8") as f:
                            chunk = f.read(max_chars_per_doc)
                            if chunk:
                                yield Document(
                                    content=chunk,
                                    source=f"training_{file_path.stem}",
                                )
                                doc_count += 1
                except Exception as e:
                    logger.warning("Failed to load %s: %s", file_path, e)
                    continue
