"""Instruction tuning dataset and formatting."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


INSTRUCTION_TEMPLATE = """Q: {instruction}
A: {response}<|end|>"""

CHAT_TEMPLATE_USER = "<|user|>"
CHAT_TEMPLATE_ASSISTANT = "<|assistant|>"
CHAT_TEMPLATE_END = "<|end|>"


def format_example(instruction: str, context: str = "", response: str = "") -> str:
    """Format a single instruction example with minimal template."""
    return INSTRUCTION_TEMPLATE.format(
        instruction=instruction,
        response=response,
    )


class InstructionDataset(Dataset):
    """Dataset for instruction tuning.

    Each example is a JSON object with fields:
        - instruction: the user's request
        - context: optional additional context
        - response: the expected output
    """

    def __init__(self, path: Path, tokenizer: Tokenizer, max_len: int = 256):
        data = json.loads(path.read_text())
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.max_len = max_len

        # Encode the response marker to find where response starts
        response_marker_ids = tokenizer.encode("\nA: ").ids

        for item in data:
            text = format_example(
                instruction=item["instruction"],
                context=item.get("context", ""),
                response=item["response"],
            )
            encoded = tokenizer.encode(text)
            ids = encoded.ids[:max_len]
            if len(ids) >= 8:  # skip too-short examples
                ids_tensor = torch.tensor(ids, dtype=torch.long)

                # Find where <|response|>\n ends — only compute loss AFTER that
                resp_start = -1
                marker_len = len(response_marker_ids)
                for i in range(len(ids) - marker_len + 1):
                    if ids[i:i + marker_len] == response_marker_ids:
                        resp_start = i + marker_len
                        break

                # Build labels: -100 for instruction/context, real ids for response
                labels = torch.full_like(ids_tensor, -100)
                if resp_start > 0:
                    labels[resp_start:] = ids_tensor[resp_start:]
                else:
                    labels[:] = ids_tensor  # fallback: loss on everything

                self.examples.append((ids_tensor, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids, labels = self.examples[idx]
        return {
            "input_ids": ids[:-1],
            "labels": labels[1:],  # shifted labels, -100 masks instruction tokens
        }


def collate_instruction(batch: list[dict], pad_to: int | None = None) -> dict[str, torch.Tensor]:
    """Pad sequences to same length within a batch.

    When pad_to is set, every batch is padded to that fixed length so XLA/TPU sees a
    constant input shape and compiles the graph once. Dynamic per-batch lengths would
    otherwise trigger a full recompile on nearly every step (catastrophically slow).
    """
    batch_max = max(item["input_ids"].size(0) for item in batch)
    max_len = pad_to if pad_to is not None else batch_max
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)  # -100 = ignore

    for i, item in enumerate(batch):
        L = min(item["input_ids"].size(0), max_len)
        input_ids[i, :L] = item["input_ids"][:L]
        labels[i, :L] = item["labels"][:L]

    return {"input_ids": input_ids, "labels": labels}


class MultiTurnDataset(Dataset):
    """Dataset for multi-turn chat fine-tuning.

    Each example is a JSON object with a 'messages' list:
        [{"role": "user"|"assistant", "content": "..."},  ...]

    Format applied:
        <|user|>Hello<|end|>
        <|assistant|>Hi! How can I help?<|end|>
        ...

    Loss is computed ONLY on assistant tokens; user turns are masked with -100.
    """

    def __init__(self, path: Path, tokenizer: Tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Pre-encode special tokens once
        self._user_ids = tokenizer.encode(CHAT_TEMPLATE_USER).ids
        self._asst_ids = tokenizer.encode(CHAT_TEMPLATE_ASSISTANT).ids
        self._end_ids = tokenizer.encode(CHAT_TEMPLATE_END).ids
        self._nl_ids = tokenizer.encode("\n").ids

        # Only load raw messages — no tokenization yet
        data = json.loads(path.read_text())
        self.examples: list[list[dict]] = [
            item["messages"]
            for item in data
            if len(item.get("messages", [])) >= 2
        ]

    def _tokenize(self, messages: list[dict]) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Tokenize a single conversation on demand."""
        all_ids: list[int] = []
        is_response: list[bool] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()
            if not content:
                continue

            if role == "user":
                prefix = self._user_ids
                in_response = False
            else:
                prefix = self._asst_ids
                in_response = True

            content_ids = self.tokenizer.encode(content).ids
            turn_ids = prefix + content_ids + self._end_ids + self._nl_ids

            all_ids.extend(turn_ids)
            is_response.extend([in_response] * len(turn_ids))

        all_ids = all_ids[:self.max_len]
        is_response = is_response[:self.max_len]

        if len(all_ids) < 8:
            return None

        ids_tensor = torch.tensor(all_ids, dtype=torch.long)
        labels = torch.full_like(ids_tensor, -100)
        for i, resp in enumerate(is_response):
            if resp:
                labels[i] = ids_tensor[i]

        if (labels != -100).sum() == 0:
            return None

        return ids_tensor, labels

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        result = self._tokenize(self.examples[idx])
        if result is None:
            # Return a minimal dummy example — collate will handle it gracefully
            dummy = torch.zeros(8, dtype=torch.long)
            return {"input_ids": dummy[:-1], "labels": torch.full((7,), -100, dtype=torch.long)}
        ids, labels = result
        return {
            "input_ids": ids[:-1],
            "labels": labels[1:],
        }


# collate_instruction works for MultiTurnDataset too (same output dict shape)
collate_chat = collate_instruction
