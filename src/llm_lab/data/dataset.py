"""Text dataset for language modeling."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class TextDataset(Dataset):
    """Tokenized text dataset for causal LM training.

    On first load, tokenizes the text and saves a .bin cache.
    Subsequent loads use the cache directly (mmap for zero-copy reads).
    """

    def __init__(self, path: Path, tokenizer: Tokenizer, seq_len: int = 256):
        self.seq_len = seq_len
        cache_path = path.with_suffix(".bin")

        # Use the token cache if it exists and is at least as new as the source text.
        # When only the cache is present (e.g. restored from a data Dataset, source
        # text not downloaded), treat the cache as authoritative.
        cache_ok = cache_path.exists() and (
            not path.exists() or cache_path.stat().st_mtime >= path.stat().st_mtime
        )
        if cache_ok:
            # Load from binary cache (mmap = no RAM copy)
            self.ids = np.memmap(cache_path, dtype=np.uint16, mode="r")
        else:
            if not path.exists():
                raise FileNotFoundError(
                    f"No text file or token cache found for {path} (looked for {cache_path})"
                )
            # Tokenize with the Rust tokenizer's multithreaded batch encoder (uses all
            # CPU cores) and stream IDs straight to the cache. Batches are bounded by
            # *bytes* of text (not line count), so long documents can't spike memory and
            # OOM the kernel. GPU/TPU can't help: BPE is a string algorithm, not tensor math.
            BATCH_BYTES = 4 * 1024 * 1024  # ~4 MB of text per encode_batch call
            tmp_path = cache_path.with_suffix(".bin.tmp")

            def _encode_write(lines: list[str], out) -> None:
                for enc in tokenizer.encode_batch(lines):
                    out.write(np.asarray(enc.ids, dtype=np.uint16).tobytes())

            buf: list[str] = []
            buf_bytes = 0
            with open(path, "r", encoding="utf-8") as f, open(tmp_path, "wb") as out:
                for line in f:
                    buf.append(line)
                    buf_bytes += len(line)
                    if buf_bytes >= BATCH_BYTES:
                        _encode_write(buf, out)
                        buf.clear()
                        buf_bytes = 0
                if buf:
                    _encode_write(buf, out)
            os.replace(tmp_path, cache_path)
            self.ids = np.memmap(cache_path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = torch.from_numpy(self.ids[idx : idx + self.seq_len + 1].astype(np.int64))
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }
