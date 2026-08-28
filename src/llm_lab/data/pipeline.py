"""Dataset pipeline: ingest → clean → dedup → split → manifest."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from tqdm import tqdm


def ingest_raw_texts(src_dir: Path, out_dir: Path) -> list[Path]:
    """Read .txt files from src_dir, write cleaned lines to out_dir/cleaned/."""
    cleaned_dir = out_dir / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for p in sorted(src_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="replace")
        text = _normalize(text)
        dest = cleaned_dir / p.name
        dest.write_text(text, encoding="utf-8")
        results.append(dest)
    return results


def dedup_lines(files: list[Path], out_path: Path) -> int:
    """Deduplicate across all files by line hash. Returns count of unique lines."""
    seen: set[str] = set()
    unique_lines: list[str] = []

    for f in tqdm(files, desc="dedup"):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            h = hashlib.md5(line.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique_lines.append(line)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(unique_lines) + "\n", encoding="utf-8")
    return len(unique_lines)


def train_val_split(
    corpus_path: Path,
    out_dir: Path,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> dict:
    """Split corpus into train/val. Returns manifest dict."""
    import random

    lines = corpus_path.read_text(encoding="utf-8").splitlines()
    rng = random.Random(seed)
    rng.shuffle(lines)

    split_idx = max(1, int(len(lines) * (1 - val_ratio)))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"
    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    manifest = {
        "total_lines": len(lines),
        "train_lines": len(train_lines),
        "val_lines": len(val_lines),
        "val_ratio": val_ratio,
        "seed": seed,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _normalize(text: str) -> str:
    """Unicode normalize, collapse whitespace, strip control chars."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\S\n]+", " ", text)  # collapse spaces but keep newlines
    text = re.sub(r"\n{3,}", "\n\n", text)  # max 2 consecutive newlines
    # strip control chars except newline and tab
    text = "".join(c for c in text if c in ("\n", "\t") or not unicodedata.category(c).startswith("C"))
    return text.strip() + "\n"
