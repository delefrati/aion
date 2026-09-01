"""Download datasets from HuggingFace for AION training.

Usage (inside lab container):
    python -m llm_lab.data.download --target /app/data/raw --preset medium

Presets:
    medium: ~80MB SlimPajama subset + dolly-15k + no_robots
    small:  ~20MB SlimPajama subset + dolly-15k only
    chat:   oasst1 best-path conversations + dolly-15k; no large pretraining corpus
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def _slimpajama_set_name(example) -> str | None:
    """Extract the RedPajama source subset name from a SlimPajama row's meta field."""
    meta = example.get("meta")
    if isinstance(meta, dict):
        return meta.get("redpajama_set_name")
    if isinstance(meta, str):
        try:
            return json.loads(meta).get("redpajama_set_name")
        except Exception:
            return None
    return None


def _is_prose(text: str) -> bool:
    """Reject LaTeX/code-heavy docs that make the model emit pseudocode and math."""
    if len(text) < 100:
        return False
    sample = text[:2000]
    # Prose almost never has many backslashes / braces / dollar signs; LaTeX and code do.
    if sum(sample.count(c) for c in ("\\", "{", "}", "$")) / len(sample) > 0.02:
        return False
    if any(m in sample for m in ("\\begin{", "\\frac", "\\sum", "\\int", "#include", "public static void")):
        return False
    return True


# Keep prose-heavy SlimPajama subsets; drop Github (code) and ArXiv (LaTeX) which
# otherwise teach the model to emit source code and math markup.
_SLIMPAJAMA_KEEP_SETS = {
    "RedPajamaCommonCrawl", "RedPajamaC4", "RedPajamaBook",
    "RedPajamaWikipedia", "RedPajamaStackExchange",
}


def download_slimpajama_subset(out_dir: Path, target_mb: int = 80) -> Path:
    """Download a prose-filtered subset of SlimPajama from HuggingFace (streaming)."""
    from datasets import load_dataset

    out_path = out_dir / "slimpajama_subset.txt"
    if out_path.exists():
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"SlimPajama subset already exists ({size_mb:.1f} MB), skipping")
        return out_path

    print(f"Downloading SlimPajama subset (~{target_mb} MB, prose-filtered)...")
    target_bytes = target_mb * 1024 * 1024
    total_bytes = 0
    kept = skipped = 0

    ds = load_dataset(
        "DKYoon/SlimPajama-6B",
        split="train",
        streaming=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for example in tqdm(ds, desc="SlimPajama"):
            # Drop code/math subsets when the source subset is known.
            set_name = _slimpajama_set_name(example)
            if set_name is not None and set_name not in _SLIMPAJAMA_KEEP_SETS:
                skipped += 1
                continue
            text = example["text"].strip()
            if not _is_prose(text):  # heuristic fallback (schema-independent)
                skipped += 1
                continue
            f.write(text + "\n\n")
            kept += 1
            total_bytes += len(text.encode("utf-8"))
            if total_bytes >= target_bytes:
                break

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Saved: {out_path} ({size_mb:.1f} MB) — kept {kept:,}, skipped {skipped:,} code/math docs")
    return out_path


def download_dolly15k(out_dir: Path) -> Path:
    """Download databricks-dolly-15k instruction dataset."""
    from datasets import load_dataset

    out_path = out_dir / "dolly_15k.json"
    if out_path.exists():
        print("dolly-15k already exists, skipping")
        return out_path

    print("Downloading databricks-dolly-15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")

    examples = []
    for row in tqdm(ds, desc="dolly-15k"):
        examples.append({
            "instruction": row["instruction"],
            "context": row.get("context", ""),
            "response": row["response"],
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(examples, ensure_ascii=False, indent=None), encoding="utf-8")
    print(f"Saved: {out_path} ({len(examples)} examples)")
    return out_path


def download_no_robots(out_dir: Path) -> Path:
    """Download HuggingFaceH4/no_robots instruction dataset."""
    from datasets import load_dataset

    out_path = out_dir / "no_robots.json"
    if out_path.exists():
        print("no_robots already exists, skipping")
        return out_path

    print("Downloading no_robots (10k examples)...")
    ds = load_dataset("HuggingFaceH4/no_robots", split="train")

    examples = []
    for row in tqdm(ds, desc="no_robots"):
        # no_robots has 'messages' format
        messages = row["messages"]
        # Extract instruction (first user message) and response (first assistant message)
        instruction = ""
        response = ""
        for msg in messages:
            if msg["role"] == "user" and not instruction:
                instruction = msg["content"]
            elif msg["role"] == "assistant" and not response:
                response = msg["content"]
        if instruction and response:
            examples.append({
                "instruction": instruction,
                "context": "",
                "response": response,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(examples, ensure_ascii=False, indent=None), encoding="utf-8")
    print(f"Saved: {out_path} ({len(examples)} examples)")
    return out_path


def download_openassistant_guanaco(out_dir: Path) -> Path:
    """Download timdettmers/openassistant-guanaco multi-turn chat dataset (~9.8k conversations)."""
    from datasets import load_dataset

    out_path = out_dir / "openassistant_guanaco.json"
    if out_path.exists():
        print("openassistant-guanaco already exists, skipping")
        return out_path

    print("Downloading openassistant-guanaco (~9.8k multi-turn conversations)...")
    ds = load_dataset("timdettmers/openassistant-guanaco", split="train")

    examples = []
    for row in tqdm(ds, desc="guanaco"):
        # Each row has a single "text" field with the full conversation formatted as
        # "### Human:\n...\n### Assistant:\n..." blocks — parse into messages list
        messages = _parse_guanaco_text(row["text"])
        if len(messages) >= 2:
            examples.append({"messages": messages})

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(examples, ensure_ascii=False, indent=None), encoding="utf-8")
    print(f"Saved: {out_path} ({len(examples)} conversations)")
    return out_path


def _parse_guanaco_text(text: str) -> list[dict]:
    """Parse guanaco conversation text into [{role, content}, ...] messages."""
    import re
    messages = []
    # Split on role markers; guanaco uses "### Human:" and "### Assistant:"
    parts = re.split(r"###\s*(Human|Assistant)\s*:\s*", text)
    # parts[0] is empty or preamble; then alternating role/content pairs
    i = 1
    while i + 1 < len(parts):
        role_raw = parts[i].strip()
        content = parts[i + 1].strip()
        if content:
            role = "user" if role_raw == "Human" else "assistant"
            messages.append({"role": role, "content": content})
        i += 2
    return messages


def download_oasst1(out_dir: Path) -> Path:
    """Download OpenAssistant/oasst1 and extract the highest-scored reply chain per tree.

    Produces ~9k high-quality multi-turn conversations (English only).
    Output format: [{"messages": [{"role": "user"|"assistant", "content": "..."}]}]
    """
    from datasets import load_dataset

    out_path = out_dir / "oasst1.json"
    if out_path.exists():
        print("oasst1 already exists, skipping")
        return out_path

    print("Downloading OpenAssistant/oasst1...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")

    # Index all messages by message_id
    by_id: dict[str, dict] = {}
    children: dict[str, list[str]] = {}
    roots: list[str] = []

    for row in ds:
        mid = row["message_id"]
        by_id[mid] = row
        parent = row.get("parent_id")
        if parent:
            children.setdefault(parent, []).append(mid)
        else:
            roots.append(mid)

    def best_child(mid: str) -> str | None:
        """Return the child with the highest rank (rank=0 is best)."""
        kids = children.get(mid, [])
        if not kids:
            return None
        return min(kids, key=lambda k: by_id[k].get("rank") or 999)

    def extract_chain(root_id: str) -> list[dict] | None:
        """Walk the best-path chain from root to leaf, filtering low-quality assistant messages."""
        msgs = []
        mid = root_id
        while mid:
            row = by_id[mid]
            if row.get("lang", "en") != "en":
                return None  # skip non-English trees
            role = "user" if row["role"] == "prompter" else "assistant"
            content = row["text"].strip()
            # Skip trees with negatively-rated assistant messages
            if role == "assistant":
                score = row.get("score")
                if score is not None and score < 0:
                    return None
            if content:
                msgs.append({"role": role, "content": content})
            mid = best_child(mid)
        # Require at least 2 full turns (4 messages: user, assistant, user, assistant)
        if len(msgs) < 4:
            return None
        return msgs

    examples = []
    for root_id in tqdm(roots, desc="oasst1 trees"):
        chain = extract_chain(root_id)
        if chain:
            examples.append({"messages": chain})

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path} ({len(examples)} conversations)")
    return out_path


def download_wikipedia(out_dir: Path, target_mb: int = 500) -> Path:
    """Download a subset of Wikipedia articles (English, streaming)."""
    from datasets import load_dataset

    out_path = out_dir / "wikipedia_subset.txt"
    if out_path.exists():
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"Wikipedia subset already exists ({size_mb:.1f} MB), skipping")
        return out_path

    print(f"Downloading Wikipedia subset (~{target_mb} MB)...")
    target_bytes = target_mb * 1024 * 1024
    total_bytes = 0

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for example in tqdm(ds, desc="Wikipedia"):
            text = example["text"].strip()
            if len(text) < 200:  # skip stubs
                continue
            f.write(text + "\n\n")
            total_bytes += len(text.encode("utf-8"))
            if total_bytes >= target_bytes:
                break

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")
    return out_path


def download_hh_rlhf(out_dir: Path) -> Path:
    from datasets import load_dataset

    out_path = out_dir / "hh_rlhf.json"
    if out_path.exists():
        print("hh-rlhf already exists, skipping")
        return out_path

    print("Downloading Anthropic/hh-rlhf (streaming ~160k conversations)...")
    ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)

    examples = []
    for row in tqdm(ds, desc="hh-rlhf"):
        # Use the 'chosen' response (preferred by humans over 'rejected')
        messages = _parse_hh_rlhf(row["chosen"])
        if len(messages) >= 2:
            examples.append({"messages": messages})

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(examples, ensure_ascii=False, indent=None), encoding="utf-8")
    print(f"Saved: {out_path} ({len(examples)} conversations)")
    return out_path


def download_ultrachat(out_dir: Path, max_examples: int = 50000) -> Path:
    """Download HuggingFaceH4/ultrachat_200k (train_sft) multi-turn conversations.

    Rows already carry a 'messages' list of {role, content}. Streamed and capped to
    max_examples to keep on-disk size and merge time manageable.
    Output format: [{"messages": [{"role": "user"|"assistant", "content": "..."}]}]
    """
    from datasets import load_dataset

    out_path = out_dir / "ultrachat.json"
    if out_path.exists():
        print("ultrachat already exists, skipping")
        return out_path

    print(f"Downloading HuggingFaceH4/ultrachat_200k (streaming, up to {max_examples} conversations)...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)

    examples = []
    for row in tqdm(ds, desc="ultrachat"):
        msgs = [
            {"role": m["role"], "content": m["content"].strip()}
            for m in row.get("messages", [])
            if m.get("content", "").strip() and m.get("role") in ("user", "assistant")
        ]
        if len(msgs) >= 2:
            examples.append({"messages": msgs})
        if len(examples) >= max_examples:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path} ({len(examples)} conversations)")
    return out_path


def _parse_hh_rlhf(text: str) -> list[dict]:
    """Parse hh-rlhf 'Human: ...\n\nAssistant: ...' format into messages list."""
    import re
    messages = []
    # Normalize leading newlines then split on role markers
    parts = re.split(r"\n\n(Human|Assistant):\s*", "\n\n" + text.strip())
    i = 1
    while i + 1 < len(parts):
        role_raw = parts[i].strip()
        content = parts[i + 1].strip()
        if content:
            role = "user" if role_raw == "Human" else "assistant"
            messages.append({"role": role, "content": content})
        i += 2
    return messages


def merge_chat_datasets(raw_dir: Path, out_path: Path, seed_path: Path | None = None,
                        max_per_source: dict[str, int] | None = None) -> Path:
    """Merge all downloaded chat JSON files into a single chat_merged.json.

    Combines (in order of preference):
      - oasst1.json            — best-path multi-turn conversations
      - openassistant_guanaco.json — fallback if oasst1 not present
      - ultrachat.json         — HuggingFaceH4/ultrachat_200k multi-turn (if present)
      - hh_rlhf.json           — Anthropic multi-turn (if present)
      - dolly_15k.json         — converted from {instruction,response} to {messages}
      - seed_path              — AION identity / custom examples (instruction_seed.json)

    All examples are shuffled with a fixed seed and written as a flat list of
    {"messages": [...]} objects to out_path.
    """
    import random

    examples: list[dict] = []
    max_per_source = max_per_source or {}

    # Multi-turn chat files (already in {messages} format)
    for fname in ("oasst1.json", "openassistant_guanaco.json", "ultrachat.json", "hh_rlhf.json"):
        p = raw_dir / fname
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            cap = max_per_source.get(fname.replace('.json', ''))
            if cap and len(data) > cap:
                random.Random(42).shuffle(data)
                data = data[:cap]
                print(f"  {fname}: {len(data)} conversations (capped from original)")
            else:
                print(f"  {fname}: {len(data)} conversations")
            examples.extend(data)

    # Dolly — single-turn, convert to {messages}
    dolly_path = raw_dir / "dolly_15k.json"
    if dolly_path.exists():
        data = json.loads(dolly_path.read_text(encoding="utf-8"))
        converted = [
            {"messages": [
                {"role": "user", "content": (row.get("context", "").strip() + "\n\n" + row["instruction"]).strip()},
                {"role": "assistant", "content": row["response"]},
            ]}
            for row in data
            if row.get("instruction") and row.get("response")
        ]
        examples.extend(converted)
        print(f"  dolly_15k.json: {len(converted)} examples")

    # Custom seed (instruction_seed.json or expand_instructions output)
    if seed_path and seed_path.exists():
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        converted = []
        for row in data:
            instruction = row.get("instruction", "")
            response = row.get("response", "")
            # Handle both {instruction, response} and {messages} formats
            if instruction and response:
                converted.append({"messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response},
                ]})
            elif row.get("messages"):
                converted.append(row)
        examples.extend(converted)
        print(f"  {seed_path.name}: {len(converted)} examples")

    random.Random(42).shuffle(examples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    print(f"\nMerged {len(examples)} total examples -> {out_path}")
    return out_path


PRESETS = {
    "small": {"slimpajama_mb": 20, "dolly": True, "no_robots": False, "guanaco": False, "hh_rlhf": False},
    "medium": {"slimpajama_mb": 80, "dolly": True, "no_robots": True, "guanaco": False, "hh_rlhf": False},
    "large": {"slimpajama_mb": 200, "dolly": True, "no_robots": True, "guanaco": False, "hh_rlhf": False},
    "chat": {"slimpajama_mb": 0, "dolly": True, "no_robots": True, "guanaco": False, "oasst1": True, "ultrachat": True, "ultrachat_max": 50000, "hh_rlhf": True},
    # ~1000+8000 MB ≈ ~2.2B tokens — Chinchilla-optimal for the 110M base and matched to
    # the 40k-step continue budget (~2.6B tokens processed). Raising further only helps if
    # max_steps also goes past 40k, else the model won't see all the extra tokens.
    "pretrain": {"wikipedia_mb": 1000, "slimpajama_mb": 8000, "dolly": False, "no_robots": False, "guanaco": False, "hh_rlhf": False},
    # ~2000+18000 MB ≈ ~5B unique tokens — Chinchilla-optimal for the 235M base (~20 tok/param).
    # Use with the 235M run: removes the repetition the 8000 preset hits at 50k steps, and lets
    # you extend past 50k without re-seeing data. Fold it in via the notebook's ADD_DATA path so
    # val.bin + tokenizer stay frozen (losses stay comparable to the current run).
    "pretrain_xl": {"wikipedia_mb": 2000, "slimpajama_mb": 18000, "dolly": False, "no_robots": False, "guanaco": False, "hh_rlhf": False},
}


def main():
    parser = argparse.ArgumentParser(description="Download training datasets")
    parser.add_argument("--target", required=True, help="Output directory for raw data")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="medium")
    args = parser.parse_args()

    out_dir = Path(args.target)
    preset = PRESETS[args.preset]

    print(f"Preset: {args.preset}")
    print(f"Target dir: {out_dir}")
    print()

    # Pretraining data
    if preset.get("wikipedia_mb", 0) > 0:
        download_wikipedia(out_dir, target_mb=preset["wikipedia_mb"])
    if preset["slimpajama_mb"] > 0:
        download_slimpajama_subset(out_dir, target_mb=preset["slimpajama_mb"])

    # Instruction / chat data
    if preset.get("dolly", False):
        download_dolly15k(out_dir)
    if preset.get("no_robots", False):
        download_no_robots(out_dir)
    if preset.get("guanaco", False):
        download_openassistant_guanaco(out_dir)
    if preset.get("oasst1", False):
        download_oasst1(out_dir)
    if preset.get("ultrachat", False):
        download_ultrachat(out_dir, max_examples=preset.get("ultrachat_max", 50000))
    if preset.get("hh_rlhf", False):
        download_hh_rlhf(out_dir)

    print("\nDone! Next steps:")
    print(f"  1. Run: python -m llm_lab.cli prepare --src-dir {out_dir} --out-dir /app/data")
    print(f"  2. Re-train tokenizer if vocab needs updating")
    print(f"  3. Train: python -m llm_lab.cli train --config /app/llm_lab/configs/mamba_medium.yaml")


if __name__ == "__main__":
    main()
