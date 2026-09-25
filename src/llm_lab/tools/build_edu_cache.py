"""Build the tokenized cache for the continued-pretraining (`pretrain_edu`) phase and push
it to its OWN Kaggle Dataset, leaving the base run's cache untouched.

Made to run inside a Kaggle CPU notebook (kaggle_build_edu_cache.ipynb): CPU sessions
don't spend the GPU/TPU quota. Disk is the constraint there, so each source is downloaded,
split and tokenized straight into train.bin, then its raw text is deleted before the next
one: peak ~= the largest raw source (~5GB) + train.bin (~4GB).

    python -m llm_lab.tools.build_edu_cache --work /tmp/edu            # build only
    python -m llm_lab.tools.build_edu_cache --work /tmp/edu --push     # build + upload

Output Dataset (default <user>/aion-pretrain-edu-tokenized):
  train.bin       new mix, uint16 ids, tokenized with the base's FROZEN tokenizer
  val.bin         held-out slice of the new mix (drives best.pt/early stop in the phase)
  val_old.bin     the base run's val.bin, unchanged: a regression check on the old data
  tokenizer.json  the base's tokenizer (changing it would force a restart from scratch)

Split rule is the base run's (md5(line)[-1] < 13 is val), so no line the base scored as
val lands in train. Hashed lines beyond each source's val quota are dropped, not trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# (preset key, download function name) in build order.
SOURCES = [
    ("fineweb_edu_mb", "download_fineweb_edu"),
    ("cosmopedia_mb", "download_cosmopedia"),
    ("wikipedia_mb", "download_wikipedia"),
    ("slimpajama_mb", "download_slimpajama_subset"),
]
BATCH_BYTES = 4 * 1024 * 1024  # ~4MB of text per encode_batch call


def _kaggle_username() -> str:
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    kj = Path.home() / ".kaggle" / "kaggle.json"
    if kj.exists():
        return json.loads(kj.read_text())["username"]
    sys.exit("No Kaggle username — set KAGGLE_USERNAME or ~/.kaggle/kaggle.json")


def _is_val(line: str) -> bool:
    return hashlib.md5(line.encode()).digest()[-1] < 13


def _tokenize_source(txt: Path, tok, train_out, val_out, val_quota: int) -> tuple[int, int, int]:
    """Append txt's train lines to train_out and up to val_quota bytes of val lines to
    val_out. Returns (train tokens, val tokens, dropped val lines)."""
    import numpy as np

    n_train = n_val = n_drop = 0
    val_bytes = 0
    buf: list[str] = []
    buf_bytes = 0
    val_buf: list[str] = []

    def _flush(lines, out) -> int:
        n = 0
        for enc in tok.encode_batch(lines):
            out.write(np.asarray(enc.ids, dtype=np.uint16).tobytes())
            n += len(enc.ids)
        return n

    with open(txt, "r", encoding="utf-8") as f:
        for line in f:
            if _is_val(line):
                if val_bytes < val_quota:
                    val_buf.append(line)
                    val_bytes += len(line)
                else:
                    n_drop += 1
                continue
            buf.append(line)
            buf_bytes += len(line)
            if buf_bytes >= BATCH_BYTES:
                n_train += _flush(buf, train_out)
                buf.clear()
                buf_bytes = 0
    if buf:
        n_train += _flush(buf, train_out)
    if val_buf:
        n_val += _flush(val_buf, val_out)
    return n_train, n_val, n_drop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="scratch dir (~10GB free)")
    ap.add_argument("--preset", default="pretrain_edu", help="raw corpus preset (data/download.py)")
    ap.add_argument("--src-dataset", default=None,
                    help="Dataset holding the base's tokenizer.json + val.bin "
                         "(default: <user>/aion-pretrain-tokenized). Only those two files are read.")
    ap.add_argument("--dataset", default=None,
                    help="Dataset to push the new cache to (default: <user>/aion-pretrain-edu-tokenized)")
    ap.add_argument("--val-mb", type=int, default=24, help="held-out text for the new val.bin, split by source share")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every source size (0.001 = smoke test)")
    ap.add_argument("--push", action="store_true", help="upload the cache to Kaggle")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    from llm_lab.data import download
    from llm_lab.kaggle_io import download_dataset

    user = _kaggle_username()
    src = args.src_dataset or f"{user}/aion-pretrain-tokenized"
    dst = args.dataset or f"{user}/aion-pretrain-edu-tokenized"
    if src == dst:
        sys.exit("--dataset must differ from --src-dataset: the base's cache stays untouched.")
    preset = download.PRESETS[args.preset]
    sources = [(key, fn, int(preset[key] * args.scale) or 1) for key, fn in SOURCES if preset.get(key, 0) > 0]
    total_mb = sum(mb for _, _, mb in sources)

    work = Path(args.work)
    frozen, raw, out = work / "frozen", work / "raw", work / "out"
    for d in (frozen, raw, out):
        d.mkdir(parents=True, exist_ok=True)
    plan = ", ".join(f"{k[:-3]} {mb}MB" for k, _, mb in sources)
    print(f"== {args.preset}: {plan} | free disk {shutil.disk_usage(work).free / 1e9:.0f} GB")

    # 1. Frozen tokenizer + old val from the base's cache (not its 10GB train.bin).
    print(f"== pulling tokenizer.json + val.bin from {src}")
    if download_dataset(src, frozen, required=("tokenizer.json", "val.bin"),
                        only=("tokenizer.json", "val.bin")) is None:
        sys.exit(f"{src} not found.")
    tok = Tokenizer.from_file(str(frozen / "tokenizer.json"))
    if tok.get_vocab_size() > 65536:
        sys.exit("Tokenizer vocab doesn't fit uint16.")

    # 2. Per source: download -> split + tokenize into the shared bins -> delete raw.
    stats = {}
    with open(out / "train.bin", "wb") as train_out, open(out / "val.bin", "wb") as val_out:
        for key, fn_name, mb in sources:
            txt = getattr(download, fn_name)(raw, target_mb=mb)
            quota = int(args.val_mb * 1024 * 1024 * mb / total_mb)
            print(f"== tokenizing {txt.name} (val quota {quota / 1e6:.1f} MB)")
            n_train, n_val, n_drop = _tokenize_source(txt, tok, train_out, val_out, quota)
            stats[key[:-3]] = {"raw_mb": round(txt.stat().st_size / 2**20),
                                             "train_tokens": n_train, "val_tokens": n_val}
            print(f"   train {n_train / 1e6:,.1f}M tok | val {n_val / 1e6:,.2f}M tok | dropped {n_drop:,} val-hashed lines")
            txt.unlink()  # reclaim disk before the next source

    shutil.copy2(frozen / "val.bin", out / "val_old.bin")
    shutil.copy2(frozen / "tokenizer.json", out / "tokenizer.json")
    total = sum(s["train_tokens"] for s in stats.values())
    for name, s in stats.items():
        s["share"] = round(s["train_tokens"] / max(1, total), 3)
    manifest = {"preset": args.preset, "scale": args.scale, "src_dataset": src,
                "train_tokens": total, "sources": stats}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"== cache ready in {out}: {sum(p.stat().st_size for p in out.iterdir()) / 1e9:.2f} GB, "
          f"{total / 1e9:.2f}B train tokens")

    if not args.push:
        print("Build only. Re-run with --push to upload.")
        return

    # 3. Push: create the Dataset on the first run, add a version afterwards.
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "AION Pretrain Edu Tokenized", "id": dst, "licenses": [{"name": "CC0-1.0"}]}))
    exists = subprocess.run(["kaggle", "datasets", "files", dst], capture_output=True).returncode == 0
    cmd = (["kaggle", "datasets", "version", "-p", str(out), "-m", f"{args.preset} x{args.scale}"]
           if exists else ["kaggle", "datasets", "create", "-p", str(out)])
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Pushed to {dst}.")


if __name__ == "__main__":
    main()
