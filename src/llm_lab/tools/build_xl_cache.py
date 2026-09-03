"""Build the enlarged `pretrain_xl` tokenized cache OFF-Kaggle, then push it to the
Kaggle Dataset the pretrain notebooks pull from — so Kaggle only ever downloads the
~10GB `.bin` files (which fit its disk) instead of tokenizing the ~20GB raw on-device
(which blows Kaggle's disk, the failure we hit).

It FREEZES val.bin + tokenizer.json (reused from the current cache) and drops val-hashed
lines from train, so val_loss stays directly comparable to the 235M base's 4.006.

Run on a machine with ~40GB free disk + your Kaggle creds:

    python -m llm_lab.tools.build_xl_cache --work /path/with/40GB
    python -m llm_lab.tools.build_xl_cache --work /path/with/40GB --push   # also upload

Steps: pull frozen val.bin+tokenizer -> download pretrain_xl raw -> split (drop val-hashed
lines) -> tokenize train with the frozen tokenizer -> (optional) push to the Kaggle Dataset.
Then on Kaggle run the pretrain notebook with ADD_DATA=False: it pulls the enlarged cache.
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


def _kaggle_username() -> str:
    u = os.environ.get("KAGGLE_USERNAME")
    if u:
        return u
    kj = Path.home() / ".kaggle" / "kaggle.json"
    if kj.exists():
        return json.loads(kj.read_text())["username"]
    sys.exit("No Kaggle username — set KAGGLE_USERNAME or ~/.kaggle/kaggle.json")


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True, help="scratch dir with ~40GB free")
    ap.add_argument("--preset", default="pretrain_xl", help="raw corpus preset")
    ap.add_argument("--dataset", default=None,
                    help="Kaggle dataset slug for the cache (default: <user>/aion-pretrain-tokenized)")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--push", action="store_true", help="upload the rebuilt cache to Kaggle")
    args = ap.parse_args()

    user = _kaggle_username()
    slug = args.dataset or f"{user}/aion-pretrain-tokenized"
    work = Path(args.work)
    cache, raw, build, push = (work / d for d in ("cache", "raw", "build", "push"))
    for d in (cache, raw, build):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Pull the CURRENT cache to reuse its frozen val.bin + tokenizer.json.
    print("== pulling frozen val.bin + tokenizer from", slug)
    _run(["kaggle", "datasets", "download", "-d", slug, "-p", str(cache), "--unzip"])
    val_bin = cache / "val.bin"
    tok_json = cache / "tokenizer.json"
    for p in (val_bin, tok_json):
        if not p.exists():
            sys.exit(f"Frozen {p.name} missing from {slug} — cannot keep val comparable.")

    # 2. Download the enlarged raw corpus locally.
    print("== downloading raw corpus preset", args.preset, "(~20GB, one-time)")
    _run([sys.executable, "-m", "llm_lab.cli", "download",
          "--target", str(raw), "--preset", args.preset])

    # 3. Split raw -> train.txt, DROPPING val-hashed lines (same rule as the notebook:
    #    md5(line)[-1] < 13 is val). val stays frozen; no val text leaks into train.
    train_txt = build / "train.txt"
    print("== building train.txt (val-hashed lines dropped)")
    n_train = n_drop = 0
    with open(train_txt, "w", encoding="utf-8") as out:
        for txt in sorted(raw.glob("*.txt")):
            print("   +", txt.name)
            with open(txt, "r", encoding="utf-8") as f:
                for line in f:
                    if hashlib.md5(line.encode()).digest()[-1] < 13:
                        n_drop += 1
                    else:
                        out.write(line)
                        n_train += 1
    print(f"   train lines {n_train:,} | dropped (val-hashed) {n_drop:,}")

    # 4. Tokenize train.txt -> train.bin with the FROZEN tokenizer. Inlined (mirrors
    #    data/dataset.py) so the builder needs only tokenizers+numpy, not torch.
    import numpy as np
    from tokenizers import Tokenizer
    shutil.copy2(tok_json, build / "tokenizer.json")
    print("== tokenizing train.txt -> train.bin")
    tok = Tokenizer.from_file(str(tok_json))
    train_bin = build / "train.bin"
    BATCH_BYTES = 4 * 1024 * 1024  # ~4MB of text per encode_batch call

    def _flush(lines, out):
        for enc in tok.encode_batch(lines):
            out.write(np.asarray(enc.ids, dtype=np.uint16).tobytes())

    buf: list[str] = []
    buf_bytes = 0
    with open(train_txt, "r", encoding="utf-8") as f, open(train_bin, "wb") as out:
        for line in f:
            buf.append(line)
            buf_bytes += len(line)
            if buf_bytes >= BATCH_BYTES:
                _flush(buf, out)
                buf.clear()
                buf_bytes = 0
        if buf:
            _flush(buf, out)
    train_txt.unlink()  # reclaim disk; keep only the .bin

    # 5. Assemble the cache dir (frozen val + tokenizer + new train).
    if push.exists():
        shutil.rmtree(push)
    push.mkdir(parents=True)
    shutil.copy2(train_bin, push / "train.bin")
    shutil.copy2(val_bin, push / "val.bin")
    shutil.copy2(tok_json, push / "tokenizer.json")
    (push / "dataset-metadata.json").write_text(json.dumps({
        "title": "AION Pretrain Tokenized Data",
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }))
    sz = sum(p.stat().st_size for p in push.iterdir()) / 1e9
    print(f"== cache ready in {push} ({sz:.1f} GB): "
          f"train.bin {train_bin.stat().st_size/1e9:.1f}GB + frozen val.bin + tokenizer.json")

    if not args.push:
        print("Dry build done. Re-run with --push to upload to Kaggle, or upload push/ manually.")
        return

    # 6. Push a new version of the Kaggle Dataset the notebooks pull.
    print("== pushing new dataset version to", slug)
    _run(["kaggle", "datasets", "version", "-p", str(push),
          "-m", f"pretrain_xl train (frozen val/tokenizer) via {args.preset}", "--dir-mode", "zip"])
    print("Done. On Kaggle: run the pretrain notebook with MODEL_SIZE='large', ADD_DATA=False —")
    print("it will pull this enlarged cache (bins only, fits disk) and train on the new data.")


if __name__ == "__main__":
    main()
