"""Measure TPU throughput for several trainer settings, one short run each, nothing pushed.

Made for the Kaggle TPU notebook (kaggle_pretrain_tpu.ipynb, TPU_SWEEP=True): each variant
is a fresh `llm_lab.cli train` process (SPMD needs one per run) seeded with the same weights
at step 0. LR is 0, so the weights never move: the end-of-run val_loss must then agree across
variants to ~1e-2 (bf16 noise), which catches a variant that computes something different
(say, a mis-sharded flash kernel) as well as measuring its speed.

    python -m llm_lab.tools.tpu_sweep --config configs/transformer_tpu_large_edu_spmd.yaml \\
        --seed-ckpt /tmp/checkpoints/latest.pt --data-dir /tmp/data --work /tmp/sweep

Prints a table and writes the rows to --out (JSON).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

# (name, overrides) in run order: the proven setup first, so a later crash still leaves a
# baseline to compare against. batch_size is the GLOBAL micro-batch (8 chips).
VARIANTS = [
    ("baseline 2/chip x4", {}),
    ("flash 2/chip x4", {"flash_attention": True}),
    ("fsdp 8/chip x1", {"tpu_fsdp": True, "batch_size": 64, "grad_accum_steps": 1, "tpu_fuse_step": True}),
    ("flash+fsdp 8/chip x1", {"flash_attention": True, "tpu_fsdp": True, "batch_size": 64,
                              "grad_accum_steps": 1, "tpu_fuse_step": True}),
    ("flash+fsdp 8/chip x1 no-ckpt", {"flash_attention": True, "tpu_fsdp": True, "batch_size": 64,
                                      "grad_accum_steps": 1, "tpu_fuse_step": True,
                                      "grad_checkpoint": False}),
    ("flash+fsdp 16/chip x1", {"flash_attention": True, "tpu_fsdp": True, "batch_size": 128,
                               "grad_accum_steps": 1, "tpu_fuse_step": True}),
    ("flash+fsdp 16/chip x1 no-ckpt", {"flash_attention": True, "tpu_fsdp": True, "batch_size": 128,
                                       "grad_accum_steps": 1, "tpu_fuse_step": True,
                                       "grad_checkpoint": False}),
]

EVAL_WINDOWS = 512  # divisible by every batch_size above
_HBM = re.compile(r"hbm ([\d.]+)/([\d.]+)GB")


def _run(name: str, overrides: dict, base: dict, seed: Path, work: Path, steps: int) -> dict:
    d = work / re.sub(r"[^\w.-]+", "_", name)
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy2(seed, d / "latest.pt")
    batch = overrides.get("batch_size", base["batch_size"])
    cfg = {**base, **overrides,
           "lr": 0.0, "warmup_steps": 0, "lr_schedule": "cosine", "lr_decay_steps": 0,
           "max_steps": steps, "log_every": 10, "eval_every": steps, "checkpoint_every": 10**9,
           # The same EVAL_WINDOWS val windows for every batch size, so val_loss compares.
           "max_eval_batches": max(1, EVAL_WINDOWS // batch), "early_stop_patience": 0,
           "max_train_minutes": 0, "checkpoint_dir": str(d)}
    (d / "run.yaml").write_text(yaml.dump(cfg))

    print(f"\n===== {name}: {overrides or 'config as is'}", flush=True)
    t0 = time.time()
    hbm = []
    proc = subprocess.Popen([sys.executable, "-u", "-m", "llm_lab.cli", "train", "--config", str(d / "run.yaml")],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        if "| loss" in line or "val_loss" in line or "Error" in line or "error" in line:
            print(line, end="", flush=True)
        tail = (tail + [line])[-15:]
        m = _HBM.search(line)
        if m:
            hbm.append(float(m.group(1)))
    rc = proc.wait()
    row = {"name": name, "overrides": overrides, "rc": rc, "wall_s": round(time.time() - t0)}
    try:
        log = json.loads((d / "metrics.json").read_text())
    except Exception:
        log = []
    toks = [e["tok_s"] for e in log if "tok_s" in e][1:]  # the first window includes the compile
    vals = [e["val_loss"] for e in log if "val_loss" in e]
    losses = [e["train_loss"] for e in log if "train_loss" in e]
    row.update(tok_s=round(statistics.median(toks)) if toks else None,
               val_loss=vals[-1] if vals else None,
               first_loss=losses[0] if losses else None,
               hbm_gb=max(hbm) if hbm else None)
    if rc != 0:
        row["tail"] = "".join(tail)[-1500:]
        print(f"FAILED (exit {rc}):\n{row['tail']}", flush=True)
    shutil.rmtree(d, ignore_errors=True)  # ~3.7GB of checkpoints per variant
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="base training YAML (the run to tune)")
    ap.add_argument("--seed-ckpt", required=True, help="checkpoint whose weights every variant starts from")
    ap.add_argument("--data-dir", required=True, help="holds train.bin, val.bin, tokenizer.json")
    ap.add_argument("--work", default="/tmp/sweep")
    ap.add_argument("--steps", type=int, default=40, help="steps per variant (log every 10)")
    ap.add_argument("--only", default="", help="comma-separated variant numbers to run, e.g. 0,3")
    ap.add_argument("--out", default="/kaggle/working/tpu_sweep.json")
    args = ap.parse_args()

    from llm_lab.training.trainer import seed_checkpoint

    data = Path(args.data_dir)
    base = yaml.safe_load(Path(args.config).read_text())
    base.update(dataset_type="text", train_path=str(data / "train.txt"), val_path=str(data / "val.txt"),
                val_old_path="", tokenizer_path=str(data / "tokenizer.json"), instruction_data="",
                tpu_cores=0, tpu_spmd=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    seed = work / "seed.pt"
    seed_checkpoint(Path(args.seed_ckpt), seed)  # weights only, step 0: no optimizer to restore

    picked = [VARIANTS[int(i)] for i in args.only.split(",")] if args.only else VARIANTS
    rows = []
    for name, overrides in picked:
        rows.append(_run(name, overrides, base, seed, work, args.steps))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))  # after every variant, in case of a timeout

    ref = next((r["tok_s"] for r in rows if r["tok_s"]), None)
    print("\n| variant | tok/s | vs first | hbm GB | val_loss | first loss | wall s | exit |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        speed = f"{r['tok_s'] / 1e3:.1f}k" if r["tok_s"] else "-"
        rel = f"{r['tok_s'] / ref:.2f}x" if r["tok_s"] and ref else "-"
        hbm = f"{r['hbm_gb']:.1f}" if r["hbm_gb"] else "-"
        val = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "-"
        first = f"{r['first_loss']:.3f}" if r["first_loss"] is not None else "-"
        print(f"| {r['name']} | {speed} | {rel} | {hbm} | {val} | {first} | {r['wall_s']} | {r['rc']} |")
    print(f"\nRows -> {args.out}. val_loss should agree across variants (LR is 0); one that "
          "disagrees by more than ~0.02 is computing something different - don't use it.")


if __name__ == "__main__":
    main()
