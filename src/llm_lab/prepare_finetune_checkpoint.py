"""Prepare a clean fine-tune checkpoint from a diverged/peaked training run.

Loads a good mid-run checkpoint (e.g. step_5000.pt), strips the optimizer
and scheduler state, resets step to 0, and saves a fresh latest.pt that the
trainer will use as a warm-started model for a new config.

Usage (local):
    python prepare_finetune_checkpoint.py \\
        --input training-data/step_5000.pt \\
        --output training-data/finetune_latest.pt

Then upload finetune_latest.pt to Google Drive as:
    /content/drive/MyDrive/aion_checkpoints/mamba_chat_finetune/latest.pt

The new Colab run (Cell 6 with mamba_chat_resume.yaml) will resume from it
and train for 3000 more steps with lr=5e-5.
"""

import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tune checkpoint")
    parser.add_argument("--input", required=True, help="Source checkpoint (.pt file)")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")

    print(f"Loading {src} ...")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    original_step = ckpt.get("step", "?")
    original_metrics = ckpt.get("metrics_log", [])
    val_entries = [e for e in original_metrics if "val_loss" in e]
    last_val = val_entries[-1] if val_entries else None

    print(f"  Original step:     {original_step}")
    if last_val:
        print(f"  Last val_loss:     {last_val['val_loss']:.4f} (at step {last_val['step']})")

    # Keep only model weights — fresh optimizer/scheduler/step
    clean_ckpt = {
        "step": 0,
        "model": ckpt["model"],
        "optimizer": None,   # trainer will init fresh
        "scheduler": None,   # trainer will init fresh
        "metrics_log": [],   # start fresh log
        "_source": str(src),
        "_source_step": original_step,
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean_ckpt, dst)
    size_mb = dst.stat().st_size / 1e6
    print(f"Saved clean checkpoint to {dst} ({size_mb:.1f} MB)")
    print()
    print("Next steps:")
    print(f"  1. Upload {dst} to Google Drive as:")
    print(f"       /content/drive/MyDrive/aion_checkpoints/mamba_chat_finetune/latest.pt")
    print(f"  2. In Colab, run the updated Cell 6 (uses mamba_chat_resume.yaml)")
    print(f"  3. Training will run for 3000 steps with lr=5e-5 — expect val_loss to drop")


if __name__ == "__main__":
    main()
