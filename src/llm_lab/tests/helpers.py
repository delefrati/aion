"""Helpers shared by the tests."""
from __future__ import annotations

from pathlib import Path

import torch

from llm_lab.training import checkpoint
from llm_lab.training.model_factory import build_model


def legacy_checkpoint(cfg, path: Path, step: int) -> dict:
    """An old-style latest.pt, as the trainer wrote it before AION-8."""
    torch.manual_seed(1)
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.schedule.lr)
    model(torch.zeros(1, 4, dtype=torch.long)).sum().backward()
    opt.step()
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": {"_orig_mod." + k: v for k, v in model.state_dict().items()},
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "metrics_log": [{"step": step, "train_loss": 3.0}]}, path)
    return checkpoint.strip_prefixes(model.state_dict())
