"""AION-9: metrics as an append-only metrics.jsonl outside the checkpoint."""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import torch

from llm_lab.training import checkpoint, metrics, trainer
from llm_lab.tests.helpers import legacy_checkpoint as _legacy_checkpoint


def _train_steps(log: list[dict]) -> list[int]:
    return [e["step"] for e in log if "train_loss" in e]


def test_read_replays_resumes(tmp_path):
    f = tmp_path / metrics.FILE
    metrics.append(f, [{"step": s, "train_loss": 1.0} for s in range(1, 6)])
    metrics.mark_resume(f, 3)  # rolled back to step 3
    metrics.append(f, [{"step": s, "train_loss": 2.0} for s in range(4, 6)])
    log = metrics.read(f)
    assert _train_steps(log) == [1, 2, 3, 4, 5]
    assert [e["train_loss"] for e in log] == [1.0] * 3 + [2.0] * 2


def test_a_line_cut_short_is_skipped_and_ended(tmp_path):
    f = tmp_path / metrics.FILE
    metrics.append(f, [{"step": 1, "train_loss": 1.0}])
    with open(f, "a") as fh:
        fh.write('{"step": 2, "train_lo')  # killed mid-write
    metrics.append(f, [{"step": 2, "train_loss": 2.0}])
    assert metrics.read(f) == [{"step": 1, "train_loss": 1.0}, {"step": 2, "train_loss": 2.0}]


def test_training_writes_jsonl_not_into_the_checkpoint(tiny_cfg):
    trainer.train(tiny_cfg)
    d = Path(tiny_cfg.checkpoint_dir)
    assert not (d / "metrics.json").exists()
    log = metrics.read(d / metrics.FILE)
    assert _train_steps(log) == [1, 2, 3, 4, 5, 6]
    assert [e["step"] for e in log if "val_loss" in e] == [3, 6]
    assert "metrics_log" not in torch.load(d / "step_6.optim.pt", weights_only=False)


def test_rolled_back_checkpoint_resumes_without_abandoned_steps(tiny_cfg):
    cfg = replace(tiny_cfg, checkpoint_keep_last=2)
    trainer.train(cfg)  # checkpoints at 3 and 6
    d = Path(cfg.checkpoint_dir)
    checkpoint.set_marker(d, "latest", 3)  # roll latest back to step 3
    assert checkpoint.read_state(checkpoint.find(d))["step"] == 3

    result = trainer.train(replace(cfg, max_steps=5))
    assert result["steps"] == 5
    log = metrics.read(d / metrics.FILE)
    assert _train_steps(log) == [1, 2, 3, 4, 5]  # step 6 belonged to the abandoned branch
    assert [e["step"] for e in log if "val_loss" in e] == [3]
    assert checkpoint.read_state(checkpoint.find(d))["step"] == 5


def test_an_old_checkpoints_history_moves_to_jsonl(tiny_cfg):
    d = Path(tiny_cfg.checkpoint_dir)
    _legacy_checkpoint(tiny_cfg, d / "latest.pt", step=4)  # carries a step-4 entry
    (d / "metrics.json").write_text(json.dumps([{"step": 4, "train_loss": 3.0}]))
    trainer.train(tiny_cfg)
    assert not (d / "metrics.json").exists()
    log = metrics.read(d / metrics.FILE)
    assert _train_steps(log) == [4, 5, 6] and log[0]["train_loss"] == 3.0
