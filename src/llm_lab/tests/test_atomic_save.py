"""AION-10: a process killed mid-save leaves the previous checkpoint intact, and the
uploader's view (checkpoint.banked) never includes a partial one."""
from __future__ import annotations

import json
import signal
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from llm_lab.training import checkpoint, trainer

SRC = Path(__file__).resolve().parents[2]

# Trains with the config in argv[1]; dies (SIGKILL, as on a hard timeout) halfway through
# writing the temp file of argv[2] (for COMPLETE: step 6's), before it can be renamed into place.
CHILD = r"""
import json, os, signal, sys
from llm_lab.training import checkpoint, trainer
from llm_lab.training.config import TrainConfig

cfg = TrainConfig(**json.loads(sys.argv[1]))
victim = sys.argv[2]
real = checkpoint._atomic

def dying(path, write):
    # COMPLETE is written at step 3 too: die on step 6's.
    if path.name == victim and (path.parent / "step_6.state.json").exists() == (victim == "COMPLETE"):
        tmp = path.with_name(path.name + checkpoint.TMP)
        write(tmp)
        with open(tmp, "r+b") as f:
            f.truncate(max(1, tmp.stat().st_size // 2))
        os.kill(os.getpid(), signal.SIGKILL)
    real(path, write)

checkpoint._atomic = dying
trainer.train(cfg)
"""


@pytest.mark.parametrize("victim", ["step_6.safetensors", "step_6.optim.pt",
                                    "step_6.state.json", "COMPLETE"])
def test_kill_mid_save_keeps_the_previous_checkpoint(tiny_cfg, tmp_path, victim):
    from dataclasses import replace
    cfg = replace(tiny_cfg, checkpoint_keep_last=2, eval_every=100)  # no best: only latest saves
    clean = replace(cfg, checkpoint_dir=str(tmp_path / "clean"), max_steps=3)
    trainer.train(clean)

    proc = subprocess.run([sys.executable, "-c", CHILD, json.dumps(asdict(cfg)), victim],
                          cwd=SRC, capture_output=True, text=True, timeout=120)
    assert proc.returncode == -signal.SIGKILL, proc.stderr[-2000:]

    d = Path(cfg.checkpoint_dir)
    step, files = checkpoint.banked(d)
    assert step == 3
    assert all(f.is_file() for f in files)
    assert not any("step_6" in f.name or f.name.endswith(checkpoint.TMP) for f in files)
    kept = checkpoint.read_weights(checkpoint.find(d))
    ref = checkpoint.read_weights(checkpoint.find(clean.checkpoint_dir))
    assert all((kept[k] == ref[k]).all() for k in ref)  # bit-identical to the clean run's step 3

    result = trainer.train(cfg)  # resumes from 3, as if the save had never started
    assert result["steps"] == 6
    assert checkpoint.banked(d)[0] == 6
    assert not list(d.glob(f"*{checkpoint.TMP}"))


def test_banked_lists_latest_best_and_markers(tiny_cfg):
    trainer.train(tiny_cfg)
    d = Path(tiny_cfg.checkpoint_dir)
    step, files = checkpoint.banked(d)
    best = checkpoint.read_marker(d, "best")["step"]
    names = {f.name for f in files}
    assert step == 6
    assert {"step_6.safetensors", "step_6.state.json", "step_6.optim.pt", "COMPLETE", "BEST",
            f"step_{best}.safetensors", f"step_{best}.state.json"} <= names


def test_nothing_banked_before_the_first_save(tmp_path):
    assert checkpoint.banked(tmp_path) is None
