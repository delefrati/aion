"""AION-8: self-describing checkpoints (safetensors + state.json), and old .pt still loading."""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from llm_lab.training import checkpoint, trainer
from llm_lab.training.model_factory import build_model
from llm_lab.tests.helpers import legacy_checkpoint as _legacy_checkpoint

SRC = Path(__file__).resolve().parents[2]


def _same_weights(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_training_writes_the_new_format(tiny_cfg, corpus):
    trainer.train(tiny_cfg)
    d = Path(tiny_cfg.checkpoint_dir)
    for f in ("step_6.safetensors", "step_6.state.json", "step_6.optim.pt", "tokenizer.json"):
        assert (d / f).is_file(), f
    assert json.loads((d / "COMPLETE").read_text()) == {"step": 6, "checkpoint": "step_6"}

    state = json.loads((d / "step_6.state.json").read_text())
    assert state["step"] == 6
    assert state["tokens_seen"] == 6 * tiny_cfg.batch_size * tiny_cfg.seq_len
    assert state["config"] == asdict(tiny_cfg)
    assert state["tokenizer_sha256"] == checkpoint.file_sha256(corpus["tokenizer"])
    assert state["data"]["files"]["train.bin"] > 0
    assert state["has_optimizer"] is True
    assert state["tied"] == {"lm_head.weight": "embedding.weight"}
    # keep_last=1: the latest, and the best (weights only) the evals found.
    best = json.loads((d / "BEST").read_text())["step"]
    assert set(checkpoint.steps_on_disk(d)) == {6, best}


def test_load_model_needs_no_config(tiny_cfg):
    trainer.train(tiny_cfg)
    d = Path(tiny_cfg.checkpoint_dir)
    model, state = checkpoint.load_model(d)
    assert state["step"] == 6
    assert model.lm_head.weight is model.embedding.weight  # tied again, as trained
    # Every way of naming the checkpoint finds the same one.
    for p in (d, d / "step_6.safetensors", d / "step_6.state.json", d / "step_6"):
        assert checkpoint.locate(p) == checkpoint.Ref(d, "step_6", legacy=False)


def test_tied_weights_are_stored_once(tiny_cfg):
    trainer.train(tiny_cfg)
    from safetensors import safe_open
    with safe_open(str(checkpoint.find(tiny_cfg.checkpoint_dir).weights), "pt") as f:
        assert "lm_head.weight" not in f.keys() and "embedding.weight" in f.keys()


def test_untied_weights_build_an_untied_model(tiny_cfg, tmp_path):
    """A TPU run's config says tie_embeddings=True but its weights come back untied; the
    loader must follow the weights, or one matrix would overwrite the other."""
    model = build_model(replace(tiny_cfg, tie_embeddings=False))
    weights = checkpoint.strip_prefixes(model.state_dict())
    state = {"step": 1, "config": asdict(tiny_cfg)}  # claims tied
    checkpoint.write(tmp_path, weights, checkpoint.aliases(model), state, None)
    loaded, _ = checkpoint.load_model(tmp_path)
    assert loaded.lm_head.weight is not loaded.embedding.weight
    assert torch.equal(loaded.lm_head.weight, model.lm_head.weight)
    assert torch.equal(loaded.embedding.weight, model.embedding.weight)


def test_old_pt_still_loads_with_a_config(tiny_cfg, tmp_path):
    weights = _legacy_checkpoint(tiny_cfg, tmp_path / "latest.pt", step=40)
    with pytest.raises(ValueError, match="--config"):
        checkpoint.load_model(tmp_path / "latest.pt")
    model, state = checkpoint.load_model(tmp_path / "latest.pt", cfg=tiny_cfg)
    assert state["step"] == 40 and state["format"] == 0
    assert _same_weights(checkpoint.strip_prefixes(model.state_dict()), weights)
    assert checkpoint.locate(tmp_path) == checkpoint.Ref(tmp_path, "latest", legacy=True)


def test_training_resumes_an_old_pt_and_replaces_it(tiny_cfg):
    d = Path(tiny_cfg.checkpoint_dir)
    _legacy_checkpoint(tiny_cfg, d / "latest.pt", step=4)
    result = trainer.train(tiny_cfg)  # max_steps 6: two more steps
    assert result["steps"] == 6
    assert not (d / "latest.pt").exists()
    state = checkpoint.read_state(checkpoint.find(d))
    assert state["step"] == 6
    # The old checkpoint never counted tokens: estimated from this session's rate.
    assert state["tokens_seen"] == 6 * tiny_cfg.batch_size * tiny_cfg.seq_len
    assert state["tokens_estimated"] is True


def test_resume_carries_tokens_and_parent(tiny_cfg):
    d = Path(tiny_cfg.checkpoint_dir)
    d.mkdir(parents=True)
    (d / "base.json").write_text(json.dumps({"dataset": "someone/base", "step": 123}))
    trainer.train(replace(tiny_cfg, max_steps=3))
    trainer.train(replace(tiny_cfg, max_steps=5, batch_size=2))
    state = checkpoint.read_state(checkpoint.find(d))
    tok = tiny_cfg.seq_len
    assert state["tokens_seen"] == 3 * 4 * tok + 2 * 2 * tok
    assert state["parent"] == {"dataset": "someone/base", "step": 123}
    assert state["tokens_estimated"] is False


def test_seed_checkpoint_from_old_and_new(tiny_cfg, tmp_path):
    weights = _legacy_checkpoint(tiny_cfg, tmp_path / "old" / "latest.pt", step=40)
    ref = trainer.seed_checkpoint(tmp_path / "old" / "latest.pt", tmp_path / "seeded")
    state = checkpoint.read_state(ref)
    assert state["step"] == 0 and state["has_optimizer"] is False
    assert state["parent"]["step"] == 40
    assert _same_weights(checkpoint.read_weights(ref), weights)

    trainer.train(tiny_cfg)
    ref = trainer.seed_checkpoint(tiny_cfg.checkpoint_dir, tmp_path / "seeded2",
                                  parent={"run": "tiny", "step": 6})
    state = checkpoint.read_state(ref)
    assert state["config"]["d_model"] == 32 and state["parent"] == {"run": "tiny", "step": 6}
    model, _ = checkpoint.load_model(ref.weights)  # the seed alone rebuilds the model
    assert model.lm_head.weight is model.embedding.weight


def test_generate_cli_without_config(tiny_cfg):
    trainer.train(tiny_cfg)
    out = subprocess.run(
        [sys.executable, "-m", "llm_lab.cli", "generate", "--checkpoint", tiny_cfg.checkpoint_dir,
         "--prompt", "the cat", "--max-tokens", "5"],
        cwd=SRC, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip()


def test_bench_loader_without_config(tiny_cfg):
    from llm_lab.eval.benchmarks import load_checkpoint_model
    trainer.train(tiny_cfg)
    model, step = load_checkpoint_model(None, Path(tiny_cfg.checkpoint_dir) / "step_6.safetensors", "cpu")
    assert step == 6 and not model.training
