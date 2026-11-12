"""AION-8: self-describing checkpoints (safetensors + state.json), and old .pt still loading."""
from __future__ import annotations

import json
import subprocess
import sys
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
    d = Path(tiny_cfg.data.checkpoint_dir)
    for f in ("step_6.safetensors", "step_6.state.json", "step_6.optim.pt", "tokenizer.json"):
        assert (d / f).is_file(), f
    assert json.loads((d / "COMPLETE").read_text()) == {"step": 6, "checkpoint": "step_6"}

    state = json.loads((d / "step_6.state.json").read_text())
    assert state["step"] == 6
    assert state["tokens_seen"] == 6 * tiny_cfg.schedule.batch_size * tiny_cfg.model.seq_len
    assert state["config"] == tiny_cfg.to_dict()
    assert state["tokenizer_sha256"] == checkpoint.file_sha256(corpus["tokenizer"])
    assert state["data"]["files"]["train.bin"] > 0
    assert state["has_optimizer"] is True
    assert state["tied"] == {"lm_head.weight": "embedding.weight"}
    # keep_last=1: only the latest. A pretraining run keeps no best (AION-13).
    assert checkpoint.steps_on_disk(d) == [6] and not (d / "BEST").exists()


def test_load_model_needs_no_config(tiny_cfg):
    trainer.train(tiny_cfg)
    d = Path(tiny_cfg.data.checkpoint_dir)
    model, state = checkpoint.load_model(d)
    assert state["step"] == 6
    assert model.lm_head.weight is model.embedding.weight  # tied again, as trained
    # Every way of naming the checkpoint finds the same one.
    for p in (d, d / "step_6.safetensors", d / "step_6.state.json", d / "step_6"):
        assert checkpoint.locate(p) == checkpoint.Ref(d, "step_6", legacy=False)


def test_tied_weights_are_stored_once(tiny_cfg):
    trainer.train(tiny_cfg)
    from safetensors import safe_open
    with safe_open(str(checkpoint.find(tiny_cfg.data.checkpoint_dir).weights), "pt") as f:
        assert "lm_head.weight" not in f.keys() and "embedding.weight" in f.keys()


def test_untied_weights_build_an_untied_model(tiny_cfg, tmp_path):
    """A TPU run's config says tie_embeddings=True but its weights come back untied; the
    loader must follow the weights, or one matrix would overwrite the other."""
    model = build_model(tiny_cfg.replace(tie_embeddings=False))
    weights = checkpoint.strip_prefixes(model.state_dict())
    state = {"step": 1, "config": tiny_cfg.to_flat()}  # claims tied
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
    d = Path(tiny_cfg.data.checkpoint_dir)
    _legacy_checkpoint(tiny_cfg, d / "latest.pt", step=4)
    result = trainer.train(tiny_cfg)  # max_steps 6: two more steps
    assert result["steps"] == 6
    assert not (d / "latest.pt").exists()
    state = checkpoint.read_state(checkpoint.find(d))
    assert state["step"] == 6
    # The old checkpoint never counted tokens: estimated from this session's rate.
    assert state["tokens_seen"] == 6 * tiny_cfg.schedule.batch_size * tiny_cfg.model.seq_len
    assert state["tokens_estimated"] is True


def test_resume_carries_tokens_and_parent(tiny_cfg):
    d = Path(tiny_cfg.data.checkpoint_dir)
    d.mkdir(parents=True)
    (d / "base.json").write_text(json.dumps({"dataset": "someone/base", "step": 123}))
    trainer.train(tiny_cfg.replace(max_steps=3))
    trainer.train(tiny_cfg.replace(max_steps=5, batch_size=2))
    state = checkpoint.read_state(checkpoint.find(d))
    tok = tiny_cfg.model.seq_len
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
    ref = trainer.seed_checkpoint(tiny_cfg.data.checkpoint_dir, tmp_path / "seeded2",
                                  parent={"run": "tiny", "step": 6})
    state = checkpoint.read_state(ref)
    assert state["config"]["model"]["d_model"] == 32 and state["parent"] == {"run": "tiny", "step": 6}
    model, _ = checkpoint.load_model(ref.weights)  # the seed alone rebuilds the model
    assert model.lm_head.weight is model.embedding.weight


def test_generate_cli_without_config(tiny_cfg):
    trainer.train(tiny_cfg)
    out = subprocess.run(
        [sys.executable, "-m", "llm_lab.cli", "generate", "--checkpoint", tiny_cfg.data.checkpoint_dir,
         "--prompt", "the cat", "--max-tokens", "5"],
        cwd=SRC, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip()


def test_bench_loader_without_config(tiny_cfg):
    from llm_lab.eval.benchmarks import load_checkpoint_model
    trainer.train(tiny_cfg)
    model, step = load_checkpoint_model(None, Path(tiny_cfg.data.checkpoint_dir) / "step_6.safetensors", "cpu")
    assert step == 6 and not model.training


# --- AION-5: the config's sections ----------------------------------------------------------

def test_a_misspelled_key_fails_with_its_name(tmp_path):
    from llm_lab.training.config import ConfigError, TrainConfig
    bad = tmp_path / "bad.yaml"
    bad.write_text("d_model: 64\nlr_decya_steps: 100\n")
    with pytest.raises(ConfigError, match=r"'lr_decya_steps'.*did you mean 'lr_decay_steps'"):
        TrainConfig.load(bad)
    with pytest.raises(ConfigError, match=r"'d_model' in section 'schedule'.*belongs in 'model'"):
        TrainConfig.from_dict({"schedule": {"d_model": 64}})


def test_every_config_in_the_repo_loads():
    from llm_lab.training.config import TrainConfig
    configs = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))
    assert len(configs) > 20
    for p in configs:
        TrainConfig.load(p)


def test_flat_nested_and_mixed_dicts_agree(tmp_path):
    from llm_lab.training.config import TrainConfig
    flat = TrainConfig.from_dict({"d_model": 64, "lr": "5e-4", "tpu_spmd": True, "train_path": "t"})
    nested = TrainConfig.from_dict({"model": {"d_model": 64}, "schedule": {"lr": 5e-4},
                                    "device": {"tpu_spmd": True}, "data": {"train_path": "t"}})
    mixed = TrainConfig.from_dict({"model": {"d_model": 32}, "d_model": 64, "lr": 5e-4,
                                   "device": {"tpu_spmd": True}, "train_path": "t"})
    assert flat == nested == mixed and flat.schedule.lr == 5e-4
    flat.save(tmp_path / "c.yaml")  # written nested, read back the same
    assert TrainConfig.load(tmp_path / "c.yaml") == flat
    assert TrainConfig.from_dict(flat.to_flat()) == flat


def test_saved_configs_from_older_versions_still_load():
    from llm_lab.training.config import TrainConfig
    cfg = TrainConfig.from_dict({"d_model": 64, "a_knob_since_removed": 1}, strict=False)
    assert cfg.model.d_model == 64
