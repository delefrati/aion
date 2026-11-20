"""AION-11: three short sessions through the runner against a local store, one killed
mid-run, end with the same weights, step and metrics as one uninterrupted run."""
from __future__ import annotations

import time

import pytest
import torch

from llm_lab import session
from llm_lab.session import SessionSpec, run_session
from llm_lab.storage import LocalStore
from llm_lab.training import checkpoint, metrics

TARGET = 12


def _spec(cfg, store, work, **kw) -> SessionSpec:
    config = {k: v for k, v in cfg.to_flat().items() if k != "checkpoint_dir"}
    return SessionSpec(name="resume", store=store, config=config, target_steps=TARGET,
                       work_dir=work, bank_every=3600, **kw)


def _losses(store: LocalStore) -> dict[int, float]:
    log = metrics.read(store.path / metrics.FILE)
    steps = [e["step"] for e in log if "train_loss" in e]
    assert steps == sorted(set(steps)), f"steps logged twice or out of order: {steps}"
    return {e["step"]: e["train_loss"] for e in log if "train_loss" in e}


def _kill_after_step(n: int):
    """on_line hook: SIGKILL the trainer once it logs step n (as a hard timeout would)."""
    def hook(line, proc):
        if line.startswith(f"step {n:>5d} | loss"):
            proc.kill()
    return hook


def _three_sessions(cfg, tmp_path) -> LocalStore:
    store = LocalStore(tmp_path / "store")
    work = tmp_path / "work"
    run_session(_spec(cfg, store, work, start=True, steps=4))            # 0 -> 4
    with pytest.raises(RuntimeError, match="exited with code -9"):       # 4 -> killed after 7
        run_session(_spec(cfg, store, work, steps=6), on_line=_kill_after_step(7))
    assert checkpoint.read_marker(store.path)["step"] == 6               # banked: the last save
    run_session(_spec(cfg, store, work))                                 # 6 -> 12
    return store


@pytest.fixture
def cfg(tiny_cfg):
    # Dropout on, so the RNG must resume too; a checkpoint every 2 steps.
    return tiny_cfg.replace(dropout=0.1, checkpoint_every=2, eval_every=4, max_steps=TARGET)


def test_killed_and_resumed_matches_uninterrupted(cfg, tmp_path):
    t0 = time.time()
    ref_store = LocalStore(tmp_path / "ref")
    run_session(_spec(cfg, ref_store, tmp_path / "ref_work", start=True))  # 0 -> 12 at once
    store = _three_sessions(cfg, tmp_path)

    ref, got = checkpoint.find(ref_store.path), checkpoint.find(store.path)
    assert checkpoint.read_state(got)["step"] == checkpoint.read_state(ref)["step"] == TARGET
    want, have = checkpoint.read_weights(ref), checkpoint.read_weights(got)
    assert want.keys() == have.keys()
    for k in want:
        assert torch.equal(want[k], have[k]), f"{k} differs from the uninterrupted run"
    assert _losses(store) == _losses(ref_store)  # every step once, with the same loss
    assert sorted(_losses(store)) == list(range(1, TARGET + 1))
    assert checkpoint.read_state(got)["tokens_seen"] == checkpoint.read_state(ref)["tokens_seen"]
    assert time.time() - t0 < 120


@pytest.mark.parametrize("off_by", [-1, 1])
def test_the_check_catches_a_resume_that_skips_or_repeats(cfg, tmp_path, monkeypatch, off_by):
    """If resume started one step early (repeat) or late (skip), the weights would differ."""
    ref_store = LocalStore(tmp_path / "ref")
    run_session(_spec(cfg, ref_store, tmp_path / "ref_work", start=True))

    from llm_lab.training import trainer
    real = trainer._resume_if_available

    def off(*a, **kw):
        step, *rest = real(*a, **kw)
        return (step + off_by if step else step), *rest

    monkeypatch.setattr(trainer, "_resume_if_available", off)
    # The runner trains in a subprocess, which wouldn't see the patch: train in-process.
    monkeypatch.setattr(session, "train", lambda spec, c, on_line=None: _train_inline(c))
    store = LocalStore(tmp_path / "store")
    run_session(_spec(cfg, store, tmp_path / "work", start=True, steps=4))
    run_session(_spec(cfg, store, tmp_path / "work"))
    want = checkpoint.read_weights(checkpoint.find(ref_store.path))
    have = checkpoint.read_weights(checkpoint.find(store.path))
    assert any(not torch.equal(want[k], have[k]) for k in want)


def _train_inline(cfg: dict) -> int:
    from llm_lab.training import trainer
    from llm_lab.training.config import TrainConfig
    trainer.train(TrainConfig.from_dict(cfg))
    return 0


def test_step_batch_sampler_is_a_function_of_seed_and_batch():
    from itertools import islice
    from llm_lab.data.sampling import StepBatchSampler
    full = list(islice(iter(StepBatchSampler(1000, 4, seed=7)), 10))
    resumed = list(islice(iter(StepBatchSampler(1000, 4, seed=7, start=6)), 4))
    assert resumed == full[6:]
    assert list(islice(iter(StepBatchSampler(1000, 4, seed=8)), 10)) != full

    # Two ranks together draw what one process would, interleaved.
    r0 = list(islice(iter(StepBatchSampler(1000, 4, seed=7, world=2, rank=0)), 5))
    r1 = list(islice(iter(StepBatchSampler(1000, 4, seed=7, world=2, rank=1)), 5))
    assert [b for pair in zip(r0, r1) for b in pair] == full

    # Without replacement: each epoch is a permutation, and epochs differ.
    perm = StepBatchSampler(10, 5, seed=1, replacement=False)
    e0, e1 = perm.batch(0) + perm.batch(1), perm.batch(2) + perm.batch(3)
    assert sorted(e0) == sorted(e1) == list(range(10)) and e0 != e1


@pytest.mark.parametrize("capturable", [False, True])
def test_restored_optimizer_keeps_step_where_a_fresh_one_does(capturable):
    """AION-27: moving a restored AdamW's `step` onto an XLA device made every optimizer
    step call .item() there, ~30% of a TPU session's throughput. The moments move; `step`
    stays on the CPU unless the optimizer is capturable."""
    from llm_lab.training.trainer import _optimizer_state_to
    p = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.AdamW([p])
    p.grad = torch.ones(4)
    opt.step()
    restored = torch.optim.AdamW([p])
    restored.load_state_dict(opt.state_dict())
    restored.param_groups[0]["capturable"] = capturable  # CPU AdamW refuses capturable=True
    _optimizer_state_to(restored, torch.device("meta"))  # any device other than the CPU
    state = restored.state[p]
    assert state["exp_avg"].device.type == "meta" and state["exp_avg_sq"].device.type == "meta"
    assert state["step"].device.type == ("meta" if capturable else "cpu")
