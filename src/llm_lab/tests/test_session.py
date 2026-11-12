"""AION-2: the session runner: restore, seed, train, bank, push."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from llm_lab import session
from llm_lab.session import PushError, SessionError, SessionSpec, run_session
from llm_lab.storage import LocalStore, StoreError
from llm_lab.training import checkpoint, metrics, trainer


class RecordingStore(LocalStore):
    """A LocalStore that records each push's step, checks it's never partial, and can fail."""

    def __init__(self, path, fail: int = 0):
        super().__init__(path)
        self.pushes: list[int] = []
        self.fail = fail

    def push(self, paths, message=""):
        paths = [Path(p) for p in paths]
        names = {p.name for p in paths}
        marker = json.loads(next(p for p in paths if p.name == "COMPLETE").read_text())
        assert {marker["checkpoint"] + ".safetensors", marker["checkpoint"] + ".state.json"} <= names
        if self.fail:
            self.fail -= 1
            raise StoreError("scripted failure")
        super().push(paths, message)
        self.pushes.append(marker["step"])

    def step(self) -> int | None:
        m = self.path / "COMPLETE"
        return json.loads(m.read_text())["step"] if m.exists() else None


@pytest.fixture
def spec_for(tiny_cfg, corpus, tmp_path):
    """spec_for(store, **kw): a session of the tiny model on the tiny corpus."""
    def make(store, **kw):
        cfg = {k: v for k, v in tiny_cfg.to_flat().items() if k != "checkpoint_dir"}
        kw.setdefault("work_dir", tmp_path / "work")
        return SessionSpec(name="tiny", store=store, config=cfg, target_steps=kw.pop("target_steps", 6),
                           bank_every=kw.pop("bank_every", 3600), **kw)
    return make


def test_a_missing_run_is_not_started_unless_asked(spec_for, tmp_path):
    store = RecordingStore(tmp_path / "store")
    with pytest.raises(SessionError, match="start=True"):
        run_session(spec_for(store))
    assert store.files() is None


def test_first_session_from_scratch_then_resume(spec_for, tmp_path):
    store = RecordingStore(tmp_path / "store")
    r = run_session(spec_for(store, start=True, steps=3))
    assert (r.start_step, r.end_step, r.pushed_step, r.phase_done) == (0, 3, 3, False)
    assert store.step() == 3 and store.pushes == [3]

    with pytest.raises(SessionError, match="already exists"):
        run_session(spec_for(store, start=True))  # a leftover start flag can't wipe the run

    r = run_session(spec_for(store))  # to the budget
    assert (r.start_step, r.end_step, r.pushed_step, r.phase_done) == (3, 6, 6, True)
    assert set(store.files()) >= {"COMPLETE", "step_6.safetensors", "step_6.state.json",
                                  "step_6.optim.pt", "metrics.jsonl", "tokenizer.json", "run.yaml"}
    log = metrics.read(store.path / metrics.FILE)
    assert [e["step"] for e in log if "train_loss" in e] == [1, 2, 3, 4, 5, 6]


def test_seeding_from_a_parent(spec_for, tiny_cfg, tmp_path):
    parent = RecordingStore(tmp_path / "parent")
    run_session(spec_for(parent, start=True, work_dir=tmp_path / "pw"))  # parent at step 6
    child = RecordingStore(tmp_path / "child")
    r = run_session(spec_for(child, start=True, parent=parent, steps=3, target_steps=10))
    assert (r.start_step, r.end_step) == (0, 3)
    state = checkpoint.read_state(checkpoint.find(child.path))
    assert state["step"] == 3 and state["parent"]["step"] == 6
    assert state["parent"]["store"] == str(parent)


def test_seeding_from_an_old_pt_parent(spec_for, tiny_cfg, tmp_path):
    from llm_lab.tests.helpers import legacy_checkpoint
    parent = LocalStore(tmp_path / "old_parent")
    src = tmp_path / "old_src"
    legacy_checkpoint(tiny_cfg, src / "latest.pt", step=40)
    (src / "metrics.json").write_text("[]")
    parent.push(list(src.iterdir()))
    child = RecordingStore(tmp_path / "child")
    run_session(spec_for(child, start=True, parent=parent, steps=2))
    state = checkpoint.read_state(checkpoint.find(child.path))
    assert state["step"] == 2 and state["parent"]["step"] == 40
    assert state["config"]["model"]["d_model"] == tiny_cfg.model.d_model  # loads with no YAML


def test_a_store_without_a_checkpoint_stops_the_session(spec_for, tmp_path):
    store = LocalStore(tmp_path / "store")
    store.push([_file(tmp_path / "x" / "notes.txt")])
    with pytest.raises(SessionError, match="no checkpoint"):
        run_session(spec_for(store))


def _file(p: Path, body: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


def test_budget_reached_trains_and_pushes_nothing(spec_for, tmp_path):
    store = RecordingStore(tmp_path / "store")
    run_session(spec_for(store, start=True))
    store.pushes.clear()
    r = run_session(spec_for(store))
    assert r.phase_done and r.end_step == 6 and store.pushes == []


def test_no_progress_pushes_nothing(spec_for, tmp_path):
    store = RecordingStore(tmp_path / "store")
    run_session(spec_for(store, start=True, steps=3))
    before = {n: (store.path / n).stat().st_mtime for n in store.files()}
    store.pushes.clear()
    with pytest.raises(RuntimeError, match="exited with code"):  # crashes before a checkpoint
        run_session(spec_for(store, overrides={"train_path": str(tmp_path / "missing.txt")}))
    assert store.pushes == []
    assert {n: (store.path / n).stat().st_mtime for n in store.files()} == before


def test_a_failed_push_is_retried(spec_for, tmp_path):
    store = RecordingStore(tmp_path / "store", fail=1)
    r = run_session(spec_for(store, start=True))  # the final push fails once, then retries
    assert r.pushed_step == 6 and store.step() == 6


def test_a_push_that_keeps_failing_is_never_recorded_as_done(spec_for, tmp_path):
    store = RecordingStore(tmp_path / "store")
    run_session(spec_for(store, start=True, steps=3))
    store.fail = 99
    with pytest.raises(PushError, match="not pushed"):
        run_session(spec_for(store, fallback_dir=tmp_path / "fallback"))
    assert store.step() == 3  # the store still says what it really has
    assert json.loads((tmp_path / "fallback" / "COMPLETE").read_text())["step"] == 6
    store.fail = 0
    r = run_session(spec_for(store))  # the next session resumes from what the store holds
    assert r.start_step == 3 and store.step() == 6


def test_a_rolled_back_store_resumes_and_pushes(spec_for, tmp_path):
    """AION-9's last clause. The store goes back to its step-3 version (as restoring an older
    Kaggle Dataset version would), but keeps a metrics.jsonl that runs to step 6. Progress
    comes from state.json, so the session resumes at 3 and pushes 5; nothing freezes on the
    metrics' higher step, and the abandoned steps 4-6 drop out of the history."""
    import shutil
    store = RecordingStore(tmp_path / "store")
    run_session(spec_for(store, start=True, steps=3))
    v3 = tmp_path / "v3"
    shutil.copytree(store.path, v3)
    run_session(spec_for(store))
    assert store.step() == 6
    shutil.copyfile(store.path / metrics.FILE, v3 / metrics.FILE)  # history up to step 6
    store.push(sorted(v3.iterdir()), "roll back to step 3")
    store.pushes.clear()

    r = run_session(spec_for(store, target_steps=5))
    assert (r.start_step, r.end_step) == (3, 5) and store.pushes == [5]
    assert store.step() == 5 and checkpoint.steps_on_disk(store.path)[-1] == 5
    log = metrics.read(store.path / metrics.FILE)
    assert [e["step"] for e in log if "train_loss" in e] == [1, 2, 3, 4, 5]


def test_background_banking_pushes_while_training(spec_for, tmp_path, monkeypatch):
    """A stand-in trainer saves a checkpoint every 0.3s; the watcher banks them as they come,
    so a hard timeout would lose at most one interval."""
    def slow_train(spec, cfg, on_line=None):
        w = {"w": torch.zeros(2)}
        for step in range(cfg["max_steps"] - 4 + 1, cfg["max_steps"] + 1):
            time.sleep(0.3)
            checkpoint.write(spec.ckpt_dir, w, {}, {"step": step}, {"optimizer": None, "scheduler": None})
        time.sleep(0.3)
        return 0

    store = RecordingStore(tmp_path / "store")
    run_session(spec_for(store, start=True, steps=1))
    monkeypatch.setattr(session, "train", slow_train)
    store.pushes.clear()
    r = run_session(spec_for(store, steps=4, target_steps=10, bank_every=0.1))
    assert r.end_step == 5 and store.step() == 5
    assert len(store.pushes) >= 3 and store.pushes == sorted(set(store.pushes))


def test_kaggle_sessions_like_the_tpu_notebook(fake_kaggle, tiny_cfg, corpus, tmp_path):
    """The notebook's flow against the fake kaggle CLI: data from a Dataset, the first
    session seeds from an old-format parent Dataset and creates the run's Dataset, the
    second resumes from it."""
    from llm_lab.data.dataset import TextDataset
    from llm_lab.storage import KaggleStore
    from llm_lab.tests.helpers import legacy_checkpoint
    from llm_lab.tokenizer.bpe import load_tokenizer

    def kaggle(slug):
        return KaggleStore(f"me/{slug}", wait=0, push_wait=0)

    tok = load_tokenizer(corpus["tokenizer"])
    for name in ("train", "val"):
        TextDataset(corpus[name], tok, tiny_cfg.model.seq_len)  # writes train.bin / val.bin
    kaggle("data").push([corpus["dir"] / n for n in ("train.bin", "val.bin", "tokenizer.json")])
    parent_src = tmp_path / "parent_src"
    legacy_checkpoint(tiny_cfg, parent_src / "latest.pt", step=20000)
    (parent_src / "tokenizer.json").write_bytes(corpus["tokenizer"].read_bytes())
    kaggle("parent").push(sorted(parent_src.iterdir()))

    cfg = {k: v for k, v in tiny_cfg.to_flat().items()
           if k not in ("checkpoint_dir", "train_path", "val_path", "tokenizer_path")}

    def spec(**kw):
        return SessionSpec(name="phase2", store=kaggle("phase2"), parent=kaggle("parent"),
                           data=kaggle("data"), config=cfg, target_steps=6, steps=3,
                           work_dir=tmp_path / "run", bank_every=3600, **kw)

    r = run_session(spec(start=True))
    assert (r.start_step, r.end_step, r.pushed_step) == (0, 3, 3)
    pushed = fake_kaggle.dataset("me/phase2")
    state = json.loads((pushed / "step_3.state.json").read_text())
    assert state["parent"]["step"] == 20000 and state["data"]["id"] == "kaggle:me/data"
    assert state["config"]["data"]["train_path"].endswith("train.txt")

    r = run_session(spec())
    assert (r.start_step, r.end_step, r.phase_done) == (3, 6, True)
    assert json.loads((pushed / "COMPLETE").read_text())["step"] == 6
    assert not any(c["token"] for c in fake_kaggle.calls())


@pytest.fixture
def toy_bench(monkeypatch):
    """A one-question benchmark in place of the HF ones: nothing downloads."""
    from llm_lab.eval import benchmarks
    item = benchmarks.Item(context="the cat sat on", choices=[" the mat", " blue"], label=0)
    monkeypatch.setitem(benchmarks.TASKS, "toy", lambda: [item] * 3)
    monkeypatch.setitem(benchmarks.HEADLINE, "toy", "acc")
    monkeypatch.setitem(benchmarks.CHANCE, "toy", 0.5)
    return ("toy",)


def test_pretraining_keeps_no_best_and_benches_at_phase_end(spec_for, toy_bench, tmp_path):
    """AION-13: a text run writes no BEST, and the session that finishes the phase scores
    its last checkpoint and pushes bench.json with it."""
    store = RecordingStore(tmp_path / "store")
    run_session(spec_for(store, start=True, steps=3, bench=toy_bench))
    assert not (store.path / "BEST").exists() and "bench.json" not in store.files()

    run_session(spec_for(store, bench=toy_bench, output_dir=tmp_path / "out"))
    assert not (store.path / "BEST").exists()
    bench = json.loads((store.path / "bench.json").read_text())
    assert bench["step"] == 6 and bench["checkpoint"] == "step_6.safetensors"
    assert bench["tasks"]["toy"]["n"] == 3
    assert json.loads((tmp_path / "out" / "bench.json").read_text()) == bench
    assert store.pushes == [3, 6, 6]  # the checkpoint, then again with bench.json


def test_a_bench_the_last_session_missed_runs_next_session(spec_for, toy_bench, tmp_path):
    store = RecordingStore(tmp_path / "store")
    run_session(spec_for(store, start=True))  # finished the phase without benching
    assert "bench.json" not in store.files()
    store.pushes.clear()
    r = run_session(spec_for(store, bench=toy_bench))
    assert r.phase_done and json.loads((store.path / "bench.json").read_text())["step"] == 6
    store.pushes.clear()
    run_session(spec_for(store, bench=toy_bench))  # already benched: nothing more to do
    assert store.pushes == []


def test_text_configs_cannot_early_stop(tiny_cfg):
    with pytest.raises(ValueError, match="early_stop_patience"):
        trainer.train(tiny_cfg.replace(early_stop_patience=3))


def test_finetuning_still_keeps_a_best(chat_cfg):
    trainer.train(chat_cfg)
    assert checkpoint.find(chat_cfg.data.checkpoint_dir, "best") is not None
