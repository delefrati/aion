"""Session timings: every stage of a session, the trainer's included, is recorded."""
from __future__ import annotations

import time

from llm_lab import timing
from llm_lab.session import run_session
from llm_lab.tests.test_session import RecordingStore, spec_for  # noqa: F401 (fixture)


def test_a_session_times_every_stage(spec_for, tmp_path, capsys):
    store = RecordingStore(tmp_path / "store")
    out = tmp_path / "out"
    spec = spec_for(store, start=True, steps=3, output_dir=out, started_at=time.time() - 5)
    spec.config.update(eval_every=2, checkpoint_every=2)
    run_session(spec)

    events = timing.read(out / "timings.jsonl")
    stages = [e["stage"] for e in events]
    for name in ("setup", "restore", "trainer_setup", "first_step", "eval", "save", "train", "push"):
        assert name in stages, f"{name} not timed: {stages}"
    assert stages.index("restore") < stages.index("trainer_setup") < stages.index("train")
    assert next(e for e in events if e["stage"] == "setup")["s"] >= 5
    assert [e["step"] for e in events if e["stage"] == "save"] == [2, 3]
    push = next(e for e in events if e["stage"] == "push")
    assert push["step"] == 3 and push["gb"] >= 0 and "ok" not in push
    assert "where this session's time went" in capsys.readouterr().out


def test_each_session_starts_a_fresh_file(spec_for, tmp_path):
    store = RecordingStore(tmp_path / "store")
    out = tmp_path / "out"
    run_session(spec_for(store, start=True, steps=3, output_dir=out))
    run_session(spec_for(store, output_dir=out))
    restores = [e for e in timing.read(out / "timings.jsonl") if e["stage"] == "restore"]
    assert len(restores) == 1


def test_a_failed_stage_is_recorded_and_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(timing.ENV, str(tmp_path / "t.jsonl"))
    try:
        with timing.stage("push", step=4):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    (e,) = timing.read(tmp_path / "t.jsonl")
    assert e["stage"] == "push" and e["ok"] is False and e["step"] == 4


def test_nothing_is_written_outside_a_session(tmp_path, monkeypatch):
    monkeypatch.delenv(timing.ENV, raising=False)
    timing.record("save", 1.0)  # the trainer run on its own
    assert not list(tmp_path.iterdir())


def test_summary_table():
    table = timing.summary([{"stage": "save", "s": 30.0}, {"stage": "save", "s": 90.0},
                            {"stage": "train", "s": 7200.0}])
    lines = table.splitlines()
    assert lines[1].split() == ["save", "2", "2.0m", "1.0m", "1.5m"]
    assert lines[2].split() == ["train", "1", "2.00h", "2.00h", "2.00h"]
