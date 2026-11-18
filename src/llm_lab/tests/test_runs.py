"""AION-3: a notebook's active run comes from runs.yaml."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from llm_lab import runs
from llm_lab.runs import RunsError, active

NOTEBOOKS = Path(runs.__file__).resolve().parent


def test_the_repo_file_loads():
    loaded = runs.load()
    assert loaded["notebooks"], "runs.yaml names no notebook"


def test_every_named_notebook_exists_and_reads_its_run():
    for name in runs.load()["notebooks"]:
        nb = NOTEBOOKS / f"{name}.ipynb"
        assert nb.is_file(), f"runs.yaml names {name}, but there is no {nb.name}"
        code = "".join("".join(c["source"]) for c in json.loads(nb.read_text())["cells"])
        assert f"active('{name}')" in code, f"{nb.name} doesn't read its run from runs.yaml"


def test_active_gives_the_notebook_and_its_phase():
    run, phase = active("kaggle_pretrain_tpu")
    assert run["mode"] in runs.MODES and run["start"] in runs.STARTS
    assert phase == runs.load()["phases"][run["phase"]]


def _write(tmp_path, change) -> Path:
    data = yaml.safe_load(runs.RUNS_FILE.read_text())
    change(data)
    p = tmp_path / "runs.yaml"
    p.write_text(yaml.dump(data))
    (tmp_path / "configs").mkdir(exist_ok=True)
    for ph in data["phases"].values():  # the configs it names
        (tmp_path / "configs" / ph["config"]).write_text("{}")
    return p


@pytest.mark.parametrize("change, match", [
    (lambda d: d["notebooks"]["kaggle_pretrain_tpu"].update(phase="edu3"), "unknown phase"),
    (lambda d: d["notebooks"]["kaggle_pretrain_tpu"].update(mode="Train"), "mode must be"),
    (lambda d: d["notebooks"]["kaggle_pretrain_tpu"].update(start="yes"), "start must be"),
    (lambda d: d["notebooks"]["kaggle_pretrain_tpu"].update(steps_per_session=True), "wrong type"),
    (lambda d: d["notebooks"]["kaggle_pretrain_tpu"].pop("tpu_spmd"), "missing"),
    (lambda d: d["notebooks"]["kaggle_pretrain_tpu"].update(start_run=True), "unknown"),
    (lambda d: d["phases"]["edu2"].update(budget="nope"), "unknown budget"),
])
def test_mistakes_fail_on_the_cpu_not_on_kaggle(tmp_path, change, match):
    with pytest.raises(RunsError, match=match):
        runs.load(_write(tmp_path, change), tmp_path / "configs")


def test_a_config_that_isnt_there_is_caught(tmp_path):
    p = _write(tmp_path, lambda d: None)
    (tmp_path / "configs" / "transformer_tpu_large_edu2_spmd.yaml").unlink()
    with pytest.raises(RunsError, match="no config"):
        runs.load(p, tmp_path / "configs")


def test_an_unknown_notebook_is_named(tmp_path):
    with pytest.raises(RunsError, match="no notebook"):
        active("kaggle_nope")
