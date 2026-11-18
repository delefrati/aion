"""Which run each notebook executes, read from runs.yaml beside this file (AION-3).

    run, phase = active("kaggle_pretrain_tpu")
    run["mode"], run["start"], phase["ckpt"], phase["config"], ...

The file is checked here, on the CPU, so a typo fails the test suite instead of a queued
Kaggle session.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from llm_lab.run_config import BUDGETS

CONFIGS = Path(__file__).resolve().parent / "configs"
RUNS_FILE = Path(__file__).resolve().parent / "runs.yaml"

MODES = ("train", "smoke", "sweep")
STARTS = (True, False, "auto")
NOTEBOOK_KEYS = {"phase": str, "mode": str, "start": (bool, str), "smoke_steps": int,
                 "session_hours": (int, float), "session_reserve_min": (int, float),
                 "steps_per_session": int, "tpu_spmd": bool}
PHASE_KEYS = {"budget": str, "ckpt": str, "parent": (str, type(None)), "data": str,
              "config": str, "timed": bool, "title": str}


class RunsError(ValueError):
    """runs.yaml names something that doesn't exist, or leaves out a setting."""


def load(path: Path = RUNS_FILE, configs: Path = CONFIGS) -> dict:
    """The whole file, checked. Phase configs are looked up in `configs`."""
    runs = yaml.safe_load(Path(path).read_text())
    phases = runs.get("phases") or {}
    for name, ph in phases.items():
        _check_keys(f"phase {name!r}", ph, PHASE_KEYS)
        if ph["budget"] not in BUDGETS:
            raise RunsError(f"phase {name!r}: unknown budget {ph['budget']!r}")
        if not (Path(configs) / ph["config"]).is_file():
            raise RunsError(f"phase {name!r}: no config {ph['config']!r} in {configs}")
    for name, nb in (runs.get("notebooks") or {}).items():
        _check_keys(f"notebook {name!r}", nb, NOTEBOOK_KEYS)
        if nb["phase"] not in phases:
            raise RunsError(f"notebook {name!r}: unknown phase {nb['phase']!r} "
                            f"(known: {sorted(phases)})")
        if nb["mode"] not in MODES:
            raise RunsError(f"notebook {name!r}: mode must be one of {MODES}, not {nb['mode']!r}")
        if nb["start"] not in STARTS:
            raise RunsError(f"notebook {name!r}: start must be auto, true or false, "
                            f"not {nb['start']!r}")
    return runs


def active(notebook: str, path: Path = RUNS_FILE) -> tuple[dict, dict]:
    """(the notebook's settings, its phase), from runs.yaml."""
    runs = load(path)
    notebooks = runs.get("notebooks") or {}
    if notebook not in notebooks:
        raise RunsError(f"{path} has no notebook {notebook!r} (known: {sorted(notebooks)})")
    nb = notebooks[notebook]
    return dict(nb), dict(runs["phases"][nb["phase"]])


def _check_keys(what: str, d, keys: dict) -> None:
    if not isinstance(d, dict):
        raise RunsError(f"{what}: expected a mapping, got {d!r}")
    missing = sorted(set(keys) - set(d))
    unknown = sorted(set(d) - set(keys))
    if missing or unknown:
        raise RunsError(f"{what}: " + "; ".join(
            ([f"missing {missing}"] if missing else []) + ([f"unknown {unknown}"] if unknown else [])))
    for k, t in keys.items():
        # bool is an int in Python: an int setting must not quietly accept true/false
        if not isinstance(d[k], t) or (isinstance(d[k], bool) and bool not in _types(t)):
            raise RunsError(f"{what}: {k} = {d[k]!r} has the wrong type")


def _types(t) -> tuple:
    return t if isinstance(t, tuple) else (t,)
