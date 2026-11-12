"""A run's metrics: metrics.jsonl, one JSON object per line, next to its checkpoints.

The file is only ever appended to. Training appends a line per log and eval as they happen,
and a resume appends {"resume_from": step}. A resume can go back: a checkpoint can be
rolled back, or a session can log past the last checkpoint it banked and then die. So a
reader replays the file and, at each resume line, drops what it has for later steps.

The file is history, not progress: how far a run has got is its checkpoint's state.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

FILE = "metrics.jsonl"


def append(path, entries) -> None:
    """Append entries (dicts), one line each. A line cut short by a kill is ended first, so
    it can't run into the next one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ""
    if path.is_file() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            head = "" if f.read(1) == b"\n" else "\n"
    with open(path, "a") as f:
        f.write(head + "".join(json.dumps(e) + "\n" for e in entries))


def mark_resume(path, step: int) -> None:
    append(path, [{"resume_from": step, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}])


def read(path) -> list[dict]:
    """The run's history as of its last resume: entries in order, none from abandoned steps."""
    path = Path(path)
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue  # the tail of a line cut short by a kill
        if "resume_from" in e:
            out = [x for x in out if x.get("step", 0) <= e["resume_from"]]
        else:
            out.append(e)
    return out
