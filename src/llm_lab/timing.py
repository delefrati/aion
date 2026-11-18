"""Where a session's time goes: one JSONL line per timed stage.

The session runner names the file in $AION_TIMINGS and passes it to the trainer's
subprocess, so both write to it:

    session  setup (notebook start -> session start), restore, data, train, push, bench
    trainer  trainer_setup (import -> first step starts), first_step (compile + one step),
             eval, save

With $AION_TIMINGS unset, record() does nothing, so the trainer runs the same outside a
session. summary() turns the file into the table printed at the end of a session.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ENV = "AION_TIMINGS"
_lock = threading.Lock()  # the session's background banker records pushes from its thread


def record(stage: str, seconds: float, **info) -> None:
    path = os.environ.get(ENV)
    if not path:
        return
    line = json.dumps({"stage": stage, "s": round(seconds, 2), "t": round(time.time(), 1), **info})
    with _lock, open(path, "a") as f:
        f.write(line + "\n")


@contextmanager
def stage(name: str, **info):
    """Time the block as `name`, also when it raises (recorded with ok=False)."""
    t = time.time()
    ok = False
    try:
        yield
        ok = True
    finally:
        record(name, time.time() - t, **info, **({} if ok else {"ok": False}))


def read(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def summary(events: list[dict]) -> str:
    """One row per stage, in the order each first happened: count, total, mean, max."""
    rows: dict[str, list[float]] = {}
    for e in events:
        rows.setdefault(e["stage"], []).append(e["s"])
    if not rows:
        return "timings: nothing recorded"
    out = [f"{'stage':<14}{'n':>4}{'total':>10}{'mean':>10}{'max':>10}"]
    for name, s in rows.items():
        out.append(f"{name:<14}{len(s):>4}{_fmt(sum(s)):>10}{_fmt(sum(s) / len(s)):>10}{_fmt(max(s)):>10}")
    return "\n".join(out)


def _fmt(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"
