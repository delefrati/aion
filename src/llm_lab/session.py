"""One training session of a run: restore, seed, train, bank, push.

A notebook session used to do this in its own cells, and most of the lost time came from
there: checkpoints never pushed (43k-44k), a run restarted from scratch in silence, and
no-op re-pushes of restored files. This is the one implementation, with tests.

    spec = SessionSpec(name="edu2", store=KaggleStore(...), config=yaml_dict,
                       target_steps=70000, work_dir=Path("/tmp/aion"),
                       parent=KaggleStore(...), data=KaggleStore(...))
    result = run_session(spec)

What a session does:
1. Restore the run's store into work_dir/checkpoints. The run must already exist unless
   spec.start says this session may start it (seeded from spec.parent, or from scratch).
   start="auto" decides from the stores: resume if the run's store exists, else seed it
   from the parent, once the parent is readable with the same credential and a second
   lookup still finds the run missing. Fails closed: a store without a checkpoint, or a
   missing store the spec didn't say to start, stops the session before anything trains
   or pushes.
2. Restore the tokenized data (spec.data) into work_dir/data.
3. Train in a subprocess (`python -m llm_lab.cli train`), so every launch path the CLI knows
   (CPU, CUDA DDP, TPU SPMD) works the same way.
4. While it trains, bank each newly completed checkpoint (checkpoint.banked: only what a
   COMPLETE marker names, never a partial save) to the store in the background, so a hard
   timeout loses at most one checkpoint interval. A failed push is retried later and never
   recorded as done; the checkpoint is also mirrored to spec.fallback_dir, if set.
5. Push once more at the end, if this session got further than the store already has.
   A session that made no progress pushes nothing.
6. When the session reaches the budget, score the final checkpoint on the zero-shot
   benchmarks (spec.bench) and push bench.json with it. A pretraining phase keeps no best
   checkpoint, so this is how it's judged. If the session dies first, the next one (which
   finds the budget spent) runs the bench instead.

Every stage is timed (llm_lab.timing), the trainer's too: the session ends by printing where
its time went, and copies the raw timings.jsonl to spec.output_dir.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from llm_lab import timing
from llm_lab.storage import Store, StoreError
from llm_lab.training import checkpoint, metrics


class SessionError(RuntimeError):
    """The session stopped on purpose, before training, so nothing was changed."""


class PushError(RuntimeError):
    """The session trained, but its progress couldn't be pushed to the store."""


@dataclass
class SessionSpec:
    name: str                          # the run, for messages
    store: Store                       # where the run's checkpoints live between sessions
    config: dict                       # the training YAML, as loaded
    target_steps: int                  # the run's budget: no session trains past it
    work_dir: Path                     # local: checkpoints/, data/, run.yaml
    parent: Store | None = None        # the run's first session seeds from its latest checkpoint
    start: bool | str = False          # may start the run: True, False, or "auto" (see restore)
    data: Store | None = None          # tokenized cache: train.bin, val.bin, tokenizer.json
    steps: int | None = None           # steps this session; None trains to target_steps
    deadline: float | None = None      # time.time() by which training must stop (timed session)
    reserve_minutes: float = 20        # of a timed session, kept for the final eval/save/push
    lr_decay_to_target: bool = False   # schedule spans the budget (WSD runs that may extend)
    push: bool = True                  # False: a smoke test; nothing leaves the machine
    bank_every: float = 120            # seconds between background bank checks
    fallback_dir: Path | None = None   # mirror of the checkpoint when a push fails
    overrides: dict = field(default_factory=dict)  # last-word config changes for this session
    bench: tuple[str, ...] = ()        # tasks to score the final checkpoint on at the phase end
    bench_limit: int = 0               # questions per task; 0 = all
    bench_batch_size: int = 32
    output_dir: Path | None = None     # also gets bench.json and timings.jsonl (Kaggle: /kaggle/working)
    started_at: float | None = None    # time.time() the notebook started: its setup gets timed too
    python: str = sys.executable

    @property
    def ckpt_dir(self) -> Path:
        return Path(self.work_dir) / "checkpoints"

    @property
    def data_dir(self) -> Path:
        return Path(self.work_dir) / "data"

    @property
    def timings_file(self) -> Path:
        return Path(self.work_dir) / "timings.jsonl"


@dataclass
class SessionResult:
    start_step: int
    end_step: int            # the latest checkpoint on disk when the session ended
    pushed_step: int | None  # the latest step the store holds (None: never pushed this run)
    returncode: int          # the trainer's; 0 when there was nothing to train
    phase_done: bool         # end_step reached target_steps


# --- restore --------------------------------------------------------------------------------

def _latest_step(ckpt_dir: Path) -> int | None:
    ref = checkpoint.find(ckpt_dir)
    return None if ref is None else checkpoint.read_state(ref)["step"]


def restore(spec: SessionSpec) -> int:
    """Put the run's latest checkpoint in spec.ckpt_dir; return its step."""
    return _restore(spec)[0]


def _restore(spec: SessionSpec) -> tuple[int, bool]:
    """restore(), and whether the run's store existed (False: this session starts the run).

    The store's files replace whatever the directory held. A missing store is a new run
    only when spec.start says so; then the parent's latest weights become step 0 (or, with
    no parent, training starts from random init).

    start="auto" resumes a store that exists and seeds one that doesn't, but only from a
    parent: a random-init start must be asked for (start=True). Before seeding it checks
    that "missing" is a real answer and not a failed lookup: the parent must list with the
    same credential (a credential without access also answers 403), and a second listing
    of the run's store must still find it missing."""
    if spec.start not in (True, False, "auto"):
        raise ValueError(f"start must be True, False or 'auto', not {spec.start!r}")
    d = spec.ckpt_dir
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    files = spec.store.files()
    if files is not None:
        if spec.start is True:
            raise SessionError(f"{spec.store} already exists, but this session was told to start "
                               f"{spec.name}. Resume it instead (start=False); starting over "
                               f"would push step 0 over its progress.")
        spec.store.download(d)
        step = _latest_step(d)
        if step is None:
            raise SessionError(f"{spec.store} exists but holds no checkpoint "
                               f"(files: {sorted(files)}). Refusing to train from scratch over it.")
        print(f"{spec.name}: resuming from step {step:,} ({spec.store}).")
        return step, True
    if spec.start is False:
        raise SessionError(f"{spec.store} not found. If this is {spec.name}'s first session, "
                           f"start it explicitly (start=True); a lookup that wrongly reports "
                           f"'missing' must never seed a run over its real progress.")
    if spec.start == "auto":
        _confirm_missing(spec)
    if spec.parent is None:
        print(f"{spec.name}: new run, training from random init.")
        return 0, False
    _seed_from_parent(spec)
    return 0, False


def _confirm_missing(spec: SessionSpec) -> None:
    """start="auto" found no store: make sure that's the answer before seeding over it."""
    if spec.parent is None:
        raise SessionError(f"{spec.store} not found and {spec.name} has no parent. A run from "
                           f"random init must be started explicitly (start=True).")
    if spec.parent.files() is None:
        raise SessionError(f"{spec.store} not found, and neither is its parent {spec.parent}. "
                           f"Not starting {spec.name}: the credential may not see either.")
    if spec.store.files() is not None:
        raise SessionError(f"{spec.store} was missing, then found on a second look. Not seeding "
                           f"over it; run the session again to resume it.")
    print(f"{spec.name}: {spec.store} not found (twice) and its parent {spec.parent} is readable: "
          f"this is {spec.name}'s first session.")


def _seed_from_parent(spec: SessionSpec) -> None:
    """The parent's latest weights as this run's step 0 (fresh optimizer, no history)."""
    from llm_lab.training.config import TrainConfig
    from llm_lab.training.trainer import seed_checkpoint
    src = Path(spec.work_dir) / "parent"
    shutil.rmtree(src, ignore_errors=True)
    files = spec.parent.files()
    if files is None:
        raise SessionError(f"Parent {spec.parent} not found: nothing to seed {spec.name} from.")
    if checkpoint.MARKERS["latest"] in files:
        spec.parent.download(src, only=[checkpoint.MARKERS["latest"]])
        name = checkpoint.read_marker(src)["checkpoint"]
        need = [name + checkpoint.WEIGHTS, name + checkpoint.STATE]
    else:  # an old run: latest.pt
        need = ["latest.pt"]
    spec.parent.download(src, only=need + ["tokenizer.json"], required=need)
    ref = checkpoint.find(src)
    parent_step = checkpoint.read_state(ref)["step"]
    cfg = TrainConfig.from_dict({**spec.config, **spec.overrides})
    seed_checkpoint(ref.weights, spec.ckpt_dir, cfg=cfg,
                    parent={"store": str(spec.parent), "checkpoint": ref.weights.name, "step": parent_step})
    if (src / "tokenizer.json").exists():
        shutil.copy2(src / "tokenizer.json", spec.ckpt_dir / "tokenizer.json")
    shutil.rmtree(src)  # gigabytes back before the data cache lands
    print(f"{spec.name}: seeded from {spec.parent} {ref.weights.name} @ step {parent_step:,} "
          f"(step 0, fresh optimizer). The first push creates {spec.store}.")


DATA_FILES = ("train.bin", "val.bin", "tokenizer.json")


def fetch_data(spec: SessionSpec) -> None:
    if spec.data is None:
        return
    print(f"Free disk in {spec.work_dir}: {shutil.disk_usage(spec.work_dir).free / 1e9:.0f} GB")
    if spec.data.download(spec.data_dir, only=DATA_FILES + ("val_old.bin",),
                          required=DATA_FILES) is None:
        raise SessionError(f"Data {spec.data} not found.")


# --- config ---------------------------------------------------------------------------------

def session_config(spec: SessionSpec, start_step: int) -> dict | None:
    """The config this session trains with, or None when the budget is already spent."""
    if start_step >= spec.target_steps:
        return None
    cfg = dict(spec.config)
    cfg["checkpoint_dir"] = str(spec.ckpt_dir)
    if spec.data is not None:
        dd = spec.data_dir
        cfg.update(dataset_type="text", train_path=str(dd / "train.txt"), val_path=str(dd / "val.txt"),
                   tokenizer_path=str(dd / "tokenizer.json"), instruction_data="",
                   val_old_path=str(dd / "val_old.txt") if (dd / "val_old.bin").exists() else "",
                   data_id=str(spec.data))
    end = spec.target_steps if spec.steps is None else min(start_step + spec.steps, spec.target_steps)
    cfg["max_steps"] = end
    if spec.lr_decay_to_target or not cfg.get("lr_decay_steps"):
        # A session's max_steps isn't where the run ends. With lr_decay_steps unset the
        # schedule would decay to it, and a short session would take the LR to zero.
        cfg["lr_decay_steps"] = spec.target_steps
    if spec.deadline is not None:
        minutes = (spec.deadline - time.time()) / 60 - spec.reserve_minutes
        cfg["max_train_minutes"] = max(10.0, minutes)
    cfg.update(spec.overrides)
    return cfg


# --- banking --------------------------------------------------------------------------------

BENCH_FILE = "bench.json"
EXTRAS = ("tokenizer.json", "run.yaml", BENCH_FILE)
DEFAULT_BENCH = ("hellaswag", "arc_easy", "arc_challenge", "piqa")


def snapshot(ckpt_dir: Path, dest: Path, attempts: int = 5) -> int | None:
    """Hard-link the banked checkpoint files (plus tokenizer, run.yaml, a copy of
    metrics.jsonl) into dest. Returns the banked step, or None if nothing is banked yet.

    Links, not copies: the files are immutable once a marker names them, and a later save
    can prune them while an upload still reads them; a link keeps them alive. If one is
    pruned before it's linked, banked() is asked again.
    """
    for _ in range(attempts):
        b = checkpoint.banked(ckpt_dir)
        if b is None:
            return None
        step, files = b
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        try:
            for f in files + [ckpt_dir / n for n in EXTRAS if (ckpt_dir / n).exists()]:
                try:
                    os.link(f, dest / f.name)
                except OSError:
                    if not f.exists():
                        raise
                    shutil.copy2(f, dest / f.name)
        except FileNotFoundError:
            continue  # pruned under us: a newer checkpoint is banked now
        m = ckpt_dir / metrics.FILE
        if m.exists():
            shutil.copyfile(m, dest / m.name)  # still being appended to: copy what's there
        return step
    raise RuntimeError(f"{ckpt_dir}: the banked checkpoint kept changing while being linked")


class Banker:
    """Pushes the newest banked checkpoint when it's newer than the store's. Never records a
    push as done unless the store accepted it."""

    def __init__(self, spec: SessionSpec, pushed_step: int | None):
        self.spec = spec
        self.pushed = pushed_step
        self.lock = threading.Lock()
        self.stage = Path(spec.work_dir) / "push"
        self.fail_streak = 0
        self.retry_after = 0.0

    def bank(self, why: str, force: bool = False) -> bool:
        """Push if there is something newer (or, with force, the same step again, e.g. with
        bench.json added). True when the store is up to date afterwards."""
        with self.lock:
            step = snapshot(self.spec.ckpt_dir, self.stage)
            if step is None or (not force and self.pushed is not None and step <= self.pushed):
                return True
            print(f"[bank] {why}: pushing step {step:,} to {self.spec.store}...", flush=True)
            files = sorted(self.stage.iterdir())
            size_gb = round(sum(f.stat().st_size for f in files) / 1e9, 2)
            try:
                with timing.stage("push", step=step, gb=size_gb, why=why):
                    self.spec.store.push(files, f"step {step}")
            except StoreError as e:
                self.fail_streak += 1
                self.retry_after = time.time() + min(600 * self.fail_streak, 1800)
                print(f"[bank] step {step:,} NOT banked: {e}", flush=True)
                self._mirror(step)
                return False
            finally:
                shutil.rmtree(self.stage, ignore_errors=True)
            self.pushed, self.fail_streak, self.retry_after = step, 0, 0.0
            print(f"[bank] step {step:,} banked.", flush=True)
            return True

    def _mirror(self, step: int) -> None:
        """Last-ditch copy where the platform keeps output (Kaggle: /kaggle/working)."""
        dest = self.spec.fallback_dir
        if dest is None:
            return
        try:
            tmp = Path(self.spec.work_dir) / "mirror"
            if snapshot(self.spec.ckpt_dir, tmp) is None:
                return
            need = sum(p.stat().st_size for p in tmp.iterdir())
            Path(dest).mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(dest).free
            if free < need * 1.5:
                print(f"[bank] no room to mirror to {dest} ({need / 1e9:.1f}GB needed, "
                      f"{free / 1e9:.1f}GB free).")
                return
            for old in Path(dest).iterdir():
                if old.is_file():
                    old.unlink()
            for p in tmp.iterdir():
                shutil.copy2(p, Path(dest) / p.name)
            shutil.rmtree(tmp)
            print(f"[bank] mirrored step {step:,} to {dest}.")
        except Exception as e:  # the mirror must never take the session down
            print(f"[bank] mirror to {dest} failed: {e}")

    def watch(self, stop: threading.Event) -> None:
        while not stop.wait(self.spec.bank_every):
            if time.time() < self.retry_after:
                continue
            try:
                self.bank("background")
            except Exception as e:  # keep watching; the final push still runs
                print(f"[bank] skipped this cycle: {e}", flush=True)


# --- the phase-end bench ----------------------------------------------------------------------

def _bench_done(spec: SessionSpec, step: int) -> bool:
    f = spec.ckpt_dir / BENCH_FILE
    if not f.exists():
        return False
    done = json.loads(f.read_text())
    return done.get("step") == step and set(spec.bench) <= set(done.get("tasks", {}))


def run_bench(spec: SessionSpec) -> dict:
    """Score the latest checkpoint on spec.bench; write bench.json beside it (and in
    output_dir). CUDA if there is one, else the CPU: the TPU belongs to training."""
    import torch
    from llm_lab.eval.benchmarks import format_table, run_task
    from llm_lab.tokenizer.bpe import load_tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    ref = checkpoint.find(spec.ckpt_dir)
    model, state = checkpoint.load_model(ref.weights, device)
    seq_len = checkpoint.config_from_state(state).model.seq_len
    tokenizer = load_tokenizer(checkpoint.tokenizer_path(ref.weights))
    print(f"{spec.name}: phase done at step {state['step']:,}; benchmarking {ref.weights.name} "
          f"on {device} ({', '.join(spec.bench)})...", flush=True)
    result = {"run": spec.name, "checkpoint": ref.weights.name, "step": state["step"],
              "limit": spec.bench_limit, "tasks": {}}
    for task in spec.bench:
        result["tasks"][task] = run_task(model, tokenizer, task, seq_len, device,
                                         batch_size=spec.bench_batch_size, limit=spec.bench_limit)
    body = json.dumps(result, indent=2)
    (spec.ckpt_dir / BENCH_FILE).write_text(body)
    if spec.output_dir is not None:
        Path(spec.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(spec.output_dir) / BENCH_FILE).write_text(body)
    print(format_table({f"{spec.name}-{state['step']}": result}), flush=True)
    return result


def _bench_and_push(spec: SessionSpec, banker: "Banker") -> None:
    with timing.stage("bench"):
        run_bench(spec)
    if spec.push and not banker.bank("bench results", force=True):
        raise PushError(f"bench.json for step {banker.pushed} couldn't be pushed to {spec.store}.")


# --- the session ----------------------------------------------------------------------------

def train(spec: SessionSpec, cfg: dict, on_line=None) -> int:
    """Run `llm_lab.cli train` on cfg in a subprocess; stream its output; return its exit code."""
    run_yaml = spec.ckpt_dir / "run.yaml"
    run_yaml.write_text(yaml.dump(cfg))
    src_root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(src_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
           timing.ENV: str(spec.timings_file)}
    proc = subprocess.Popen([spec.python, "-u", "-m", "llm_lab.cli", "train", "--config", str(run_yaml)],
                            cwd=src_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if on_line:
                on_line(line, proc)
        return proc.wait()
    except BaseException:
        # Interrupted (a notebook "stop"): SIGINT lets the trainer save its last completed
        # step, which the final push in run_session then banks.
        import signal
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise


def run_session(spec: SessionSpec, on_line=None) -> SessionResult:
    """Restore, train, bank and push one session. See the module docstring."""
    Path(spec.work_dir).mkdir(parents=True, exist_ok=True)
    spec.timings_file.unlink(missing_ok=True)
    previous = os.environ.get(timing.ENV)
    os.environ[timing.ENV] = str(spec.timings_file)  # this process's stages (the banker's too)
    try:
        if spec.started_at is not None:
            timing.record("setup", time.time() - spec.started_at)
        return _run_session(spec, on_line)
    finally:
        if previous is None:
            os.environ.pop(timing.ENV, None)
        else:
            os.environ[timing.ENV] = previous
        _report_timings(spec)


def _report_timings(spec: SessionSpec) -> None:
    """Print where the session's time went; keep the raw lines in output_dir. Never raises."""
    try:
        events = timing.read(spec.timings_file)
        print(f"\n{spec.name}: where this session's time went\n{timing.summary(events)}", flush=True)
        if spec.output_dir is not None and spec.timings_file.exists():
            Path(spec.output_dir).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(spec.timings_file, Path(spec.output_dir) / spec.timings_file.name)
    except Exception as e:  # timings must never take the session down
        print(f"timings: couldn't report ({e})")


def _run_session(spec: SessionSpec, on_line=None) -> SessionResult:
    with timing.stage("restore"):
        start, existed = _restore(spec)
    # A seeded run's step 0 isn't in the store yet; a resumed run's start step is.
    pushed = start if existed else None
    cfg = session_config(spec, start)
    if cfg is None:
        print(f"{spec.name}: budget reached ({start:,}/{spec.target_steps:,} steps): nothing to train.")
        if spec.bench and spec.push and not _bench_done(spec, start):
            _bench_and_push(spec, Banker(spec, pushed))  # the session that finished died first
        return SessionResult(start, start, pushed, 0, True)
    with timing.stage("data"):
        fetch_data(spec)
    print(f"{spec.name}: training steps {start:,} -> {cfg['max_steps']:,} of {spec.target_steps:,}"
          + (f", for at most {cfg['max_train_minutes']:.0f} min" if spec.deadline else "")
          + ("" if spec.push else " (smoke test: nothing will be pushed)"), flush=True)

    banker = Banker(spec, pushed)
    stop = threading.Event()
    watcher = threading.Thread(target=banker.watch, args=(stop,), daemon=True)
    if spec.push:
        watcher.start()
    rc = 1
    try:
        with timing.stage("train", start_step=start):
            rc = train(spec, cfg, on_line)
    finally:
        stop.set()
        if watcher.is_alive():
            watcher.join(timeout=60)
        end = _latest_step(spec.ckpt_dir)
        end = start if end is None else end
        if not spec.push:
            print(f"Smoke test done at step {end:,}: nothing pushed.")
        elif end > start:
            if not banker.bank("final") and not banker.bank("final, retry"):
                print(f"[bank] FINAL PUSH FAILED: {spec.store} still holds step "
                      f"{banker.pushed if banker.pushed is not None else 'none'}, so the next "
                      f"session resumes behind step {end:,}"
                      + (f". A copy is in {spec.fallback_dir}." if spec.fallback_dir else "."))
        else:
            print(f"No progress this session (still step {end:,}): nothing pushed; "
                  f"{spec.store} keeps what it had.")
    result = SessionResult(start, end, banker.pushed, rc, end >= spec.target_steps)
    if spec.push and end > start and (banker.pushed is None or banker.pushed < end):
        raise PushError(f"Step {end:,} was trained but not pushed; {spec.store} holds "
                        f"{banker.pushed}." + (f" Copy in {spec.fallback_dir}." if spec.fallback_dir else ""))
    if rc != 0:
        raise RuntimeError(f"Training exited with code {rc} at step {end:,} "
                           f"(store holds {result.pushed_step}).")
    if result.phase_done and spec.bench and spec.push and not _bench_done(spec, end):
        _bench_and_push(spec, banker)
    return result
