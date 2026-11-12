"""Checkpoint files: safetensors weights, a state.json that describes them, optimizer apart.

A checkpoint is three files named after its step, in the run's checkpoint directory `d`:

    d/step_N.safetensors   model weights; a tensor shared by two names (tied embeddings) is
                           stored once, and the header maps the other name to it
    d/step_N.state.json    step, tokens seen, the full training config, tokenizer hash,
                           data id and parent run: enough to rebuild the model with no YAML
    d/step_N.optim.pt      optimizer + scheduler state; absent for a weights-only checkpoint

Two markers name checkpoints: d/COMPLETE the latest, d/BEST the best by val loss. Each holds
{"step": N, "checkpoint": "step_N"}. A save writes every file to a temporary name, renames
it into place, and only then rewrites the marker, the same way. A process killed mid-save
leaves the marker on the previous checkpoint, whose files the save never touched; anything
that reads or uploads "the latest checkpoint" goes through the marker.

A run's metrics aren't part of any checkpoint: they go to metrics.jsonl (training/metrics.py).

Checkpoints written before this format are one pickle, d/latest.pt (or best.pt, step_N.pt),
holding "model", "step", "optimizer", "scheduler" and "metrics_log". Every reader here still
accepts them; they carry no config, so building a model from one needs a TrainConfig.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

FORMAT = 1
WEIGHTS, STATE, OPTIM, LEGACY = ".safetensors", ".state.json", ".optim.pt", ".pt"
MARKERS = {"latest": "COMPLETE", "best": "BEST"}
TMP = ".tmp"


def strip_prefixes(state_dict: dict) -> dict:
    """Drop FSDPv2's '_orig_module.', DataParallel's 'module.' and torch.compile's '_orig_mod.'
    key prefixes. '_orig_module.' goes first: it contains 'module.'."""
    return {k.replace("_orig_module.", "").replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in state_dict.items()}


def aliases(model) -> dict[str, str]:
    """{alias: kept} for parameters registered under two names (tied embeddings).

    Read from the live module, not from tensor storage: on TPU the tie breaks when the model
    moves to the XLA device, and after the copy to CPU the two names would look tied or
    untied by accident. The first name met is the one kept.
    """
    seen: dict[int, str] = {}
    out: dict[str, str] = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        name = strip_prefixes({name: None}).popitem()[0]
        if id(p) in seen:
            out[name] = seen[id(p)]
        else:
            seen[id(p)] = name
    return out


def file_sha256(path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Ref:
    """Where one checkpoint lives: `dir/name.*` plus the format it was written in."""
    dir: Path
    name: str
    legacy: bool

    @property
    def weights(self) -> Path:
        return self.dir / (self.name + (LEGACY if self.legacy else WEIGHTS))

    @property
    def state(self) -> Path:
        return self.dir / (self.name + STATE)

    @property
    def optim(self) -> Path:
        return self.dir / (self.name + OPTIM)

    @property
    def files(self) -> list[Path]:
        """The checkpoint's files that exist."""
        if self.legacy:
            return [self.weights]
        return [p for p in (self.weights, self.state, self.optim) if p.is_file()]

    def __str__(self) -> str:
        return str(self.weights)


def step_name(step: int) -> str:
    return f"step_{step}"


def read_marker(directory, which: str = "latest") -> dict | None:
    """The COMPLETE (which='latest') or BEST marker: {"step", "checkpoint"}, or None."""
    p = Path(directory) / MARKERS[which]
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _named(d: Path, name: str) -> Ref | None:
    if (d / (name + STATE)).is_file() and (d / (name + WEIGHTS)).is_file():
        return Ref(d, name, legacy=False)
    return None


def find(directory, which: str = "latest") -> Ref | None:
    """The run's latest (or best) checkpoint: the one its marker names, else an old
    latest.pt (best.pt); None if there is neither."""
    d = Path(directory)
    marker = read_marker(d, which)
    if marker is not None:
        ref = _named(d, marker["checkpoint"])
        if ref is None:
            raise FileNotFoundError(f"{d / MARKERS[which]} names {marker['checkpoint']}, "
                                    f"whose files are missing")
        return ref
    if (d / (which + LEGACY)).is_file():
        return Ref(d, which, legacy=True)
    return None


def locate(path) -> Ref:
    """Resolve what a user passed: a checkpoint directory (its latest), any one of a
    checkpoint's files, its bare stem (dir/step_N), or an old .pt file."""
    p = Path(path)
    if p.is_dir():
        ref = find(p)
        if ref is None:
            raise FileNotFoundError(f"No latest checkpoint in {p}")
        return ref
    for suffix in (STATE, WEIGHTS, OPTIM):
        if p.name.endswith(suffix):
            ref = _named(p.parent, p.name[: -len(suffix)])
            if ref is None:
                raise FileNotFoundError(f"{p}: the checkpoint's {WEIGHTS} or {STATE} is missing")
            return ref
    if p.suffix == LEGACY and p.is_file():
        return Ref(p.parent, p.stem, legacy=True)
    ref = _named(p.parent, p.name)
    if ref is None:
        raise FileNotFoundError(f"No checkpoint at {p}")
    return ref


def _legacy(ref: Ref) -> dict:
    # mmap: an old latest.pt also carries ~1.9GB of optimizer state nobody may need.
    return torch.load(ref.weights, map_location="cpu", weights_only=False, mmap=True)


def read_state(ref: Ref) -> dict:
    """The checkpoint's state.json; for an old .pt, what the pickle can tell (no config)."""
    if not ref.legacy:
        return json.loads(ref.state.read_text())
    ckpt = _legacy(ref)
    return {
        "format": 0,
        "step": int(ckpt.get("step", 0)),
        "tokens_seen": None,
        "config": None,
        "has_optimizer": ckpt.get("optimizer") is not None,
        "best_val_loss": ckpt.get("best_val_loss"),
        "source": ckpt.get("_source"),
    }


def read_weights(ref: Ref) -> dict[str, torch.Tensor]:
    """Model state dict on CPU, with aliased names restored and wrapper prefixes stripped."""
    if ref.legacy:
        return strip_prefixes(_legacy(ref)["model"])
    from safetensors import safe_open
    out = {}
    with safe_open(str(ref.weights), framework="pt", device="cpu") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
        meta = f.metadata() or {}
    for alias, kept in json.loads(meta.get("aliases", "{}")).items():
        out[alias] = out[kept]
    return out


def read_training(ref: Ref) -> dict:
    """{"optimizer", "scheduler"}, each None when the checkpoint has none. An old .pt also
    gives "metrics_log", the history it carried inside it."""
    if ref.legacy:
        ckpt = _legacy(ref)
        return {"optimizer": ckpt.get("optimizer"), "scheduler": ckpt.get("scheduler"),
                "metrics_log": ckpt.get("metrics_log", [])}
    out = {"optimizer": None, "scheduler": None}
    if ref.optim.is_file():
        out.update(torch.load(ref.optim, map_location="cpu", weights_only=False))
    return out


def _atomic(path: Path, write) -> None:
    """write(tmp_path), then rename over `path`: a reader sees the old file or the new one."""
    tmp = path.with_name(path.name + TMP)
    write(tmp)
    os.replace(tmp, path)


def write(directory, weights: dict, alias_map: dict, state: dict, training: dict | None,
          mark: tuple[str, ...] = ("latest",)) -> Ref:
    """Write the checkpoint for state["step"] from CPU tensors, then point the markers in
    `mark` ("latest" -> COMPLETE, "best" -> BEST) at it. `training` = {"optimizer",
    "scheduler"}, or None for weights only.

    Writing a step that already has files replaces each one atomically. Within a run that is
    the same weights again (a best and a latest at the same step); the later steps of an
    abandoned branch are pruned when the run resumes, before anything rewrites them.
    """
    from safetensors.torch import save_file
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    ref = Ref(d, step_name(state["step"]), legacy=False)
    # Names that share one storage are tied too: an old .pt from a tied model unpickles that
    # way. (A TPU run's copies don't share, so they stay untied, as they trained.)
    alias_map = dict(alias_map)
    seen: dict[tuple, str] = {}
    for k, v in weights.items():
        if k in alias_map or v.numel() == 0:
            continue
        key = (v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape), tuple(v.stride()))
        if key in seen:
            alias_map[k] = seen[key]
        else:
            seen[key] = k
    kept = {k: v.detach().contiguous() for k, v in weights.items() if k not in alias_map}
    _atomic(ref.weights, lambda p: save_file(
        kept, str(p), metadata={"format": str(FORMAT), "aliases": json.dumps(alias_map)}))
    if training is not None:
        _atomic(ref.optim, lambda p: torch.save(training, p))
    # A weights-only write of a step that has an optimizer file (a best at a step just saved
    # as latest) keeps it: same step, same weights.
    if ref.state.is_file() and "best_val_loss" not in state:
        # ...and a latest written over a best keeps what made it the best.
        prev = json.loads(ref.state.read_text())
        if prev.get("best_val_loss") is not None:
            state = {**state, "best_val_loss": prev["best_val_loss"]}
    state = {"format": FORMAT, **state, "has_optimizer": ref.optim.is_file(),
             "tied": alias_map, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _atomic(ref.state, lambda p: p.write_text(json.dumps(state, indent=2)))
    for which in mark:
        set_marker(d, which, state["step"])
    return ref


def set_marker(directory, which: str, step: int) -> None:
    d = Path(directory)
    body = json.dumps({"step": step, "checkpoint": step_name(step)})
    _atomic(d / MARKERS[which], lambda p: p.write_text(body))


def clear_marker(directory, which: str) -> None:
    (Path(directory) / MARKERS[which]).unlink(missing_ok=True)


def steps_on_disk(directory) -> list[int]:
    """Steps that have a checkpoint (new format or an old step_N.pt), ascending."""
    out = set()
    for p in Path(directory).glob("step_*"):
        head = p.name.split(".")[0]
        if head[5:].isdigit() and not p.name.endswith(TMP):
            out.add(int(head[5:]))
    return sorted(out)


def prune(directory, keep_last: int) -> list[int]:
    """Delete checkpoints nothing needs: keep the marked ones and the newest `keep_last`
    (at least the latest) at or before the latest; drop later ones and a BEST past the
    latest (an abandoned branch after a rollback), old .pt files a marker has replaced, and
    temp files a killed save left. Returns the steps removed.

    Call it right after a latest save, or before training resumes: then no best can be
    newer than the latest except one from an abandoned branch."""
    d = Path(directory)
    for p in d.glob(f"*{TMP}"):
        p.unlink()
    latest = read_marker(d, "latest")
    if latest is None:
        return []
    keep = {latest["step"]}
    best = read_marker(d, "best")
    if best is not None and best["step"] > latest["step"]:
        clear_marker(d, "best")  # found by an abandoned branch the weights never reached
    elif best is not None:
        keep.add(best["step"])
    older = [s for s in steps_on_disk(d) if s <= latest["step"]]
    keep.update(older[-max(1, keep_last):])
    removed = [s for s in steps_on_disk(d) if s not in keep]
    for s in removed:
        for p in d.glob(f"{step_name(s)}.*"):
            p.unlink()
    for which in MARKERS:  # the old single-file checkpoint each marker replaced
        if read_marker(d, which) is not None:
            (d / (which + LEGACY)).unlink(missing_ok=True)
    return removed


def banked(directory) -> tuple[int, list[Path]] | None:
    """(step, files) of the run's latest complete checkpoint: what an uploader may push.

    The files are the marked checkpoints' (latest and best) plus the markers. None while no
    save has completed. A file a concurrent prune removes can vanish after this returns;
    copy or hard-link before uploading, and call again if one is gone.
    """
    d = Path(directory)
    latest = read_marker(d, "latest")
    if latest is None:
        return None
    files = []
    for which in MARKERS:
        ref = find(d, which) if read_marker(d, which) else None
        if ref is not None:
            files += [f for f in ref.files if f not in files]
            files.append(d / MARKERS[which])
    state = read_state(Ref(d, latest["checkpoint"], legacy=False))
    if state["step"] != latest["step"]:
        raise RuntimeError(f"{d}: COMPLETE says step {latest['step']} but its state.json "
                           f"says {state['step']}")
    return latest["step"], files


def config_from_state(state: dict, cfg=None):
    """The TrainConfig to build this checkpoint's model with.

    `cfg` (from a YAML the caller passed) wins; an old .pt has no other source. A state.json
    config also gets tie_embeddings from the weights as saved: a TPU run's config says True,
    but its saved embedding and lm_head are separate tensors, and building it tied would
    load one over the other.
    """
    from llm_lab.training.config import TrainConfig
    if cfg is None:
        if not state.get("config"):
            raise ValueError("This is an old .pt checkpoint, which doesn't record its config: "
                             "pass the training YAML (--config).")
        cfg = TrainConfig.from_dict(state["config"], strict=False)  # saved by some version
        if "tied" in state and cfg.model.model_type == "transformer":
            cfg.model.tie_embeddings = bool(state["tied"])
    return cfg


def load_model(path, device="cpu", cfg=None):
    """Build a checkpoint's model and load its weights. Returns (model in eval mode, state).

    `path` is anything `locate` takes. `cfg` is needed only for an old .pt checkpoint.
    """
    from llm_lab.training.model_factory import build_model
    ref = locate(path)
    state = read_state(ref)
    model = build_model(config_from_state(state, cfg))
    model.load_state_dict(read_weights(ref))
    return model.to(device).eval(), state


def tokenizer_path(path, override=None) -> Path:
    """The tokenizer for a checkpoint: `override`, else tokenizer.json beside the checkpoint,
    else the path its config trained with. Warns when the file's hash isn't the recorded one."""
    ref = locate(path)
    state = read_state(ref)
    candidates = [override] if override else [ref.dir / "tokenizer.json",
                                              (state.get("config") or {}).get("tokenizer_path")]
    for c in candidates:
        if c and Path(c).is_file():
            want = state.get("tokenizer_sha256")
            if want and file_sha256(c) != want:
                print(f"WARNING: {c} is not the tokenizer this checkpoint trained with "
                      f"(sha256 differs from its state.json).")
            return Path(c)
    raise FileNotFoundError(f"No tokenizer for {ref}: pass --tokenizer, or put tokenizer.json "
                            f"beside the checkpoint.")
