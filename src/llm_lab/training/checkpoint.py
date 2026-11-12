"""Checkpoint files: safetensors weights, a state.json that describes them, optimizer apart.

A checkpoint called `name` in a directory `d` is three files:

    d/name.safetensors   model weights; a tensor shared by two names (tied embeddings) is
                         stored once, and the header maps the other name to it
    d/name.state.json    step, tokens seen, the full training config, tokenizer hash,
                         data id and parent run: enough to rebuild the model with no YAML
    d/name.optim.pt      optimizer + scheduler state; absent for a weights-only checkpoint

Checkpoints written before this format are one pickle, `d/name.pt`, holding "model",
"step", "optimizer", "scheduler" and "metrics_log". Every reader here still accepts them;
they carry no config, so building a model from one needs a TrainConfig from the caller.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

FORMAT = 1
WEIGHTS, STATE, OPTIM, LEGACY = ".safetensors", ".state.json", ".optim.pt", ".pt"


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
    """Where one checkpoint lives: `dir/name` plus the format it was written in."""
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

    def __str__(self) -> str:
        return str(self.weights)


def find(directory, name: str = "latest") -> Ref | None:
    """The checkpoint `name` in `directory`, new format first; None if there is none."""
    d = Path(directory)
    if (d / (name + STATE)).is_file() and (d / (name + WEIGHTS)).is_file():
        return Ref(d, name, legacy=False)
    if (d / (name + LEGACY)).is_file():
        return Ref(d, name, legacy=True)
    return None


def locate(path) -> Ref:
    """Resolve what a user passed: a checkpoint directory (its latest), any one of a new
    checkpoint's files, its bare stem, or an old .pt file."""
    p = Path(path)
    if p.is_dir():
        ref = find(p)
        if ref is None:
            raise FileNotFoundError(f"No latest checkpoint in {p}")
        return ref
    for suffix in (STATE, WEIGHTS, OPTIM):
        if p.name.endswith(suffix):
            ref = find(p.parent, p.name[: -len(suffix)])
            if ref is None or ref.legacy:
                raise FileNotFoundError(f"{p}: the checkpoint's {WEIGHTS} or {STATE} is missing")
            return ref
    if p.suffix == LEGACY and p.is_file():
        return Ref(p.parent, p.stem, legacy=True)
    ref = find(p.parent, p.name)
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
    """{"optimizer", "scheduler", "metrics_log"}: each None/[] when the checkpoint has none."""
    if ref.legacy:
        ckpt = _legacy(ref)
        return {"optimizer": ckpt.get("optimizer"), "scheduler": ckpt.get("scheduler"),
                "metrics_log": ckpt.get("metrics_log", [])}
    out = {"optimizer": None, "scheduler": None, "metrics_log": []}
    if ref.optim.is_file():
        out.update(torch.load(ref.optim, map_location="cpu", weights_only=False))
    return out


def write(directory, name: str, weights: dict, alias_map: dict, state: dict,
          training: dict | None) -> Ref:
    """Write a checkpoint from CPU tensors. `training` = {"optimizer", "scheduler", ...} or
    None for weights only (any optimizer file left by an earlier save of `name` goes)."""
    from safetensors.torch import save_file
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
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
    save_file(kept, str(d / (name + WEIGHTS)),
              metadata={"format": str(FORMAT), "aliases": json.dumps(alias_map)})
    optim = d / (name + OPTIM)
    if training is not None:
        torch.save(training, optim)
    elif optim.exists():
        optim.unlink()
    state = {"format": FORMAT, **state, "has_optimizer": training is not None,
             "tied": alias_map, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (d / (name + STATE)).write_text(json.dumps(state, indent=2))
    return Ref(d, name, legacy=False)


def remove(ref: Ref) -> None:
    for p in ((ref.weights,) if ref.legacy else (ref.weights, ref.state, ref.optim)):
        p.unlink(missing_ok=True)


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
        cfg = TrainConfig.from_dict(state["config"])
        if "tied" in state and cfg.model_type == "transformer":
            cfg.tie_embeddings = bool(state["tied"])
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
