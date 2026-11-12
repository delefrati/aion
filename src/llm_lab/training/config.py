"""Training configuration: four sections, every key known.

    model     what is built: architecture and its shape
    data      what it reads and writes: datasets, tokenizer, checkpoint directory
    schedule  how training proceeds: batch, optimizer, LR schedule, cadence, stopping
    device    how it runs on the hardware: GPU/TPU launch, memory and speed knobs

Code reads cfg.<section>.<key>. A YAML file may be flat (`d_model: 1024`) or nested
(`model: {d_model: 1024}`), or mix the two; a flat key wins over the same key in a section.
Every key name is unique across sections, which is what lets flat files and the notebooks'
flat overrides keep working.

An unknown key is an error that names it: a misspelled key used to print a warning and be
ignored, which once hid for months that `tpu_cores` didn't exist. Configs saved inside
checkpoints are read with strict=False, since a later version may have dropped a key.
"""
from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass, field, fields, replace as _replace
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    model_type: str = "mamba"  # "mamba" | "transformer"
    vocab_size: int = 4096
    d_model: int = 256
    n_layers: int = 4
    seq_len: int = 256
    # mamba-specific
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    # transformer-specific
    n_heads: int = 4
    tie_embeddings: bool = True  # share lm_head with input embedding. torch_xla breaks the tie on .to(xla), so TPU checkpoints are effectively untied; their state.json records that
    dropout: float = 0.0  # transformer residual/attention/MLP dropout; train-time only (eval()/generate disable it)


@dataclass
class DataConfig:
    dataset_type: str = "text"  # "text" (pretraining) | "instruction" | "chat"
    train_path: str = ""
    val_path: str = ""
    val_old_path: str = ""  # optional second text val set (e.g. the previous phase's), logged as val_old_loss
    tokenizer_path: str = ""
    instruction_data: str = ""  # instruction/chat runs: the JSON examples (10% held out as val)
    pretrain_checkpoint: str = ""  # finetune: where `cli finetune` seeds from
    data_id: str = ""  # where the data came from (e.g. a Kaggle Dataset); recorded in state.json
    checkpoint_dir: str = "checkpoints"


@dataclass
class ScheduleConfig:
    batch_size: int = 16  # per process; with tpu_spmd the GLOBAL micro-batch
    grad_accum_steps: int = 1  # effective batch = batch_size * grad_accum_steps * processes
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 5000
    warmup_steps: int = 200
    lr_schedule: str = "cosine"  # "cosine" | "wsd" (warmup-stable-decay: flat at lr until the
    #                              last wsd_decay_steps before lr_decay_steps, then 1-sqrt to 0;
    #                              while still flat, the run extends by raising lr_decay_steps)
    lr_decay_steps: int = 0  # decay horizon; 0 = max_steps. The session runner sets it to the
    #                          run's budget, so a session's max_steps doesn't end the schedule.
    wsd_decay_steps: int = 0  # wsd only: length of the final decay window
    seq_curriculum: str = ""  # sequence length curriculum e.g. "256:1000,512:2000,1024:3000"
    reset_optimizer: bool = False  # skip restoring optimizer/scheduler from checkpoint (use config lr)
    seed: int = 42
    max_train_minutes: float = 0  # >0: stop at the first step past this many minutes of training,
    #                               save, and return (single-process / SPMD only); 0 = off
    checkpoint_every: int = 500
    checkpoint_keep_last: int = 2  # checkpoints to keep, the latest included (0 counts as 1); the best is kept too
    eval_every: int = 250
    log_every: int = 50
    max_eval_batches: int = 0  # 0 = all. Text val takes this many batches of non-overlapping
    #                            windows spread evenly over the whole val file
    # finetuning only (pretraining is single-pass: no best, no early stop)
    early_stop_patience: int = 0  # stop after this many evals with no val_loss improvement; 0 = off
    early_stop_min_delta: float = 0.0  # min val_loss decrease to count as improvement


@dataclass
class DeviceConfig:
    gpus: int = 1  # CUDA GPUs to use: 1 = single; 0 = all visible; N = N via DistributedDataParallel
    tpu_cores: int = 1  # TPU cores: 1 = single core (in-process); anything else (0 or 8) fans
    #                     out to EVERY core via xmp.spawn. cli.cmd_train branches on != 1.
    tpu_spmd: bool = False  # with tpu_cores != 1: ONE process drives every chip via SPMD (one
    #                         compile, batch sharded on a 'data' mesh axis) instead of one
    #                         xmp.spawn process per core. batch_size must divide by the chips.
    tpu_fsdp: bool = False  # with tpu_spmd: shard params, grads and optimizer state over the chips
    #                         (FSDPv2) instead of replicating ~3.8GB of them on every chip
    tpu_fuse_step: bool = False  # TPU: skip the flush after the LAST micro-step so it compiles into
    #                              one graph with the optimizer step (one fewer graph per step)
    flash_attention: bool = False  # TPU only: Pallas flash-attention kernel instead of SDPA
    grad_checkpoint: bool = False  # activation checkpointing (saves memory, allows larger batch)
    compile: bool = False  # torch.compile for ~25-40% speedup (requires Turing+ GPU)
    use_8bit_optim: bool = True  # 8-bit AdamW when bitsandbytes is present; False forces standard AdamW (needed to resume a standard-AdamW checkpoint on a GPU image that has bitsandbytes)
    foreach_optim: bool = True  # foreach AdamW is faster but allocates temp buffers for ALL params at once (a large transient spike); set False on tight VRAM
    num_workers: int = 2  # DataLoader workers; 0 avoids subprocess orphans on interrupted runs


SECTIONS = {"model": ModelConfig, "data": DataConfig, "schedule": ScheduleConfig,
            "device": DeviceConfig}
# key -> section; every key is unique across sections
SECTION_OF = {f.name: s for s, cls in SECTIONS.items() for f in fields(cls)}
assert len(SECTION_OF) == sum(len(fields(c)) for c in SECTIONS.values()), "a key is in two sections"


class ConfigError(ValueError):
    pass


def _coerce(cls, key: str, value):
    """YAML reads 5e-4 as a string; numeric fields take numbers."""
    kind = {f.name: f.type for f in fields(cls)}[key]
    if isinstance(value, str) and kind in ("float", "int"):
        return float(value) if kind == "float" else int(float(value))
    return value


def _unknown(key: str, where: str = "") -> str:
    near = difflib.get_close_matches(key, list(SECTION_OF) + list(SECTIONS), n=1)
    return f"unknown config key {key!r}{where}" + (f" (did you mean {near[0]!r}?)" if near else "")


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)

    @classmethod
    def from_dict(cls, data: dict, strict: bool = True) -> TrainConfig:
        """Build from a flat, nested or mixed dict. Unknown keys raise ConfigError (with
        strict=False they're reported and dropped: for configs saved by older versions)."""
        values: dict[str, dict] = {s: {} for s in SECTIONS}
        problems = []
        flat = {}
        for key, value in (data or {}).items():
            if key in SECTIONS and isinstance(value, dict):
                for k, v in value.items():
                    if SECTION_OF.get(k) == key:
                        values[key][k] = v
                    else:
                        problems.append(_unknown(k, f" in section {key!r}")
                                        + (f"; it belongs in {SECTION_OF[k]!r}" if k in SECTION_OF else ""))
            elif key in SECTION_OF:
                flat[key] = value
            else:
                problems.append(_unknown(key))
        for k, v in flat.items():  # a flat key wins over the same key in a section
            values[SECTION_OF[k]][k] = v
        if problems:
            if strict:
                raise ConfigError("; ".join(problems))
            print("WARNING: config keys ignored: " + "; ".join(problems))
        built = {s: SECTIONS[s](**{k: _coerce(SECTIONS[s], k, v) for k, v in vals.items()})
                 for s, vals in values.items()}
        return cls(**built)

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        try:
            return cls.from_dict(yaml.safe_load(Path(path).read_text()))
        except ConfigError as e:
            raise ConfigError(f"{path}: {e}") from None

    def to_dict(self) -> dict:
        """Nested: {"model": {...}, "data": {...}, "schedule": {...}, "device": {...}}."""
        return asdict(self)

    def to_flat(self) -> dict:
        return {k: v for section in self.to_dict().values() for k, v in section.items()}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False))

    def replace(self, **flat) -> TrainConfig:
        """A copy with flat keys changed: cfg.replace(max_steps=3, d_model=64)."""
        per: dict[str, dict] = {s: {} for s in SECTIONS}
        for k, v in flat.items():
            if k not in SECTION_OF:
                raise ConfigError(_unknown(k))
            per[SECTION_OF[k]][k] = v
        return TrainConfig(**{s: _replace(getattr(self, s), **per[s]) for s in SECTIONS})
