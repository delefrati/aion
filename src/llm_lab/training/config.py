"""Training configuration with reproducible defaults."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    # model
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

    # training
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 5000
    warmup_steps: int = 200
    lr_decay_steps: int = 0  # cosine decay horizon; 0 = use max_steps. Set to the FULL
    #                          multi-session target so per-session max_steps doesn't zero the LR.
    grad_clip: float = 1.0
    grad_accum_steps: int = 1  # gradient accumulation (effective batch = batch_size * grad_accum_steps)
    grad_checkpoint: bool = False  # activation checkpointing (saves VRAM, allows larger batch)
    compile: bool = False  # torch.compile for ~25-40% speedup (requires Turing+ GPU)
    num_workers: int = 2  # DataLoader workers; set 0 to avoid subprocess orphans on interrupted runs
    seq_curriculum: str = ""  # sequence length curriculum e.g. "256:1000,512:2000,1024:3000"
    reset_optimizer: bool = False  # skip restoring optimizer/scheduler from checkpoint (use config lr)
    use_8bit_optim: bool = True  # 8-bit AdamW when bitsandbytes is present; set False to force standard AdamW (needed to resume a standard-AdamW checkpoint on a GPU image that has bitsandbytes)
    tie_embeddings: bool = True  # share lm_head with input embedding. NOTE: torch_xla breaks the tie on .to(xla), so TPU checkpoints are effectively UNTIED — set False to resume such a checkpoint on GPU.
    gpus: int = 1  # CUDA GPUs to use: 1 = single; 0 = all visible; N = N via DistributedDataParallel
    foreach_optim: bool = True  # foreach AdamW is faster but allocates temp buffers for ALL params at once (a large transient spike); set False on tight VRAM
    seed: int = 42

    # checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 500
    checkpoint_keep_last: int = 2  # numbered step_*.pt archives to keep; 0 = only latest.pt+best.pt (saves ~keep_last*ckpt_size of disk)
    eval_every: int = 250
    log_every: int = 50
    max_eval_batches: int = 0  # 0 = all

    # early stopping (0 patience = disabled)
    early_stop_patience: int = 0  # stop after this many evals with no val_loss improvement
    early_stop_min_delta: float = 0.0  # min val_loss decrease to count as improvement

    # data
    train_path: str = ""
    val_path: str = ""
    tokenizer_path: str = ""

    # instruction tuning
    pretrain_checkpoint: str = ""
    instruction_data: str = ""
    dataset_type: str = "text"  # "text" | "instruction" | "chat"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(asdict(self), default_flow_style=False))

    @classmethod
    def from_dict(cls, data: dict) -> TrainConfig:
        """Build a config from a dict, ignoring unknown keys and coercing numerics."""
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        # coerce numeric fields that YAML may parse as strings (e.g. 5e-4)
        for k, v in filtered.items():
            if isinstance(v, str) and cls.__dataclass_fields__[k].type in ("float", "int"):
                filtered[k] = float(v) if cls.__dataclass_fields__[k].type == "float" else int(v)
        return cls(**filtered)

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        return cls.from_dict(yaml.safe_load(path.read_text()))
