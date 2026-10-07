"""Training loop with checkpoint/resume, grad clipping, and eval."""
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

import torch

# TPU support via torch_xla (optional)
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

# Global flag for on-demand checkpoint save (set via SIGUSR1)
_save_requested = False

# SPMD device mesh when one process drives every TPU chip (cfg.tpu_spmd); None otherwise.
_spmd_mesh = None


def _handle_save_signal(signum, frame):
    global _save_requested
    _save_requested = True
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_lab.data.dataset import TextDataset
from llm_lab.data.instruction import InstructionDataset, MultiTurnDataset, collate_instruction
from llm_lab.tokenizer.bpe import load_tokenizer
from llm_lab.training.config import TrainConfig
from llm_lab.training.model_factory import build_model  # noqa: F401


def _get_device() -> torch.device:
    """Select best available device: TPU > CUDA > CPU."""
    if HAS_XLA:
        return xm.xla_device()
    if torch.cuda.is_available():
        # Under DDP each worker owns one GPU indexed by LOCAL_RANK.
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    return torch.device("cpu")


def _autocast_context(is_tpu: bool, use_amp: bool):
    """Return the mixed-precision context: bf16 on TPU, fp16 on CUDA, none on CPU."""
    if is_tpu:
        return torch.autocast("xla", dtype=torch.bfloat16)
    if use_amp:
        return torch.amp.autocast("cuda", enabled=True)
    return contextlib.nullcontext()


def _init_spmd_mesh(fsdp: bool = False):
    """Build the device mesh over every TPU chip for the SPMD path (xr.use_spmd() done).

    Plain data parallelism uses a 1-D 'data' axis. FSDPv2 insists on an axis named 'fsdp';
    the (n, 1) shape is the layout its docs use. Either way the batch shards on axis 0, and
    the mesh is made global so the flash-attention kernel can find it.
    """
    import numpy as np
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    n = xr.global_runtime_device_count()
    if fsdp:
        mesh = xs.Mesh(np.arange(n), (n, 1), ("fsdp", "model"))
    else:
        mesh = xs.Mesh(np.arange(n), (n,), ("data",))
    xs.set_global_mesh(mesh)
    return mesh


def _to_device(t: torch.Tensor, device) -> torch.Tensor:
    """Move a [batch, ...] tensor to device; under SPMD, shard its batch dim across chips.

    Without tpu_fsdp, params stay replicated (unannotated), so sharding only the inputs makes
    this plain data parallelism: the SPMD partitioner inserts the gradient all-reduce itself.
    A batch that doesn't divide by the chip count (a chat val tail) is left replicated rather
    than unevenly sharded — every chip then computes it in full, which is correct, just slower.
    """
    t = t.to(device)
    if _spmd_mesh is not None and t.size(0) % _spmd_mesh.size() == 0:
        import torch_xla.distributed.spmd as xs
        xs.mark_sharding(t, _spmd_mesh, (_spmd_mesh.axis_names[0],) + (None,) * (t.dim() - 1))
    return t


def _fsdp_wrap(model, optimizer, is_master: bool):
    """Shard params (FSDPv2, one wrap per transformer block) and the optimizer state over chips.

    Runs after resume, so weights and optimizer moments were loaded into the plain module.
    FSDPv2 annotates the existing Parameters in place, so the optimizer keeps pointing at the
    right tensors; checked below, since a silent mismatch would train nothing.
    """
    import functools
    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        SpmdFullyShardedDataParallel as FSDPv2,
    )
    from llm_lab.models.transformer_lm import TransformerBlock

    before = [p for g in optimizer.param_groups for p in g["params"]]
    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    model = FSDPv2(model, auto_wrap_policy=policy)
    after = set(id(p) for p in model.parameters())
    if any(id(p) not in after for p in before):
        raise RuntimeError("tpu_fsdp: FSDPv2 replaced the parameters; the optimizer would update "
                           "tensors the model no longer uses.")
    _shard_optimizer_state(optimizer, is_master)
    if is_master:
        tqdm.write("FSDPv2: params, grads and optimizer state sharded over the 'fsdp' axis.")
    return model


def _shard_optimizer_state(optimizer, is_master: bool) -> None:
    """Give each param-shaped optimizer tensor (Adam moments) the param's dim-0 sharding.

    Moments restored from a checkpoint arrive replicated. Fresh ones don't exist until the
    first step, so this runs again after it. The partitioner would likely shard them anyway;
    this makes it explicit.
    """
    import torch_xla.distributed.spmd as xs
    axis = _spmd_mesh.axis_names[0]
    n = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            for v in optimizer.state.get(p, {}).values():
                if isinstance(v, torch.Tensor) and v.device.type == "xla" and v.dim() >= 1 \
                        and v.shape == p.shape:
                    try:
                        xs.mark_sharding(v, _spmd_mesh, (axis,) + (None,) * (v.dim() - 1))
                        n += 1
                    except Exception as e:  # already sharded, or an API difference
                        if is_master:
                            tqdm.write(f"tpu_fsdp: optimizer state left as is ({e})")
                        return
    if is_master and n:
        tqdm.write(f"tpu_fsdp: sharded {n} optimizer state tensors.")


def _masked_ce(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy over supervised tokens: returns (mean loss, supervised-token count).

    Chat batches can have NO supervised tokens: MultiTurnDataset masks user turns with
    -100, and a conversation whose first user turn fills seq_len (or whose reply is empty)
    has every assistant token truncated away. F.cross_entropy's reduction="mean" then
    divides by zero and returns NaN. Sum / clamp(count, 1) is identical whenever count > 0,
    and gives 0 (not NaN) otherwise. No data-dependent branch, so XLA compiles one graph.
    """
    flat = labels.view(-1)
    loss_sum = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), flat, ignore_index=-100, reduction="sum"
    )
    n_tok = (flat != -100).sum()
    return loss_sum / n_tok.clamp(min=1), n_tok


def _parse_curriculum(curriculum_str: str) -> list[tuple[int, int]]:
    """Parse curriculum string like '256:1000,512:2000,1024:3000' into [(seq_len, until_step), ...]."""
    if not curriculum_str:
        return []
    stages = []
    for part in curriculum_str.split(","):
        seq_str, step_str = part.strip().split(":")
        stages.append((int(seq_str), int(step_str)))
    return stages


def _get_curriculum_seq_len(stages: list[tuple[int, int]], step: int, default: int) -> int:
    """Get the sequence length for the current step based on curriculum."""
    for seq_len, until_step in stages:
        if step < until_step:
            return seq_len
    return default


# Tag on val entries scored by _text_val_dataset. Earlier text runs scored overlapping windows
# at the head of val.bin (~1.3k distinct tokens), so their val_loss is not comparable.
TEXT_VAL_METRIC = "spread"


def _text_val_dataset(path, tokenizer, seq_len: int, n_windows: int):
    """Val set for text runs: non-overlapping windows spread evenly over the whole file.

    The file is written source by source, so the head of it is one source (and with stride 1,
    one document). n_windows evenly spaced windows sample every source in proportion;
    0 (or more than the file holds) keeps all of them.
    """
    from torch.utils.data import Subset
    ds = TextDataset(path, tokenizer, seq_len, stride=seq_len)
    total = len(ds)
    if n_windows <= 0 or n_windows >= total:
        return ds
    return Subset(ds, [i * total // n_windows for i in range(n_windows)])


def _val_loader(val_ds, batch_size, num_workers, distributed, world_size, ordinal, drop_last,
                pin, collate_fn, is_tpu, device):
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=ordinal,
                                     shuffle=False, drop_last=drop_last)
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, sampler=sampler, drop_last=drop_last,
        num_workers=max(0, num_workers - 1), pin_memory=pin, collate_fn=collate_fn,
    )
    if is_tpu and distributed:
        from torch_xla.distributed.parallel_loader import MpDeviceLoader
        loader = MpDeviceLoader(loader, device)
    return loader


def _make_old_val_loader(val_old_path, tokenizer, seq_len, batch_size, max_eval_batches: int,
                         num_workers: int = 2, world_size: int = 1, ordinal: int = 0,
                         is_tpu: bool = False, device=None):
    """Loader for the optional second text val set (cfg.val_old_path), scored like val."""
    ds = _text_val_dataset(val_old_path, tokenizer, seq_len, max_eval_batches * batch_size * world_size)
    return _val_loader(ds, batch_size, num_workers, world_size > 1, world_size, ordinal, True,
                       num_workers > 0 and not is_tpu, None, is_tpu, device)


def _make_loaders(train_path, val_path, tokenizer, seq_len, batch_size, num_workers: int = 2,
                  dataset_type: str = "text", instruction_data: str = "",
                  world_size: int = 1, ordinal: int = 0, is_tpu: bool = False, device=None,
                  max_eval_batches: int = 0):
    """Build train and val dataloaders for a given seq_len.

    When world_size > 1, each replica gets a DistributedSampler shard and the
    loaders are wrapped in an MpDeviceLoader for async host->TPU transfer.
    """
    distributed = world_size > 1
    pin = num_workers > 0 and not is_tpu  # pinned memory only helps CUDA host->device copies

    if dataset_type in ("instruction", "chat"):
        # instruction_data points to a JSON file with training examples
        if not instruction_data:
            raise ValueError("instruction_data path required for dataset_type='instruction'/'chat'")
        data_path = Path(instruction_data)
        DatasetCls = MultiTurnDataset if dataset_type == "chat" else InstructionDataset
        train_ds = DatasetCls(data_path, tokenizer, max_len=seq_len)
        # Use 10% of the same file as val (deterministic slice)
        val_ds = DatasetCls(data_path, tokenizer, max_len=seq_len)
        val_size = max(1, len(val_ds) // 10)
        from torch.utils.data import Subset
        val_ds = Subset(val_ds, list(range(len(val_ds) - val_size, len(val_ds))))
        train_ds = Subset(train_ds, list(range(len(train_ds) - val_size)))
        # On TPU, pad every batch to a fixed length so XLA compiles the graph once
        # instead of recompiling on each new sequence-length shape (100x+ slowdown).
        if is_tpu:
            import functools
            collate_fn = functools.partial(collate_instruction, pad_to=seq_len)
        else:
            collate_fn = collate_instruction
        val_drop_last = False
    else:
        train_ds = TextDataset(train_path, tokenizer, seq_len)
        # Every replica consumes up to max_eval_batches of its DistributedSampler shard.
        val_ds = _text_val_dataset(val_path, tokenizer, seq_len, max_eval_batches * batch_size * world_size)
        collate_fn = None
        val_drop_last = True

    is_text = dataset_type not in ("instruction", "chat")
    if distributed and is_text and not is_tpu:
        # GPU DDP on a small host: DistributedSampler builds randperm(len)=17.6GB PER RANK for a
        # 2.2B-token corpus -> OOM. Per-rank-seeded bounded replacement sampling instead (each
        # rank draws different random samples; overlap is negligible on a huge corpus).
        from torch.utils.data import RandomSampler
        _g = torch.Generator()
        _g.manual_seed(1234 + ordinal)
        n_samples = min(len(train_ds), 2_000_000)
        train_sampler = RandomSampler(train_ds, replacement=True, num_samples=n_samples, generator=_g)
    elif distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=ordinal,
                                           shuffle=True, drop_last=True)
    elif is_text:
        # A plain shuffle=True builds torch.randperm(len(dataset)); for the token corpus
        # len can be billions -> a tens-of-GB tensor that OOMs small GPU hosts (fine on the
        # roomy TPU VM, which is why TPU worked). Sample WITH REPLACEMENT and a bounded
        # count so memory is O(num_samples), not O(dataset). Small instruction/chat sets
        # keep plain shuffling below.
        from torch.utils.data import RandomSampler
        n_samples = min(len(train_ds), 2_000_000)
        train_sampler = RandomSampler(train_ds, replacement=True, num_samples=n_samples)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
        drop_last=True, num_workers=num_workers, pin_memory=pin, collate_fn=collate_fn,
    )
    # Distributed runs shard val across replicas (the text val set is sized for all of them)
    # and average the per-replica losses afterwards.
    val_loader = _val_loader(val_ds, batch_size, num_workers, distributed, world_size, ordinal,
                             val_drop_last, pin, collate_fn, is_tpu, device)

    if is_tpu and distributed:
        from torch_xla.distributed.parallel_loader import MpDeviceLoader
        train_loader = MpDeviceLoader(train_loader, device)

    return train_loader, val_loader


def _setup_topology(is_tpu: bool) -> tuple[int, int]:
    """Return (world_size, ordinal) for the current process."""
    if is_tpu and _spmd_mesh is not None:
        # SPMD: one process owns every chip and the partitioner handles the collectives, so
        # the loop runs its single-process path (no DistributedSampler, no xm.optimizer_step).
        return 1, 0
    if is_tpu:
        import torch_xla.runtime as xr
        return xr.world_size(), xr.global_ordinal()
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def _reduce_mean(value: float, is_tpu: bool, device) -> float:
    """Average a scalar across all workers (TPU mesh reduce or CUDA NCCL all-reduce)."""
    if is_tpu:
        return xm.mesh_reduce("reduce_mean", value, lambda xs: sum(xs) / len(xs))
    import torch.distributed as dist
    t = torch.tensor([value], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def _hbm_note(device, is_tpu: bool) -> str:
    """' | hbm used/total GB' for the log line on TPU; '' when the runtime won't say."""
    if not is_tpu:
        return ""
    try:
        info = xm.get_memory_info(device)
        if "bytes_used" in info:
            used, total = info["bytes_used"], info["bytes_limit"]
        else:  # older torch_xla
            used, total = (info["kb_total"] - info["kb_free"]) * 1024, info["kb_total"] * 1024
        return f" | hbm {used / 2**30:.1f}/{total / 2**30:.1f}GB"
    except Exception:
        return ""


def _build_optimizer(cfg: TrainConfig, model, device: torch.device, is_master: bool):
    """8-bit AdamW when bitsandbytes is available (saves ~280MB), else foreach/standard AdamW."""
    try:
        if not getattr(cfg, "use_8bit_optim", True):
            raise ImportError("8-bit optimizer disabled via config")
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        if is_master:
            tqdm.write(".8-bit AdamW enabled (saves ~280MB)")
    except ImportError:
        use_foreach = device.type == "cuda" and getattr(cfg, "foreach_optim", True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
            foreach=use_foreach,
        )
        if is_master:
            tqdm.write(".Foreach AdamW (bitsandbytes not available)")
    return optimizer


def _make_scheduler(cfg: TrainConfig, optimizer):
    """Linear warmup, then cosine decay toward lr_decay_steps, or (lr_schedule='wsd') a flat
    LR until the last wsd_decay_steps before lr_decay_steps and a 1-sqrt decay to 0 over them.

    Decays toward the FULL training target (lr_decay_steps), not the per-session
    max_steps — otherwise the LR would hit 0 at every session end (each session sets
    max_steps = steps_done + steps_this_session). The lambda is rebuilt from the config
    every session (LambdaLR doesn't save it), so a WSD run still in its flat part extends
    just by raising lr_decay_steps.
    """
    lr_decay_steps = getattr(cfg, "lr_decay_steps", 0) or cfg.max_steps
    schedule = getattr(cfg, "lr_schedule", "cosine")
    if schedule not in ("cosine", "wsd"):
        raise ValueError(f"lr_schedule must be 'cosine' or 'wsd', got {schedule!r}")
    decay_len = getattr(cfg, "wsd_decay_steps", 0)
    if schedule == "wsd" and not 0 < decay_len <= lr_decay_steps - cfg.warmup_steps:
        raise ValueError(f"wsd needs 0 < wsd_decay_steps ({decay_len}) <= lr_decay_steps - "
                         f"warmup_steps ({lr_decay_steps - cfg.warmup_steps}).")
    decay_start = lr_decay_steps - decay_len

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        if schedule == "wsd":
            if step < decay_start:
                return 1.0
            # 1-sqrt cooldown (Hägele et al. 2024): matches cosine at equal compute.
            return max(0.0, 1.0 - math.sqrt(min(1.0, (step - decay_start) / decay_len)))
        progress = (step - cfg.warmup_steps) / max(1, lr_decay_steps - cfg.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _resume_if_available(cfg: TrainConfig, model, optimizer, scheduler, ckpt_dir: Path,
                         device: torch.device, is_master: bool) -> tuple[int, list[dict]]:
    """Resume from latest.pt if present. Returns (start_step, metrics_log)."""
    latest_ckpt = ckpt_dir / "latest.pt"
    if not latest_ckpt.exists():
        return 0, []

    # Load on CPU: storages saved from an XLA device are tagged "xla:0" and can't be
    # restored directly onto an XLA device. load_state_dict then copies onto `device`.
    ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
    # Load into the unwrapped model (DataParallel wraps as .module)
    target = model.module if hasattr(model, "module") else model
    incompat = target.load_state_dict(_strip_prefixes(ckpt["model"]), strict=False)
    if is_master and (incompat.missing_keys or incompat.unexpected_keys):
        tqdm.write(f"WARNING: checkpoint/model key mismatch — missing={list(incompat.missing_keys)[:4]} "
                   f"unexpected={list(incompat.unexpected_keys)[:4]} (architecture may differ from checkpoint)")
    reset_opt = getattr(cfg, "reset_optimizer", False)
    if ckpt.get("optimizer") is not None and not reset_opt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError, RuntimeError) as e:
            # Mismatched optimizer types (e.g. resuming a standard-AdamW checkpoint under
            # 8-bit AdamW) — keep training rather than crash, but warn: moments reset.
            if is_master:
                tqdm.write(f"WARNING: optimizer state not restored ({e}); starting with fresh moments.")
        else:
            # Optimizer state loaded on CPU; move it onto the training device.
            for opt_state in optimizer.state.values():
                for k, v in opt_state.items():
                    if isinstance(v, torch.Tensor):
                        opt_state[k] = v.to(device)
    if ckpt.get("scheduler") is not None and not reset_opt:
        scheduler.load_state_dict(ckpt["scheduler"])
    step = ckpt["step"]
    # Drop history from beyond the resumed step. A checkpoint rolled back to an earlier step
    # keeps the whole metrics_log, so the log can describe steps these weights never reached.
    # Everything downstream that asks "how far along are we?" by taking max(step) over
    # metrics.json then reads a counter stuck in the future — including the notebooks'
    # auto-push watcher, which goes silent for the rest of the run because the counter never
    # advances past what it already banked.
    _full_log = ckpt.get("metrics_log", [])
    metrics_log = [e for e in _full_log if e.get("step", 0) <= step]
    _dropped = len(_full_log) - len(metrics_log)
    fresh = ckpt.get("optimizer") is None or reset_opt
    # Free the ~2.8GB checkpoint from host RAM before the first step (the CPU copy is no
    # longer needed once weights + optimizer state are on the device).
    del ckpt
    import gc as _gc
    _gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if is_master:
        tqdm.write(f"Resumed from step {step}"
                   + (" (fresh optimizer — clean fine-tune start)" if fresh else "")
                   + (f"; dropped {_dropped} metrics entries from beyond step {step}"
                      if _dropped else ""))
    return step, metrics_log


def train(cfg: TrainConfig) -> dict:
    """Run training loop. Returns final metrics."""
    global _save_requested, _spmd_mesh
    _save_requested = False
    # signal handlers can only be registered on the main thread; xmp.spawn runs each
    # replica in a per-device thread, so skip (SIGUSR1 save is single-process only).
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGUSR1, _handle_save_signal)

    torch.manual_seed(cfg.seed)
    device = _get_device()
    is_tpu = HAS_XLA and device.type == "xla"
    if device.type == "cuda":
        torch.cuda.set_device(device)

    _spmd_mesh = None
    use_fsdp = bool(getattr(cfg, "tpu_fsdp", False))
    if is_tpu:
        import torch_xla.runtime as xr
        if use_fsdp and not xr.is_spmd():
            raise ValueError("tpu_fsdp needs tpu_spmd (one SPMD process over every chip).")
        if xr.is_spmd():
            _spmd_mesh = _init_spmd_mesh(fsdp=use_fsdp)
            n_chips = _spmd_mesh.size()
            if cfg.batch_size % n_chips:
                raise ValueError(f"tpu_spmd: batch_size {cfg.batch_size} is the global micro-batch "
                                 f"and must divide by the {n_chips} chips.")
            print(f"SPMD: 1 process over {n_chips} chips, batch {cfg.batch_size} "
                  f"({cfg.batch_size // n_chips}/chip) sharded on the '{_spmd_mesh.axis_names[0]}' axis.")
    elif use_fsdp:
        raise ValueError("tpu_fsdp is TPU-only.")

    # Data-parallel topology (multi-core TPU via torch_xla; single otherwise)
    world_size, ordinal = _setup_topology(is_tpu)
    is_master = ordinal == 0

    # Enable TF32 for faster matmuls on Turing+ GPUs
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = load_tokenizer(Path(cfg.tokenizer_path))

    # Sequence length curriculum
    curriculum = _parse_curriculum(cfg.seq_curriculum)
    if curriculum:
        initial_seq_len = curriculum[0][0]
        if is_master:
            print(f"Curriculum: {' → '.join(f'{s}@{t}' for s, t in curriculum)}")
    else:
        initial_seq_len = cfg.seq_len

    train_path = Path(cfg.train_path) if cfg.train_path else None
    val_path = Path(cfg.val_path) if cfg.val_path else None
    dataset_type = getattr(cfg, "dataset_type", "text")
    instruction_data = getattr(cfg, "instruction_data", "")
    current_seq_len = initial_seq_len
    train_loader, val_loader = _make_loaders(
        train_path, val_path, tokenizer, current_seq_len, cfg.batch_size,
        getattr(cfg, "num_workers", 2),
        dataset_type=dataset_type,
        instruction_data=instruction_data,
        world_size=world_size, ordinal=ordinal, is_tpu=is_tpu, device=device,
        max_eval_batches=cfg.max_eval_batches,
    )
    val_old_path = getattr(cfg, "val_old_path", "")
    if val_old_path and dataset_type != "text":
        raise ValueError("val_old_path is for text runs only.")
    val_old_loader = None
    if val_old_path:
        val_old_loader = _make_old_val_loader(
            Path(val_old_path), tokenizer, cfg.seq_len, cfg.batch_size, cfg.max_eval_batches,
            getattr(cfg, "num_workers", 2), world_size=world_size, ordinal=ordinal,
            is_tpu=is_tpu, device=device,
        )

    model = build_model(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    # Multi-GPU uses DistributedDataParallel (one process per GPU, balanced memory) — NOT
    # nn.DataParallel, which piles the optimizer + gathered logits onto GPU 0 and OOMs.
    is_ddp = device.type == "cuda" and world_size > 1
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[device.index])
        if is_master:
            tqdm.write(f".Model: {cfg.model_type}, params: {param_count:,}, device: {device} x{world_size} (DDP/NCCL)")
    elif is_master:
        topo = f" x{world_size} (xla-multiprocessing)" if world_size > 1 else ""
        tqdm.write(f".Model: {cfg.model_type}, params: {param_count:,}, device: {device}{topo}")

    # 8-bit AdamW saves ~280MB optimizer memory; fall back to foreach if unavailable
    optimizer = _build_optimizer(cfg, model, device, is_master)

    scheduler = _make_scheduler(cfg, optimizer)

    # resume from checkpoint
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_step = 0
    evals_since_improve = 0  # for early stopping
    start_step, metrics_log = _resume_if_available(
        cfg, model, optimizer, scheduler, ckpt_dir, device, is_master
    )

    if use_fsdp:
        model = _fsdp_wrap(model, optimizer, is_master)

    if getattr(cfg, "compile", False):
        model = torch.compile(model)
        if is_master:
            tqdm.write("torch.compile enabled (first step will be slow — JIT compiling kernels)")

    # Text val entries carry a metric tag; only same-metric history may set the best to beat,
    # or a resumed run would compare its loss against the old one-document number.
    val_metric = TEXT_VAL_METRIC if dataset_type == "text" else None

    def _comparable(entry: dict) -> bool:
        return "val_loss" in entry and entry.get("val_metric") == val_metric

    # Keep track of best validation from prior history for persistent best.pt updates
    for entry in metrics_log:
        if _comparable(entry) and entry["val_loss"] < best_val_loss:
            best_val_loss = entry["val_loss"]
            best_step = entry["step"]

    # Early-stop patience tracks improvement *within this session*, not against the
    # all-time historical best — otherwise a resume after a val regression counts the
    # entire recovery as "no improvement" and stops prematurely.
    early_stop_best = float("inf")

    # ...but that per-session reset also means patience must fit inside one session, or
    # the counter never reaches it and early stopping is silently dead. The first eval
    # always resets the counter (inf), so a session affords evals-1 increments.
    _patience = getattr(cfg, "early_stop_patience", 0)
    if _patience > 0 and is_master:
        _evals_this_session = (cfg.max_steps - start_step) // max(cfg.eval_every, 1)
        if _patience > _evals_this_session - 1:
            tqdm.write(
                f"Note: early_stop_patience={_patience} exceeds this session's "
                f"{_evals_this_session} evals ({cfg.max_steps - start_step} steps / "
                f"eval_every {cfg.eval_every}) — early stopping cannot trigger. "
                f"Lower it to <= {max(_evals_this_session - 1, 1)} to arm it."
            )

    # training loop
    model.train()
    data_iter = iter(train_loader)
    t0 = time.time()

    # Mixed precision: fp16 autocast + grad scaler on CUDA, bf16 autocast on TPU
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Gradient accumulation
    accum_steps = max(1, cfg.grad_accum_steps)
    effective_batch = cfg.batch_size * accum_steps * world_size
    if accum_steps > 1 and is_master:
        tqdm.write(f"Gradient accumulation: {accum_steps} steps (effective batch={effective_batch})")
    fuse_step = is_tpu and getattr(cfg, "tpu_fuse_step", False)
    tokens_per_step = effective_batch * cfg.seq_len

    # Wall-clock session cap. Each rank would read its own clock, so multi-process runs (xmp,
    # DDP) could disagree on the stopping step and deadlock; only one-process runs get it.
    max_minutes = getattr(cfg, "max_train_minutes", 0) or 0
    if max_minutes > 0 and world_size > 1:
        if is_master:
            tqdm.write("max_train_minutes ignored: multi-process run.")
        max_minutes = 0
    end_step = cfg.max_steps
    t_log, step_log = time.time(), start_step

    for step in tqdm(range(start_step, cfg.max_steps), initial=start_step, total=cfg.max_steps,
                     dynamic_ncols=True, disable=not is_master):
        try:
            # Curriculum: check if seq_len should change
            if curriculum:
                target_seq_len = _get_curriculum_seq_len(curriculum, step, cfg.seq_len)
                if target_seq_len != current_seq_len:
                    current_seq_len = target_seq_len
                    train_loader, val_loader = _make_loaders(
                        train_path, val_path, tokenizer, current_seq_len, cfg.batch_size,
                        getattr(cfg, "num_workers", 2),
                        dataset_type=dataset_type,
                        instruction_data=instruction_data,
                        world_size=world_size, ordinal=ordinal, is_tpu=is_tpu, device=device,
                        max_eval_batches=cfg.max_eval_batches,
                    )
                    data_iter = iter(train_loader)
                    if is_master:
                        tqdm.write(f"Curriculum → seq_len={current_seq_len} at step {step}")

            optimizer.zero_grad()

            for micro_step in range(accum_steps):
                # get batch, cycle through data
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                input_ids = _to_device(batch["input_ids"], device)
                labels = _to_device(batch["labels"], device)

                # DDP: only all-reduce grads on the final micro-step of the accumulation.
                _sync = (micro_step == accum_steps - 1)
                _sync_ctx = model.no_sync() if (is_ddp and not _sync) else contextlib.nullcontext()
                with _sync_ctx:
                    with _autocast_context(is_tpu, use_amp):
                        logits = model(input_ids)
                        # 0 with zero grads (not NaN) on a micro-batch with no supervised
                        # tokens. TPU has no GradScaler to skip a NaN step for us.
                        loss, _ = _masked_ce(logits, labels)
                        loss = loss / accum_steps  # normalize for accumulation

                    if is_tpu:
                        loss.backward()
                        # Flush each micro-step so the lazy XLA graph spans ONE microbatch,
                        # not all accum_steps — grads persist in .grad across mark_step, so
                        # peak HBM stays ~batch (not batch*accum, which OOMs at 235M).
                        # tpu_fuse_step leaves the last one to the optimizer step's flush.
                        if not (fuse_step and _sync):
                            xm.mark_step()
                    else:
                        scaler.scale(loss).backward()

            if is_tpu:
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if world_size > 1:
                    xm.optimizer_step(optimizer)  # all-reduce grads across cores, step, mark
                else:
                    optimizer.step()
                    xm.mark_step()
            else:
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            if use_fsdp and step == start_step:
                _shard_optimizer_state(optimizer, is_master)  # fresh moments exist only now

            # Re-scale loss for logging (undo the /accum_steps)
            loss = loss * accum_steps

            # logging
            if (step + 1) % cfg.log_every == 0 and is_master:
                train_loss = loss.item()  # host sync: the step's work is done after this
                now = time.time()
                elapsed = now - t0
                tok_s = (step + 1 - step_log) * tokens_per_step / max(now - t_log, 1e-6)
                t_log, step_log = now, step + 1
                entry = {
                    "step": step + 1,
                    "train_loss": train_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": round(elapsed, 1),
                    "tok_s": round(tok_s),
                }
                metrics_log.append(entry)
                tqdm.write(
                    f"step {entry['step']:>5d} | loss {entry['train_loss']:.4f} | "
                    f"lr {entry['lr']:.2e} | {elapsed:.0f}s | {tok_s / 1e3:.1f}k tok/s"
                    + _hbm_note(device, is_tpu)
                )

            # eval — every core evaluates its shard, then we average across cores
            if (step + 1) % cfg.eval_every == 0:
                val_loss = evaluate(model, val_loader, device, cfg.max_eval_batches)
                if world_size > 1:
                    val_loss = _reduce_mean(val_loss, is_tpu, device)
                entry = {"step": step + 1, "val_loss": val_loss}
                if val_metric:
                    entry["val_metric"] = val_metric
                note = ""
                if val_old_loader is not None:
                    val_old = evaluate(model, val_old_loader, device, cfg.max_eval_batches)
                    if world_size > 1:
                        val_old = _reduce_mean(val_old, is_tpu, device)
                    entry["val_old_loss"] = val_old
                    note = f" | val_old_loss {val_old:.4f}"
                metrics_log.append(entry)
                if is_master:
                    tqdm.write(f"step {step + 1:>5d} | val_loss {val_loss:.4f}{note}")
                min_delta = getattr(cfg, "early_stop_min_delta", 0.0)
                # best.pt: overwrite on ANY all-time improvement. min_delta is an
                # early-stopping threshold ("is progress still worth the compute?") and
                # must not gate checkpoint selection too — at min_delta=0.01 a genuinely
                # better eval that improves by less is discarded, leaving best.pt on a
                # worse step than one the run actually reached.
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step + 1
                    if is_tpu or is_master:
                        _save_checkpoint(
                            model, optimizer, scheduler, step + 1, metrics_log, ckpt_dir,
                            is_tpu=is_tpu, best=True, best_val_loss=best_val_loss,
                            is_master=is_master,
                        )
                    if is_master:
                        tqdm.write(f"New best val_loss {best_val_loss:.4f} at step {best_step} (saved: best.pt)")
                # early stop: measure improvement within this session.
                if val_loss < early_stop_best - min_delta:
                    early_stop_best = val_loss
                    evals_since_improve = 0
                else:
                    evals_since_improve += 1
                model.train()

                patience = getattr(cfg, "early_stop_patience", 0)
                if patience > 0 and evals_since_improve >= patience:
                    if is_master:
                        tqdm.write(
                            f"Early stopping at step {step + 1}: no val_loss improvement for "
                            f"{evals_since_improve} evals (best {best_val_loss:.4f} @ step {best_step})."
                        )
                    if is_tpu or is_master:
                        _save_checkpoint(model, optimizer, scheduler, step + 1, metrics_log, ckpt_dir, is_tpu=is_tpu, is_master=is_master)
                    if is_master:
                        (ckpt_dir / "metrics.json").write_text(json.dumps(metrics_log, indent=2))
                    return {
                        "early_stopped_at_step": step + 1,
                        "best_val_loss": best_val_loss,
                        "best_step": best_step,
                        "param_count": param_count,
                    }

            # checkpoint
            if (step + 1) % cfg.checkpoint_every == 0 and (is_tpu or is_master):
                _save_checkpoint(
                    model, optimizer, scheduler, step + 1, metrics_log, ckpt_dir, is_tpu=is_tpu, is_master=is_master,
                    keep_last=getattr(cfg, "checkpoint_keep_last", 2),
                )

            # on-demand checkpoint via SIGUSR1 (single-process only — unsynced save deadlocks XLA)
            if _save_requested and world_size == 1:
                _save_requested = False
                tqdm.write(f"SIGUSR1 received — saving checkpoint at step {step + 1}")
                _save_checkpoint(
                    model, optimizer, scheduler, step + 1, metrics_log, ckpt_dir, is_tpu=is_tpu,
                    is_master=is_master,
                )

            if max_minutes > 0 and (time.time() - t0) / 60 >= max_minutes and step + 1 < cfg.max_steps:
                end_step = step + 1
                if is_master:
                    tqdm.write(f"max_train_minutes={max_minutes:g} reached at step {end_step}: "
                               "stopping this session (the final save below banks it).")
                break

        except KeyboardInterrupt:
            # Multi-core: skip the save (an unsynchronized xm.save would deadlock the other cores)
            if world_size > 1:
                raise
            tqdm.write(f"\nInterrupted at step {step + 1} — saving checkpoint before exit...")
            _save_checkpoint(model, optimizer, scheduler, step + 1, metrics_log, ckpt_dir, is_tpu=is_tpu, is_master=is_master)
            metrics_path = ckpt_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics_log, indent=2))
            return {"interrupted_at_step": step + 1, "param_count": param_count}

    # final checkpoint + eval
    val_loss = evaluate(model, val_loader, device, cfg.max_eval_batches)
    if world_size > 1:
        val_loss = _reduce_mean(val_loss, is_tpu, device)
    if is_tpu or is_master:
        _save_checkpoint(model, optimizer, scheduler, end_step, metrics_log, ckpt_dir, is_tpu=is_tpu, is_master=is_master,
                         keep_last=getattr(cfg, "checkpoint_keep_last", 2))

    # save metrics
    if is_master:
        metrics_path = ckpt_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_log, indent=2))

    return {
        "final_val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "param_count": param_count,
        "steps": end_step,
    }


def _mp_fn(index, cfg: TrainConfig) -> None:
    """torch_xla multiprocessing entrypoint — one process per TPU core."""
    train(cfg)


def _prebuild_text_cache(cfg: TrainConfig) -> None:
    """Build TextDataset .bin caches once, before spawning workers.

    Each worker would otherwise build the same memmap cache concurrently, and one
    process truncating the file while another reads it faults with SIGBUS.
    """
    if getattr(cfg, "dataset_type", "text") != "text":
        return  # instruction/chat datasets tokenize in-memory, no shared cache file
    tokenizer = load_tokenizer(Path(cfg.tokenizer_path))
    for path in (cfg.train_path, cfg.val_path):
        if path:
            TextDataset(Path(path), tokenizer, cfg.seq_len)


def train_multicore(cfg: TrainConfig) -> None:
    """Run data-parallel training across all TPU cores.

    xmp.spawn requires the XLA runtime to be uninitialized in the calling process,
    but notebook kernels usually already touched the TPU (single-core runs, device
    probes). So unless we're already inside a clean worker, we re-exec the CLI in a
    fresh subprocess and spawn from there.
    """
    import os

    if os.environ.get("AION_TPU_WORKER") == "1" and getattr(cfg, "tpu_spmd", False):
        # One process, one compile for all chips. xmp.spawn's 8 processes each compile the
        # full 235M fwd+bwd graph at once, and that host-RAM spike is the BrokenProcessPool.
        import torch_xla.runtime as xr
        xr.use_spmd()  # must precede any XLA device use — hence the fresh subprocess
        train(cfg)
        return

    if os.environ.get("AION_TPU_WORKER") == "1":
        import torch_xla.distributed.xla_multiprocessing as xmp
        _prebuild_text_cache(cfg)  # avoid concurrent memmap cache writes (SIGBUS)
        # PJRT supports nprocs=None (all cores) or 1; None fans out to every TPU core.
        xmp.spawn(_mp_fn, args=(cfg,))
        return

    import subprocess
    import sys
    import tempfile
    import llm_lab

    pkg_root = str(Path(llm_lab.__file__).resolve().parent.parent)
    tmp_dir = Path(tempfile.mkdtemp(prefix="aion_mp_"))
    cfg_path = tmp_dir / "config.yaml"
    cfg.save(cfg_path)
    env = dict(
        os.environ,
        AION_TPU_WORKER="1",
        PYTHONPATH=pkg_root + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    # An earlier xm.xla_device() probe configures single-process TPU topology and leaves
    # per-process vars in the kernel env. configure_topology() writes these via setdefault,
    # so an inherited value blocks the correct per-rank value (e.g. TPU_PROCESS_ADDRESSES
    # stays "local" -> "Expected 8 worker addresses, got 1"). Drop ONLY these outputs;
    # keep TPU_PROCESS_BOUNDS / TPU_CHIPS_PER_PROCESS_BOUNDS / TPU_ACCELERATOR_TYPE, which
    # configure_topology reads as the real slice topology (Kaggle sets TPU_SKIP_MDS_QUERY).
    for _k in ("TPU_PROCESS_ADDRESSES", "TPU_VISIBLE_CHIPS", "TPU_PROCESS_PORT", "CLOUD_TPU_TASK_ID"):
        env.pop(_k, None)
    print(f"Launching multi-core TPU training in a fresh process (config: {cfg_path})...")
    subprocess.run(
        [sys.executable, "-m", "llm_lab.cli", "train", "--config", str(cfg_path)],
        env=env, cwd=pkg_root, check=True,
    )


def _ddp_worker(rank: int, world_size: int, cfg: TrainConfig) -> None:
    """One process per GPU: init the NCCL group, then run the standard training loop."""
    import torch.distributed as dist
    # mp.spawn hands this child a plain block-buffered pipe for stdout instead of the
    # notebook kernel's auto-flushing stream. tqdm.write (train/val lines) goes to stdout
    # while the progress bar goes to stderr, so at log_every=50 + eval_every=500 a whole
    # 5000-step session emits ~6KB — under the 8KB buffer — and NO loss line ever reaches
    # the log, even though the bar streams fine. Line-buffer it so eval output appears as
    # it happens and survives a hard timeout, which never runs the exit-time flush.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    # Each rank loads the full checkpoint on CPU during resume; DataLoader worker procs
    # would add more host-RAM pressure on top. Load data in-process to avoid OOM (memmap
    # dataset means no data copy anyway).
    cfg.num_workers = 0
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        train(cfg)
    finally:
        dist.barrier()
        dist.destroy_process_group()


def train_ddp(cfg: TrainConfig) -> None:
    """Data-parallel training across CUDA GPUs via DistributedDataParallel (NCCL)."""
    import torch.multiprocessing as mp
    n = torch.cuda.device_count()
    requested = getattr(cfg, "gpus", 1)
    world_size = n if requested in (0, -1) else min(requested, n)
    if world_size <= 1:
        train(cfg)
        return
    print(f"Launching DDP training across {world_size} GPUs (NCCL)...")
    mp.spawn(_ddp_worker, args=(world_size, cfg), nprocs=world_size, join=True)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, max_batches: int = 0) -> float:
    model.eval()
    total_loss = 0.0
    n = 0       # batches consumed — still capped by max_batches, so every eval scores the same slice
    scored = 0  # batches that had supervised tokens
    for batch in loader:
        input_ids = _to_device(batch["input_ids"], device)
        labels = _to_device(batch["labels"], device)
        logits = model(input_ids)
        loss, n_tok = _masked_ce(logits, labels)
        loss, n_tok = torch.stack([loss.float(), n_tok.float()]).tolist()  # one host sync
        n += 1
        # Skip a batch with no supervised tokens rather than averaging in its NaN — one such
        # batch among max_eval_batches turned the whole val_loss NaN, which never beats
        # best_val_loss and counts against early-stop patience on every eval. Skipping
        # (rather than counting it as 0) keeps this the same mean-of-batch-means as before,
        # so val_loss stays comparable with the run's existing history.
        if n_tok > 0:
            total_loss += loss
            scored += 1
        if max_batches > 0 and n >= max_batches:
            break
    return total_loss / scored if scored else float("nan")


def _write_ckpt(ckpt: dict, path: Path, is_tpu: bool) -> None:
    """Serialize a checkpoint. On TPU use xm.save (all cores must call it; only master writes)."""
    if is_tpu and _spmd_mesh is not None:
        # SPMD is one process, so there is no cross-core rendezvous to join; copy the
        # (replicated) device tensors to CPU and save like any single-process run.
        from torch.utils._pytree import tree_map
        torch.save(tree_map(lambda v: v.cpu() if isinstance(v, torch.Tensor) else v, ckpt), path)
    elif is_tpu:
        xm.save(ckpt, str(path))
    else:
        torch.save(ckpt, path)


def _strip_prefixes(state_dict: dict) -> dict:
    """Drop FSDPv2's '_orig_module.', DataParallel's 'module.' and torch.compile's '_orig_mod.'
    key prefixes. '_orig_module.' goes first: it contains 'module.'."""
    return {k.replace("_orig_module.", "").replace("module.", "").replace("_orig_mod.", ""): v
            for k, v in state_dict.items()}


def load_model(cfg: TrainConfig, ckpt_path, device: torch.device):
    """Build a model and load checkpoint weights onto device (for eval/generation)."""
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(_strip_prefixes(ckpt["model"]))
    return model


def seed_checkpoint(src: Path, dst: Path) -> None:
    """Write a fresh warm-start checkpoint (weights only, step 0, no optimizer)."""
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": 0,
            "model": _strip_prefixes(ckpt["model"]),
            "optimizer": None,
            "scheduler": None,
            "metrics_log": [],
            "_source": str(src),
        },
        dst,
    )


def _save_checkpoint(
    model, optimizer, scheduler, step: int, metrics_log: list, ckpt_dir: Path,
    keep_last: int = 2, is_tpu: bool = False, best: bool = False, best_val_loss: float | None = None,
    is_master: bool = True,
) -> None:
    ckpt = {
        "step": step,
        "model": _strip_prefixes(model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "metrics_log": metrics_log,
    }

    if best:
        # best.pt is only ever used for serving / warm-start seeding (resume reads
        # latest.pt), so it never needs optimizer/scheduler state. Writing weights only
        # keeps these frequent val-improvement writes ~3x smaller — critical on Colab,
        # where large checkpoints written to the Google Drive FUSE mount buffer in host
        # RAM faster than they upload and eventually OOM-kill the kernel.
        best_ckpt = {
            "step": step,
            "model": ckpt["model"],
            "metrics_log": metrics_log,
            "best_val_loss": best_val_loss,
        }
        _write_ckpt(best_ckpt, ckpt_dir / "best.pt", is_tpu)
        return

    latest = ckpt_dir / "latest.pt"
    if keep_last > 0:
        path = ckpt_dir / f"step_{step}.pt"
        _write_ckpt(ckpt, path, is_tpu)
        if is_tpu:
            # xm.save is a collective — every core must call it; can't substitute a file copy.
            _write_ckpt(ckpt, latest, is_tpu)
        else:
            # Serialize once, then copy the file for latest.pt instead of re-serializing the
            # whole ~GB checkpoint a second time (halves the save-time work and write burst).
            shutil.copyfile(path, latest)
    else:
        # keep_last=0 archives nothing, so write latest.pt directly. Writing step_N.pt first
        # and unlinking it below doubled the bytes per save for no benefit — ~2.8GB of extra
        # traffic to the Drive FUSE mount every checkpoint on the 235M Colab run.
        # (keep_last is uniform across ranks, so TPU cores still call _write_ckpt in lockstep.)
        path = latest
        _write_ckpt(ckpt, latest, is_tpu)

    # Aux files + cleanup: master only (xm.save already wrote only on master).
    # Use the caller's is_master, which comes from xr.global_ordinal() like every other
    # rank check in this file. This used to call xm.is_master_ordinal(), a DIFFERENT api
    # that defaults to the *local* ordinal — when the two disagreed, latest.pt (written
    # above) still updated but metrics.json did not, so the notebooks' auto-push watcher
    # polled a step counter frozen at the resume point and never banked anything.
    if is_tpu and not is_master:
        return
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics_log))
    tqdm.write(f"Checkpoint saved: {path}")

    # Auto-cleanup: keep only the last N numbered checkpoints. With keep_last=0 nothing
    # numbered is written any more, so this just sweeps archives left by earlier sessions
    # (or an interrupt save, which keeps the default). [:-0] is [:0]==empty, so guard it.
    numbered = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in (numbered[:-keep_last] if keep_last > 0 else numbered):
        old.unlink()
        tqdm.write(f"Removed old checkpoint: {old.name}")

