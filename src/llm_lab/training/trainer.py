"""Training loop with checkpoint/resume, grad clipping, and eval."""
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
import signal
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


def _make_loaders(train_path, val_path, tokenizer, seq_len, batch_size, num_workers: int = 2,
                  dataset_type: str = "text", instruction_data: str = "",
                  world_size: int = 1, ordinal: int = 0, is_tpu: bool = False, device=None):
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
        val_ds = TextDataset(val_path, tokenizer, seq_len)
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
        val_sampler = None  # eval is capped by max_eval_batches; every rank scores the same slice
    elif distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=ordinal,
                                           shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=ordinal,
                                         shuffle=False, drop_last=val_drop_last)
    elif is_text:
        # A plain shuffle=True builds torch.randperm(len(dataset)); for the token corpus
        # len can be billions -> a tens-of-GB tensor that OOMs small GPU hosts (fine on the
        # roomy TPU VM, which is why TPU worked). Sample WITH REPLACEMENT and a bounded
        # count so memory is O(num_samples), not O(dataset). Small instruction/chat sets
        # keep plain shuffling below.
        from torch.utils.data import RandomSampler
        n_samples = min(len(train_ds), 2_000_000)
        train_sampler = RandomSampler(train_ds, replacement=True, num_samples=n_samples)
        val_sampler = None
    else:
        train_sampler = val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
        drop_last=True, num_workers=num_workers, pin_memory=pin, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler, drop_last=val_drop_last,
        num_workers=max(0, num_workers - 1), pin_memory=pin, collate_fn=collate_fn,
    )

    if is_tpu and distributed:
        from torch_xla.distributed.parallel_loader import MpDeviceLoader
        train_loader = MpDeviceLoader(train_loader, device)
        val_loader = MpDeviceLoader(val_loader, device)

    return train_loader, val_loader


def _setup_topology(is_tpu: bool) -> tuple[int, int]:
    """Return (world_size, ordinal) for the current process."""
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
    """Linear warmup then cosine decay toward lr_decay_steps.

    Decays toward the FULL training target (lr_decay_steps), not the per-session
    max_steps — otherwise the LR would hit 0 at every session end (each session sets
    max_steps = steps_done + steps_this_session).
    """
    lr_decay_steps = getattr(cfg, "lr_decay_steps", 0) or cfg.max_steps

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
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
    metrics_log = ckpt.get("metrics_log", [])
    step = ckpt["step"]
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
                   + (" (fresh optimizer — clean fine-tune start)" if fresh else ""))
    return step, metrics_log


def train(cfg: TrainConfig) -> dict:
    """Run training loop. Returns final metrics."""
    global _save_requested
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

    if getattr(cfg, "compile", False):
        model = torch.compile(model)
        if is_master:
            tqdm.write("torch.compile enabled (first step will be slow — JIT compiling kernels)")

    # Keep track of best validation from prior history for persistent best.pt updates
    for entry in metrics_log:
        if "val_loss" in entry and entry["val_loss"] < best_val_loss:
            best_val_loss = entry["val_loss"]
            best_step = entry["step"]

    # Early-stop patience tracks improvement *within this session*, not against the
    # all-time historical best — otherwise a resume after a val regression counts the
    # entire recovery as "no improvement" and stops prematurely.
    early_stop_best = float("inf")

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

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # DDP: only all-reduce grads on the final micro-step of the accumulation.
                _sync = (micro_step == accum_steps - 1)
                _sync_ctx = model.no_sync() if (is_ddp and not _sync) else contextlib.nullcontext()
                with _sync_ctx:
                    with _autocast_context(is_tpu, use_amp):
                        logits = model(input_ids)
                        loss = torch.nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)), labels.view(-1)
                        )
                        loss = loss / accum_steps  # normalize for accumulation

                    if is_tpu:
                        loss.backward()
                        # Flush each micro-step so the lazy XLA graph spans ONE microbatch,
                        # not all accum_steps — grads persist in .grad across mark_step, so
                        # peak HBM stays ~batch (not batch*accum, which OOMs at 235M).
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

            # Re-scale loss for logging (undo the /accum_steps)
            loss = loss * accum_steps

            # logging
            if (step + 1) % cfg.log_every == 0 and is_master:
                elapsed = time.time() - t0
                entry = {
                    "step": step + 1,
                    "train_loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": round(elapsed, 1),
                }
                metrics_log.append(entry)
                tqdm.write(
                    f"step {entry['step']:>5d} | loss {entry['train_loss']:.4f} | "
                    f"lr {entry['lr']:.2e} | {elapsed:.0f}s"
                )

            # eval — every core evaluates its shard, then we average across cores
            if (step + 1) % cfg.eval_every == 0:
                val_loss = evaluate(model, val_loader, device, cfg.max_eval_batches)
                if world_size > 1:
                    val_loss = _reduce_mean(val_loss, is_tpu, device)
                entry = {"step": step + 1, "val_loss": val_loss}
                metrics_log.append(entry)
                if is_master:
                    tqdm.write(f"step {step + 1:>5d} | val_loss {val_loss:.4f}")
                min_delta = getattr(cfg, "early_stop_min_delta", 0.0)
                # best.pt: overwrite only when beating the all-time best.
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_step = step + 1
                    if is_tpu or is_master:
                        _save_checkpoint(
                            model, optimizer, scheduler, step + 1, metrics_log, ckpt_dir,
                            is_tpu=is_tpu, best=True, best_val_loss=best_val_loss,
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
                        _save_checkpoint(model, optimizer, scheduler, step + 1, metrics_log, ckpt_dir, is_tpu=is_tpu)
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
                    model, optimizer, scheduler, step + 1, metrics_log, ckpt_dir, is_tpu=is_tpu
                )

            # on-demand checkpoint via SIGUSR1 (single-process only — unsynced save deadlocks XLA)
            if _save_requested and world_size == 1:
                _save_requested = False
                tqdm.write(f"SIGUSR1 received — saving checkpoint at step {step + 1}")
                _save_checkpoint(
                    model, optimizer, scheduler, step + 1, metrics_log, ckpt_dir, is_tpu=is_tpu
                )

        except KeyboardInterrupt:
            # Multi-core: skip the save (an unsynchronized xm.save would deadlock the other cores)
            if world_size > 1:
                raise
            tqdm.write(f"\nInterrupted at step {step + 1} — saving checkpoint before exit...")
            _save_checkpoint(model, optimizer, scheduler, step + 1, metrics_log, ckpt_dir, is_tpu=is_tpu)
            metrics_path = ckpt_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics_log, indent=2))
            return {"interrupted_at_step": step + 1, "param_count": param_count}

    # final checkpoint + eval
    val_loss = evaluate(model, val_loader, device, cfg.max_eval_batches)
    if world_size > 1:
        val_loss = _reduce_mean(val_loss, is_tpu, device)
    if is_tpu or is_master:
        _save_checkpoint(model, optimizer, scheduler, cfg.max_steps, metrics_log, ckpt_dir, is_tpu=is_tpu)

    # save metrics
    if is_master:
        metrics_path = ckpt_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_log, indent=2))

    return {
        "final_val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "param_count": param_count,
        "steps": cfg.max_steps,
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
    n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        total_loss += loss.item()
        n += 1
        if max_batches > 0 and n >= max_batches:
            break
    return total_loss / max(n, 1)


def _write_ckpt(ckpt: dict, path: Path, is_tpu: bool) -> None:
    """Serialize a checkpoint. On TPU use xm.save (all cores must call it; only master writes)."""
    if is_tpu:
        xm.save(ckpt, str(path))
    else:
        torch.save(ckpt, path)


def _strip_prefixes(state_dict: dict) -> dict:
    """Drop DataParallel's 'module.' and torch.compile's '_orig_mod.' key prefixes."""
    return {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}


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

    path = ckpt_dir / f"step_{step}.pt"
    latest = ckpt_dir / "latest.pt"
    _write_ckpt(ckpt, path, is_tpu)
    if is_tpu:
        # xm.save is a collective — every core must call it; can't substitute a file copy.
        _write_ckpt(ckpt, latest, is_tpu)
    else:
        # Serialize once, then copy the file for latest.pt instead of re-serializing the
        # whole ~GB checkpoint a second time (halves the save-time work and write burst).
        shutil.copyfile(path, latest)

    # Aux files + cleanup: master only (xm.save already wrote only on master)
    if is_tpu and not xm.is_master_ordinal():
        return
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics_log))
    tqdm.write(f"Checkpoint saved: {path}")

    # Auto-cleanup: keep only the last N numbered checkpoints
    numbered = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in numbered[:-keep_last]:
        old.unlink()
        tqdm.write(f"Removed old checkpoint: {old.name}")

