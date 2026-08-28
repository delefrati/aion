"""CLI entry point for llm_lab operations.

Usage:
    python -m llm_lab.cli download --target <path> --preset small
    python -m llm_lab.cli prepare --src-dir <path> --out-dir <path>
    python -m llm_lab.cli tokenizer --corpus <path> --out <path> [--vocab-size 16384]
    python -m llm_lab.cli train --config <path>
    python -m llm_lab.cli eval --checkpoint <path> --config <path>
    python -m llm_lab.cli generate --checkpoint <path> --config <path> --prompt "..."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_download(args):
    from llm_lab.data.download import (
        PRESETS, download_slimpajama_subset, download_dolly15k,
        download_no_robots, download_openassistant_guanaco, download_oasst1,
        download_ultrachat, download_hh_rlhf, download_wikipedia,
    )

    out_dir = Path(args.target)
    preset = PRESETS[args.preset]

    print(f"Preset: {args.preset}")
    print(f"Target dir: {out_dir}")
    print()

    if preset.get("wikipedia_mb", 0) > 0:
        download_wikipedia(out_dir, target_mb=preset["wikipedia_mb"])
    if preset.get("slimpajama_mb", 0) > 0:
        download_slimpajama_subset(out_dir, target_mb=preset["slimpajama_mb"])
    if preset.get("dolly", False):
        download_dolly15k(out_dir)
    if preset.get("no_robots", False):
        download_no_robots(out_dir)
    if preset.get("guanaco", False):
        download_openassistant_guanaco(out_dir)
    if preset.get("oasst1", False):
        download_oasst1(out_dir)
    if preset.get("ultrachat", False):
        download_ultrachat(out_dir, max_examples=preset.get("ultrachat_max", 50000))
    if preset.get("hh_rlhf", False):
        download_hh_rlhf(out_dir)

    print("\nDone.")


def cmd_pull_cache(args):
    from llm_lab.data.remote import fetch

    ok = fetch(args.repo, args.tag, Path(args.dir), token=args.token)
    if not ok:
        print("No release cache available; caller should rebuild from source.")
    return 0 if ok else 1


def cmd_push_cache(args):
    from llm_lab.data.remote import publish

    publish(args.repo, args.tag, Path(args.dir), args.files,
            title=args.title, notes=args.notes, part_size_mb=args.part_size_mb)


def cmd_prepare(args):
    from llm_lab.data.pipeline import ingest_raw_texts, dedup_lines, train_val_split

    src = Path(args.src_dir)
    out = Path(args.out_dir)

    print(f"Ingesting from {src}...")
    cleaned = ingest_raw_texts(src, out)
    print(f"Cleaned {len(cleaned)} files")

    corpus_path = out / "corpus.txt"
    n = dedup_lines(cleaned, corpus_path)
    print(f"Deduped to {n} unique lines -> {corpus_path}")

    manifest = train_val_split(corpus_path, out / "splits", val_ratio=args.val_ratio)
    print(f"Split: {manifest['train_lines']} train, {manifest['val_lines']} val")
    print(f"Manifest: {out / 'splits' / 'manifest.json'}")


def cmd_merge_chat(args):
    """Merge downloaded chat datasets into a single chat_merged.json."""
    from llm_lab.data.download import merge_chat_datasets

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    seed_path = Path(args.seed) if args.seed else None

    max_per_source = {}
    if getattr(args, 'max_hh_rlhf', None):
        max_per_source['hh_rlhf'] = args.max_hh_rlhf
    if getattr(args, 'max_ultrachat', None):
        max_per_source['ultrachat'] = args.max_ultrachat

    print(f"Merging chat datasets from {raw_dir}...")
    merge_chat_datasets(raw_dir, out_path, seed_path=seed_path, max_per_source=max_per_source or None)


def cmd_tokenizer(args):
    from llm_lab.tokenizer.bpe import train_bpe

    corpus = Path(args.corpus)
    out = Path(args.out)

    print(f"Training BPE tokenizer (vocab_size={args.vocab_size})...")
    tok = train_bpe([corpus], out, vocab_size=args.vocab_size)
    print(f"Saved to {out}, vocab size: {tok.get_vocab_size()}")

    # quick test
    test = "Hello world! This is AION."
    encoded = tok.encode(test)
    print(f"Test: '{test}' -> {len(encoded.ids)} tokens -> '{tok.decode(encoded.ids)}'")


def cmd_train(args):
    from llm_lab.training.config import TrainConfig
    from llm_lab.training import trainer

    cfg = TrainConfig.load(Path(args.config))
    print(f"Training {cfg.model_type} for {cfg.max_steps} steps...")
    cores = getattr(cfg, "tpu_cores", 1)
    gpus = getattr(cfg, "gpus", 1)
    if trainer.HAS_XLA and cores != 1:
        trainer.train_multicore(cfg)
        print("Done (multi-core TPU run — see master logs above).")
    elif not trainer.HAS_XLA and gpus != 1:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            trainer.train_ddp(cfg)
            print("Done (multi-GPU DDP run — see rank-0 logs above).")
        else:
            result = trainer.train(cfg)
            print(f"Done. Final val_loss={result.get('final_val_loss', float('nan')):.4f}, "
                  f"params={result.get('param_count', 0):,}")
    else:
        result = trainer.train(cfg)
        print(f"Done. Final val_loss={result.get('final_val_loss', float('nan')):.4f}, "
              f"params={result.get('param_count', 0):,}")


def cmd_eval(args):
    import torch
    from llm_lab.training.config import TrainConfig
    from llm_lab.training.trainer import load_model
    from llm_lab.eval.metrics import perplexity
    from llm_lab.tokenizer.bpe import load_tokenizer
    from llm_lab.utils import pick_device

    cfg = TrainConfig.load(Path(args.config))
    device = pick_device()

    model = load_model(cfg, args.checkpoint, device)

    ppl = perplexity(model, Path(cfg.val_path), Path(cfg.tokenizer_path), cfg.seq_len)
    print(f"Perplexity on val set: {ppl:.2f}")


def cmd_generate(args):
    import torch
    from llm_lab.training.config import TrainConfig
    from llm_lab.training.trainer import load_model
    from llm_lab.eval.metrics import generate
    from llm_lab.tokenizer.bpe import load_tokenizer
    from llm_lab.utils import pick_device

    cfg = TrainConfig.load(Path(args.config))
    device = pick_device()

    model = load_model(cfg, args.checkpoint, device)

    tok_path = Path(args.tokenizer) if args.tokenizer else Path(cfg.tokenizer_path)
    tokenizer = load_tokenizer(tok_path)

    # Stop at the chat end marker so we don't ramble into a fake next turn. The trainer's
    # <|end|> isn't the same token as generate()'s built-in <eos>, so pass it explicitly.
    stop_sequences = None
    if args.stop_token:
        _stop_ids = tokenizer.encode(args.stop_token).ids
        if _stop_ids:
            stop_sequences = [_stop_ids]

    output = generate(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
        stop_sequences=stop_sequences,
    )
    print(output)


def cmd_finetune(args):
    """Instruction-tune a pretrained model via the shared trainer.

    Seeds the checkpoint dir from a pretrained checkpoint (fresh optimizer), runs the
    standard training loop with an instruction dataset, and exports a serving bundle
    with the config embedded (so the backend provider can load it directly).
    """
    import torch
    from dataclasses import asdict

    from llm_lab.training.config import TrainConfig
    from llm_lab.training import trainer

    cfg = TrainConfig.load(Path(args.config))
    if getattr(cfg, "dataset_type", "text") == "text":
        cfg.dataset_type = "instruction"
    if args.data:
        cfg.instruction_data = args.data

    ckpt_dir = Path(cfg.checkpoint_dir)
    latest = ckpt_dir / "latest.pt"
    pretrain = args.pretrain or cfg.pretrain_checkpoint
    if not latest.exists() and pretrain and Path(pretrain).is_file():
        trainer.seed_checkpoint(Path(pretrain), latest)
        print(f"Seeded fine-tune start from {pretrain} (fresh optimizer)")
    elif not latest.exists():
        print(f"No pretrained checkpoint at {pretrain!r}; training from scratch")

    result = trainer.train(cfg)
    print(f"Done. best_val_loss={result.get('best_val_loss', float('nan')):.4f}")

    # Export serving bundle with embedded config (provider reads config from bundle)
    best = ckpt_dir / "best.pt"
    src = best if best.exists() else latest
    bundle = torch.load(src, map_location="cpu", weights_only=False)
    export_path = ckpt_dir / "model_serving.pt"
    torch.save({"model": bundle["model"], "config": asdict(cfg)}, export_path)
    print(f"Serving bundle saved: {export_path}")


def main():
    parser = argparse.ArgumentParser(prog="llm_lab", description="AION LLM Lab CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    from llm_lab.data.download import PRESETS
    p = sub.add_parser("download", help="Download training datasets from HuggingFace")
    p.add_argument("--target", required=True, help="Output directory for raw data")
    p.add_argument("--preset", choices=PRESETS.keys(), default="medium")

    # pull-cache — fetch tokenized cache from a GitHub Release (token needed if private)
    p = sub.add_parser("pull-cache", help="Fetch tokenized cache from a GitHub Release")
    p.add_argument("--repo", required=True, help="owner/repo, e.g. <owner>/aion-datasets")
    p.add_argument("--tag", required=True, help="release tag, e.g. pretrain-tokenized-v1")
    p.add_argument("--dir", required=True, help="destination directory for the cache")
    p.add_argument("--token", default=None,
                   help="GitHub token for a private repo (else GITHUB_TOKEN/GH_TOKEN env)")

    # push-cache — publish tokenized cache as a GitHub Release (needs `gh auth login`)
    p = sub.add_parser("push-cache", help="Publish tokenized cache as a GitHub Release")
    p.add_argument("--repo", required=True, help="owner/repo, e.g. <owner>/aion-datasets")
    p.add_argument("--tag", required=True, help="release tag, e.g. pretrain-tokenized-v1")
    p.add_argument("--dir", required=True, help="directory containing the files")
    p.add_argument("--files", nargs="+", default=["train.bin", "val.bin", "tokenizer.json"])
    p.add_argument("--title", default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--part-size-mb", type=int, default=1900)

    # prepare
    p = sub.add_parser("prepare", help="Ingest, clean, dedup, and split data")
    p.add_argument("--src-dir", required=True, help="Directory with raw .txt files")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--val-ratio", type=float, default=0.05)

    # tokenizer
    p = sub.add_parser("tokenizer", help="Train BPE tokenizer")
    p.add_argument("--corpus", required=True, help="Path to corpus .txt")
    p.add_argument("--out", required=True, help="Output tokenizer .json path")
    p.add_argument("--vocab-size", type=int, default=16384)

    # train
    p = sub.add_parser("train", help="Train a model")
    p.add_argument("--config", required=True, help="Path to train config .yaml")

    # eval
    p = sub.add_parser("eval", help="Evaluate a checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)

    # generate
    p = sub.add_parser("generate", help="Generate text from a checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--tokenizer", default=None, help="Override tokenizer path from config")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling threshold (0 disables)")
    p.add_argument("--repetition-penalty", type=float, default=1.3, help="Penalty for repeated tokens (1.0 disables)")
    p.add_argument("--stop-token", default="<|end|>", help="Stop generation at this token/string (empty to disable)")

    # finetune
    p = sub.add_parser("finetune", help="Instruction-tune a pretrained model")
    p.add_argument("--config", required=True, help="Path to instruct config .yaml")
    p.add_argument("--pretrain", default=None, help="Override pretrained checkpoint path")
    p.add_argument("--data", default=None, help="Override instruction data path")

    # merge-chat
    p = sub.add_parser("merge-chat", help="Merge downloaded chat datasets into chat_merged.json")
    p.add_argument("--raw-dir", required=True, help="Directory containing downloaded JSON files")
    p.add_argument("--out", required=True, help="Output path for chat_merged.json")
    p.add_argument("--seed", default=None, help="Optional path to instruction_seed.json for AION-specific examples")
    p.add_argument("--max-hh-rlhf", type=int, default=None, help="Cap hh-rlhf examples (e.g. 20000)")
    p.add_argument("--max-ultrachat", type=int, default=None, help="Cap ultrachat examples (e.g. 50000)")
    args = parser.parse_args()
    commands = {
        "download": cmd_download,
        "pull-cache": cmd_pull_cache,
        "push-cache": cmd_push_cache,
        "prepare": cmd_prepare,
        "merge-chat": cmd_merge_chat,
        "tokenizer": cmd_tokenizer,
        "train": cmd_train,
        "eval": cmd_eval,
        "generate": cmd_generate,
        "finetune": cmd_finetune,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
