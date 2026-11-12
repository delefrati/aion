"""Shared fixtures: a tiny tokenizer, corpus and transformer config that train on a CPU in
seconds. Nothing here downloads anything or needs a GPU."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

WORDS = ("the cat dog sat ran on a mat log and saw big small red blue green tree house "
         "river stone bird sings quietly loudly every morning night day sun moon star").split()


def _text(n_lines: int, seed: int) -> str:
    rng = random.Random(seed)
    return "\n".join(" ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 14)))
                     for _ in range(n_lines)) + "\n"


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> dict[str, Path]:
    """train.txt / val.txt / tokenizer.json in one directory, built once per test session."""
    from llm_lab.tokenizer.bpe import train_bpe
    d = tmp_path_factory.mktemp("corpus")
    (d / "train.txt").write_text(_text(400, 1))
    (d / "val.txt").write_text(_text(60, 2))
    train_bpe([d / "train.txt"], d / "tokenizer.json", vocab_size=128, min_frequency=1)
    return {"dir": d, "train": d / "train.txt", "val": d / "val.txt", "tokenizer": d / "tokenizer.json"}


@pytest.fixture
def tiny_cfg(corpus, tmp_path):
    """A ~20k-parameter transformer that trains a step in a few milliseconds."""
    from llm_lab.training.config import TrainConfig
    return TrainConfig(
        model_type="transformer", vocab_size=128, d_model=32, n_layers=2, n_heads=2, seq_len=16,
        batch_size=4, lr=1e-3, warmup_steps=2, max_steps=6, checkpoint_every=3, eval_every=3,
        log_every=1, max_eval_batches=2, num_workers=0, use_8bit_optim=False,
        checkpoint_keep_last=1, seed=0,
        train_path=str(corpus["train"]), val_path=str(corpus["val"]),
        tokenizer_path=str(corpus["tokenizer"]), checkpoint_dir=str(tmp_path / "ckpt"),
    )
