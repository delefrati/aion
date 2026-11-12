"""Shared fixtures: a tiny tokenizer, corpus and transformer config that train on a CPU in
seconds. Nothing here downloads anything or needs a GPU."""
from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

# CPU only, here and in the subprocesses tests start: the results must not depend on whether
# the machine has a GPU, and CI has none.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
    return TrainConfig.from_dict(dict(
        model_type="transformer", vocab_size=128, d_model=32, n_layers=2, n_heads=2, seq_len=16,
        batch_size=4, lr=1e-3, warmup_steps=2, max_steps=6, checkpoint_every=3, eval_every=3,
        log_every=1, max_eval_batches=2, num_workers=0, use_8bit_optim=False,
        checkpoint_keep_last=1, seed=0,
        train_path=str(corpus["train"]), val_path=str(corpus["val"]),
        tokenizer_path=str(corpus["tokenizer"]), checkpoint_dir=str(tmp_path / "ckpt"),
    ))


@pytest.fixture
def chat_cfg(tiny_cfg, tmp_path):
    """The tiny model finetuning on a tiny chat set: a run that keeps a best checkpoint."""
    import json
    chat = tmp_path / "chat.json"
    chat.write_text(json.dumps([{"messages": [{"role": "user", "content": "the cat"},
                                              {"role": "assistant", "content": "sat on the mat"}]}] * 40))
    return tiny_cfg.replace(dataset_type="chat", instruction_data=str(chat), train_path="", val_path="")


class FakeKaggle:
    """The fake kaggle CLI (tests/fake_kaggle.py) on PATH, with its Datasets in a temp dir."""

    def __init__(self, root: Path):
        self.root = root

    def script(self, **queues) -> None:
        """Queue failures: script(files=["500"], push=["401"], **{"download:x.bin": ["500"]})."""
        import json
        (self.root / "script.json").write_text(json.dumps(queues))

    def dataset(self, slug: str) -> Path:
        return self.root / "datasets" / slug.replace("/", "__")

    def calls(self) -> list[dict]:
        import json
        f = self.root / "calls.jsonl"
        return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


@pytest.fixture
def fake_kaggle(tmp_path, monkeypatch) -> FakeKaggle:
    import sys
    root = tmp_path / "fake_kaggle"
    (root / "datasets").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "kaggle"
    exe.write_text(f"#!{sys.executable}\nimport runpy, sys\n"
                   f"sys.argv[0] = {str(Path(__file__).with_name('fake_kaggle.py'))!r}\n"
                   f"runpy.run_path(sys.argv[0], run_name='__main__')\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_KAGGLE_ROOT", str(root))
    return FakeKaggle(root)
