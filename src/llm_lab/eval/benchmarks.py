"""Zero-shot multiple-choice benchmarks (HellaSwag, ARC, PIQA) scored by log-likelihood.

Nothing is sampled: for each question the model scores every answer choice as a
continuation of the prompt, and its pick is the choice with the highest log-probability.
Same weights -> same score, so runs are directly comparable across checkpoints.

Two accuracies per task, following lm-evaluation-harness:
  acc       argmax of the summed log-prob of the choice
  acc_norm  argmax of that sum divided by the choice's length in bytes (removes the bias
            toward short answers; the headline metric for HellaSwag, ARC-Challenge, PIQA)
Each comes with its binomial standard error, sqrt(p(1-p)/n): two checkpoints whose scores
differ by less than ~2 stderr are not distinguishable on that task.

Prompt formats match lm-evaluation-harness so scores sit on the same scale as published
numbers for GPT-2, Pythia and SmolLM (not identical: those use their own tokenizers).
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Item:
    context: str
    choices: list[str]  # continuations, each starting with the separating space
    label: int


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _load_hellaswag() -> list[Item]:
    from datasets import load_dataset
    items = []
    for row in load_dataset("Rowan/hellaswag", split="validation"):
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        items.append(Item(
            context=_hellaswag_clean(row["activity_label"] + ": " + ctx),
            choices=[" " + _hellaswag_clean(e) for e in row["endings"]],
            label=int(row["label"]),
        ))
    return items


def _load_arc(subset: str) -> list[Item]:
    from datasets import load_dataset
    items = []
    for row in load_dataset("allenai/ai2_arc", subset, split="test"):
        items.append(Item(
            context=f"Question: {row['question']}\nAnswer:",
            choices=[" " + t for t in row["choices"]["text"]],
            label=row["choices"]["label"].index(row["answerKey"]),
        ))
    return items


def _load_piqa() -> list[Item]:
    from datasets import load_dataset
    items = []
    # baber/piqa is the parquet copy of ybisk/piqa (whose loading script `datasets` no longer runs).
    for row in load_dataset("baber/piqa", split="validation"):
        items.append(Item(
            context=f"Question: {row['goal']}\nAnswer:",
            choices=[" " + row["sol1"], " " + row["sol2"]],
            label=int(row["label"]),
        ))
    return items


TASKS = {
    "hellaswag": _load_hellaswag,
    "arc_easy": lambda: _load_arc("ARC-Easy"),
    "arc_challenge": lambda: _load_arc("ARC-Challenge"),
    "piqa": _load_piqa,
}

# The metric quoted in the summary table (the one published results usually report).
HEADLINE = {"hellaswag": "acc_norm", "arc_easy": "acc", "arc_challenge": "acc_norm", "piqa": "acc_norm"}
CHANCE = {"hellaswag": 0.25, "arc_easy": 0.25, "arc_challenge": 0.25, "piqa": 0.5}


@torch.no_grad()
def _score(model, requests: list[tuple[list[int], list[int]]], seq_len: int,
           batch_size: int, device: torch.device) -> list[float]:
    """Summed log-prob of each (context_ids, continuation_ids) continuation.

    Context and continuation are tokenized separately and concatenated, so the choice's
    tokens are the same no matter what precedes it. Over-long inputs keep their right end
    (the continuation) and drop context from the left. Rows are right-padded: the model is
    causal, so padding after a row's last token can't change any position we read.
    """
    out = [0.0] * len(requests)
    # Longest first: similar lengths share a batch (less padding), and an OOM shows up on
    # the first batch rather than an hour in.
    order = sorted(range(len(requests)), key=lambda i: -len(requests[i][0]) - len(requests[i][1]))
    use_amp = device.type == "cuda"
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        rows, spans = [], []
        for i in idx:
            ctx, cont = requests[i]
            full = (ctx + cont)[-(seq_len + 1):]
            inp = full[:-1]
            n_cont = min(len(cont), len(inp))
            rows.append(inp)
            spans.append((len(inp) - n_cont, len(inp), full[len(full) - n_cont:]))
        width = max(len(r) for r in rows)
        x = torch.zeros((len(rows), width), dtype=torch.long)
        for r, row in enumerate(rows):
            x[r, :len(row)] = torch.tensor(row, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x.to(device))
        for r, (lo, hi, targets) in enumerate(spans):
            lp = torch.log_softmax(logits[r, lo:hi].float(), dim=-1)
            tgt = torch.tensor(targets, device=lp.device).unsqueeze(-1)
            out[idx[r]] = lp.gather(-1, tgt).sum().item()
    return out


def run_task(model, tokenizer, name: str, seq_len: int, device: torch.device,
             batch_size: int = 32, limit: int = 0) -> dict:
    """Score one task. limit > 0 keeps the first N questions (deterministic subset)."""
    items = TASKS[name]()
    if limit > 0:
        items = items[:limit]
    ctx_cache: dict[str, list[int]] = {}
    requests, owners = [], []
    for q, item in enumerate(items):
        if item.context not in ctx_cache:
            ctx_cache[item.context] = tokenizer.encode(item.context).ids
        for c in item.choices:
            requests.append((ctx_cache[item.context], tokenizer.encode(c).ids))
            owners.append(q)

    t0 = time.time()
    scores = _score(model, requests, seq_len, batch_size, device)

    per_q: list[list[float]] = [[] for _ in items]
    for q, s in zip(owners, scores):
        per_q[q].append(s)
    hits = hits_norm = 0
    for item, s in zip(items, per_q):
        norm = [v / max(1, len(c.encode("utf-8"))) for v, c in zip(s, item.choices)]
        hits += int(max(range(len(s)), key=s.__getitem__) == item.label)
        hits_norm += int(max(range(len(norm)), key=norm.__getitem__) == item.label)

    n = len(items)
    acc, acc_norm = hits / n, hits_norm / n
    return {
        "n": n,
        "acc": acc, "acc_stderr": math.sqrt(acc * (1 - acc) / n),
        "acc_norm": acc_norm, "acc_norm_stderr": math.sqrt(acc_norm * (1 - acc_norm) / n),
        "seconds": round(time.time() - t0, 1),
    }


def load_checkpoint_model(cfg, ckpt_path: Path, device: torch.device):
    """Build the model and load weights, mmap'd so the ~2.8GB latest.pt (which also carries
    optimizer state) isn't read into host RAM in full. Returns (model, step)."""
    from llm_lab.training.model_factory import build_model
    from llm_lab.training.trainer import _strip_prefixes
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(_strip_prefixes(ckpt["model"]))
    step = int(ckpt.get("step", 0))
    del ckpt
    return model.to(device).eval(), step


def format_table(results: dict[str, dict]) -> str:
    """Markdown table: one row per checkpoint label, headline metric +/- stderr per task."""
    tasks = [t for t in TASKS if any(t in r["tasks"] for r in results.values())]
    head = "| checkpoint | step | " + " | ".join(f"{t} ({HEADLINE[t]})" for t in tasks) + " |"
    lines = [head, "|" + "---|" * (len(tasks) + 2)]
    for label, r in results.items():
        cells = []
        for t in tasks:
            m = r["tasks"].get(t)
            k = HEADLINE[t]
            cells.append(f"{100 * m[k]:.1f} ± {100 * m[k + '_stderr']:.1f}" if m else "-")
        lines.append(f"| {label} | {r.get('step', '?'):,} | " + " | ".join(cells) + " |")
    lines.append("| random guess | | " + " | ".join(f"{100 * CHANCE[t]:.0f}" for t in tasks) + " |")
    return "\n".join(lines)
