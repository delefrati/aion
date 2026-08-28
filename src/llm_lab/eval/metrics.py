"""Evaluation: perplexity, token-level metrics, and prompt harness."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from llm_lab.data.dataset import TextDataset
from llm_lab.tokenizer.bpe import load_tokenizer


@torch.no_grad()
def perplexity(model, data_path: Path, tokenizer_path: Path, seq_len: int = 256, batch_size: int = 16) -> float:
    """Compute perplexity on a dataset."""
    device = next(model.parameters()).device
    tokenizer = load_tokenizer(Path(tokenizer_path))
    ds = TextDataset(data_path, tokenizer, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

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

    avg_loss = total_loss / max(n, 1)
    return math.exp(avg_loss)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.3,
    stop_sequences: list[list[int]] | None = None,
) -> str:
    """Autoregressive generation with top-k / top-p (nucleus) and repetition penalty.

    stop_sequences: token-id sequences (e.g. a multi-token "<|end|>" marker) that end
    generation when they appear as the tail of the output.
    """
    device = next(model.parameters()).device
    model.eval()

    encoded = tokenizer.encode(prompt)
    ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    eos_id = tokenizer.token_to_id("<eos>")

    for _ in range(max_tokens):
        logits = model(ids)
        logits = logits[:, -1, :]

        # Penalize tokens already present to break repetition loops
        if repetition_penalty and repetition_penalty != 1.0:
            for tok in set(ids[0].tolist()):
                if logits[0, tok] > 0:
                    logits[0, tok] /= repetition_penalty
                else:
                    logits[0, tok] *= repetition_penalty

        logits = logits / max(temperature, 1e-8)

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")

        # Nucleus (top-p): keep the smallest set of tokens whose cumulative prob >= top_p
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            logits[remove.scatter(1, sorted_idx, remove)] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)

        if eos_id is not None and next_id.item() == eos_id:
            break

        if stop_sequences:
            tail = ids[0].tolist()
            if any(len(s) > 0 and tail[-len(s):] == s for s in stop_sequences):
                break

    return tokenizer.decode(ids[0].tolist())


def run_prompt_suite(
    model,
    tokenizer,
    suite_path: Path,
    out_path: Path,
    max_tokens: int = 128,
    temperature: float = 0.8,
) -> list[dict]:
    """Run a suite of prompts and save results.

    Suite file is JSON: [{"id": "...", "prompt": "...", "category": "..."}]
    """
    suite = json.loads(suite_path.read_text())
    results = []

    for item in suite:
        t0 = time.time()
        output = generate(model, tokenizer, item["prompt"], max_tokens, temperature)
        elapsed = time.time() - t0

        results.append({
            "id": item["id"],
            "category": item.get("category", ""),
            "prompt": item["prompt"],
            "output": output,
            "tokens_generated": len(tokenizer.encode(output).ids) - len(tokenizer.encode(item["prompt"]).ids),
            "elapsed_s": round(elapsed, 3),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return results
