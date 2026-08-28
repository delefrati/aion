# Scaling AION from SLM to LLM

> **TL;DR** — Architecturally, AION's code is already ~80% LLM-grade. The hard part is **not**
> the model: it's compute, data volume, parallelism, and serving — all of which collide with this
> project's cost-first, weak-hardware charter. A from-scratch LLM cannot be trained under the
> current doctrine. Fine-tuning an existing open LLM (QLoRA) is the realistic path.

AION is a **Small Language Model (SLM)** lab. This document analyzes what it would take to push it
into true **LLM** territory, what code would have to change, and the pragmatic alternatives.

---

## 1. Where we are vs. where "LLM" starts

| | Current (SLM) | LLM target |
|---|---|---|
| Largest model | 235M params (`src/llm_lab/configs/transformer_tpu_large.yaml`) | 7B – 70B+ |
| Vocab | 16,384 | 32k – 128k |
| Context length | 1,024 | 8k – 128k |
| Training tokens | Wikipedia + SlimPajama subsets | Trillions (RedPajama/FineWeb-scale) |
| Hardware | 1–2× T4, TPU v3-8 / v5e-8 | A100/H100 cluster |
| Parallelism | DDP (data-parallel) | FSDP/ZeRO + tensor/pipeline |

A 7B model is a **30× jump** over the current largest; 70B is **~300×**. Scaling laws make the
cost superlinear in practice.

## 2. The real wall: compute & data

Using Chinchilla-optimal ratios (~20 tokens/param) and training FLOPs ≈ `6 × N × D`:

- **7B model** → ~140B training tokens → ~`6 × 10^21` FLOPs.
- On a single T4 (~20 usable TFLOPS) that is **years** of wall-clock time.
- On the current 2× T4 / TPU-v3-8 setup: still infeasible.
- Realistically needs a **cluster of A100/H100s for weeks** → tens of thousands of USD.

This directly violates the north-star constraints in `src/PLAN.md` ("strong behavior on low-power
hardware", "strict cost gates", "zero dependency on external model APIs"). **An LLM cannot be
trained from scratch under the project's current doctrine** — that is the honest blocker, not any
code detail.

## 3. Code changes actually required

The architecture in `src/llm_lab/models/transformer_lm.py` is already LLM-grade: RoPE, RMSNorm,
tied embeddings, and gradient checkpointing are the same ingredients as Llama. What breaks at scale:

| Area | Current state | Change needed for LLM |
|------|--------------|----------------------|
| **Token cache dtype** | `np.uint16` in `src/llm_lab/data/dataset.py` — caps vocab at 65,535 | Move to `uint32` (and bump BPE vocab to 32k–128k) |
| **Data loading** | Single `.bin` memmapped as one array | TB-scale **sharded streaming** dataset (WebDataset / Mosaic-style) + dedup + quality filtering |
| **Data volume** | Wikipedia + SlimPajama subsets | Trillions of tokens |
| **Parallelism** | DDP only — model must fit one device | 7B+ won't fit; need **FSDP / DeepSpeed ZeRO** (sharding), or tensor/pipeline parallelism (Megatron) |
| **Attention** | Plain scores, XLA materializes them | **FlashAttention / paged attention** — mandatory for long-seq efficiency |
| **Inference eff.** | Full multi-head attention | **GQA/MQA** to shrink the KV cache at serving |
| **Precision** | fp32/mixed | bf16 throughout (grad-accum + checkpointing already present) |

### Post-training gap
SFT exists (the `chat_*` fine-tune notebooks), but LLM alignment needs **RLHF or DPO**, which
`src/PLAN.md` explicitly excludes. Add a preference-tuning stage + reward modeling.

### Serving gap
Current providers (`deterministic`, `local-mamba`, `hf-local`) and nano/SQLite mode cannot host a
7B+ model. Need **quantization + a real inference engine**: llama.cpp / GGUF (Q4/Q5) for weak
hardware, or vLLM for throughput. This is the one place the "runs on weak hardware" goal survives —
a **4-bit quantized 7B runs on a decent laptop for inference**, even though you can't *train* it
there.

## 4. Difficulty verdict

- **Model architecture — Easy.** Bump `d_model` / `n_layers`; the code already scales.
- **Training infra — Hard.** FSDP/ZeRO sharding, streaming data, scale checkpointing: weeks of work.
- **Compute + data — Very hard / expensive.** The actual wall. Not solvable on free Colab/Kaggle.
- **Alignment + serving — Medium.** Standard but non-trivial additions.

## 5. Pragmatic middle paths

1. **QLoRA on an existing open LLM** (Llama-3-8B, Qwen, Mistral) — LLM behavior on a single 24 GB
   GPU, low cost, reuses the existing chat-tuning pipeline. **Most realistic path within the
   constraints.**
2. **Scale the SLM to ~1–3B from scratch** — reachable with a modest cloud budget, meaningfully
   more capable, still self-trained (stays true to the lab's charter).
3. **Full from-scratch 7B+** — only with serious dedicated compute; requires the explicit doctrine
   change flagged in `src/PLAN.md` (external-API / cost-gate revision).

---

*Related docs: `README.md` (SLM positioning), `src/PLAN.md` (LLM learning scope & constraints).*
