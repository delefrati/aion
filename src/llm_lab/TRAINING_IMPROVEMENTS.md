# Training Improvements — Path to a Conversational Chatbot

## Current state (2026-08-07)
- Model: Mamba, d_model=256, n_layers=6, vocab=4096 (~5M params, 42MB)
- Training data: ~500 seed Q&A examples (single-turn) → **now replaced by OASST1 + dolly-15k pipeline**
- Active run: step ~2200, val loss ~4.42 (still declining, no overfitting yet)
- Target: val loss < 3.0 for marginally coherent text

### Done (2026-08-07)
- `trainer.py`: metrics.json now saved on every checkpoint (not just at end)
- `download.py`: added `download_oasst1()` with score filtering (skip negative-rated assistant turns, require 4+ messages)
- `download.py`: added `merge_chat_datasets()`, `merge-chat` CLI command
- `download.py`: re-added `hh-rlhf` (~160k conversations) to `chat` preset
- `chat` preset: oasst1 + hh-rlhf + dolly-15k
- Both configs: d_model=512, n_layers=12, d_state=32, vocab=16384, seq_len=1024
- `mamba_chat.yaml`: curriculum learning `256→2000 → 512→5000 → 1024→10000`
- Multi-session notebook: fixed resume step bug (max_steps = current_step + 5000)
- Both Colab notebooks updated to use new pipeline, read `GITHUB_USERNAME` from secrets

---

## 1. ✅ Training data — biggest bottleneck

**Status: Done.** OASST1 downloader and merge pipeline implemented.

- `python -m llm_lab.cli download --target /content/data/raw --preset chat` downloads oasst1 + dolly-15k
- `python -m llm_lab.cli merge-chat --raw-dir ... --out chat_merged.json --seed instruction_seed.json` merges all sources including AION identity examples
- Both Colab notebooks updated to use the new pipeline

**Dataset used:** [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) — ~9k best-path multi-turn conversations (English), + 15k dolly instruction examples + 500 AION identity Q&A pairs.

---

## 2. ✅ Model too small

**Status: Done.** Both `mamba_chat.yaml` and `mamba_chat_resume.yaml` updated.

- d_model: 256 → **512**
- n_layers: 6 → **12**
- batch_size: 64 → **16**, grad_accum_steps: 1 → **4** (effective batch still 64)
- Estimated size: ~40–50M params (~160MB checkpoint)

---

## 3. ✅ Vocabulary too small

**Status: Done.** vocab_size updated to 16384 in both configs and tokenizer training step.

- Both configs: vocab_size 4096 → **16384**
- Tokenizer will be retrained on the OASST1 + dolly-15k corpus automatically on the next Colab run
- Requires training from scratch (not compatible with old checkpoints — intentional)

---

## 4. ✅ Sequence length

**Status: Done.** `seq_len` updated to 1024 in both configs. Curriculum learning added to smooth the transition.

- Both configs: seq_len 512 → **1024**
- `mamba_chat.yaml` curriculum: `256:2000 → 512:5000 → 1024:10000` (starts short, builds up)
- batch_size: 16 → **8**, grad_accum: 4 → **8** (effective batch stays 64, avoids OOM)

---

## 5. ✅ More training steps

**Status: Done.** Configs updated.

- `mamba_chat.yaml`: max_steps 3000 → **10000**, warmup 200 → 500
- `mamba_chat_resume.yaml`: max_steps 3000 → **5000** per session (~3h on T4 at 50M params)
- Multisession notebook total target: 50000 → **20000** steps (~4 sessions)

---

## Recommended next run (priority order)

| # | Change | Impact | Status |
|---|--------|--------|--------|
| 1 | Download OASST1 + dolly-15k, build multi-turn dataset | High | ✅ Done |
| 2 | d_model=512, n_layers=12 | High | ✅ Done |
| 3 | Retrain tokenizer with vocab=16384 | Medium | ✅ Done |
| 4 | seq_len=1024 | Medium | ✅ Done |
| 5 | max_steps=10000 | Low without items 1–3 | ✅ Done |
