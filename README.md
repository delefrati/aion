# AION LLM Lab

A cost-first conversational AI platform with a from-scratch language-model training lab. Runs on weak hardware with zero external API dependencies.

> **Note — this is an SLM (Small Language Model) lab, not an LLM one.**
> The term "LLM" here refers to the general field of language modeling. In practice AION trains
> **Small Language Models** (a few million up to a few hundred million parameters): the largest
> config is a 235M-param Transformer. SLMs are the right fit for this project's cost-first,
> weak-hardware, run-anywhere (CPU / single GPU / TPU) constraints — they are cheap to train,
> fast to run, can operate fully offline for privacy, and are quick to fine-tune for a task.
> **LLMs** (tens of billions to trillions of parameters, e.g. GPT-4-class models) trade that
> efficiency and controllability for broad general-purpose reasoning at a much higher compute cost,
> which is explicitly out of scope here.
>
> See [SCALING_TO_LLM.md](SCALING_TO_LLM.md) for what it would take to push AION into LLM territory.

## What's Inside

- **Chat frontend** — Vue 3 web UI with streaming responses
- **Backend** — FastAPI server with SSE streaming, SQLite (nano) or PostgreSQL (standard)
- **LLM Lab** — From-scratch Mamba/SSM **and** Transformer training pipeline with BPE tokenizer, runnable on CPU, CUDA GPU (single or multi-GPU DDP), and TPU
- **Cloud notebooks** — Ready-to-run training on Google Colab and Kaggle (T4 GPU and TPU v3-8)

## Operating Modes

| Mode | What it does | Command |
|------|-------------|---------|
| **Nano** (default) | Backend + frontend, SQLite, minimal resources | `docker compose up` |
| **Standard** | Adds PostgreSQL + BFF layer | `docker compose --profile standard up` |
| **Lab CPU** | Training pipeline on CPU | `docker compose --profile lab-cpu up` |
| **Lab GPU** | Training pipeline with GPU | `docker compose --profile lab-gpu up` |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/aion.git
cd aion/src

# 2. Configure
cp infra/.env.example .env
# Edit .env as needed (defaults work out of the box)

# 3. Run
docker compose up
```

Frontend: http://localhost:3900 — Backend: http://localhost:8900

## Requirements

- Docker + Docker Compose
- (Optional) NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for lab-gpu mode

## Running the LLM Lab CLI

All training-pipeline operations are exposed through a single command-line entry point,
`python -m llm_lab.cli <command>`. Run it from the `src/` directory (so the `llm_lab`
package is importable), after installing the lab dependencies:

```bash
cd src
pip install -e llm_lab        # installs the llm_lab package and its deps
python -m llm_lab.cli --help  # list all commands
```

### End-to-end local workflow

```bash
cd src

# 1. Download raw datasets (presets: small, medium, large, chat, pretrain)
python -m llm_lab.cli download --target data/raw --preset small

# 2. Ingest, clean, dedup, and split into train/val
python -m llm_lab.cli prepare --src-dir data/raw --out-dir data/prepared --val-ratio 0.05

# 3. Train a BPE tokenizer on the cleaned corpus
python -m llm_lab.cli tokenizer --corpus data/prepared/corpus.txt \
    --out data/prepared/tokenizer.json --vocab-size 16384

# 4. Pretrain a model from a YAML config (configs live in llm_lab/configs/)
python -m llm_lab.cli train --config llm_lab/configs/mamba_small.yaml

# 5. Evaluate a checkpoint (perplexity on the val set)
python -m llm_lab.cli eval --checkpoint checkpoints/best.pt \
    --config llm_lab/configs/mamba_small.yaml

# 6. Generate text from a checkpoint
python -m llm_lab.cli generate --checkpoint checkpoints/best.pt \
    --config llm_lab/configs/mamba_small.yaml \
    --prompt "What is machine learning?" \
    --max-tokens 128 --temperature 0.8 --top-k 40
```

### Chat fine-tuning

```bash
cd src

# Merge downloaded chat datasets into a single file
python -m llm_lab.cli merge-chat --raw-dir data/raw --out data/chat_merged.json

# Instruction-tune a pretrained checkpoint (exports a serving bundle)
python -m llm_lab.cli finetune --config llm_lab/configs/mamba_chat.yaml \
    --pretrain checkpoints/best.pt --data data/chat_merged.json
```

### Commands

| Command | Purpose |
|---------|---------|
| `download` | Download training datasets from HuggingFace (`--preset small\|medium\|large\|chat\|pretrain`) |
| `prepare` | Ingest, clean, dedup, and split raw `.txt` files into train/val |
| `tokenizer` | Train a BPE tokenizer on a corpus (`--vocab-size`, default 16384) |
| `merge-chat` | Merge downloaded chat datasets into a single `chat_merged.json` |
| `train` | Train a model from a YAML config (auto-detects CPU / multi-GPU DDP / multi-core TPU) |
| `finetune` | Instruction-tune a pretrained model and export a serving bundle |
| `eval` | Report perplexity of a checkpoint on the val set |
| `generate` | Generate text from a checkpoint (`--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--stop-token`) |
| `pull-cache` | Fetch a tokenized cache from a GitHub Release |
| `push-cache` | Publish a tokenized cache as a GitHub Release |

Run `python -m llm_lab.cli <command> --help` for the full argument list of any command.

## Training in the Cloud (Colab & Kaggle)

The `llm_lab/` directory ships ready-to-run notebooks for training on free accelerators.
Two model families are supported, both trained from scratch with the project's BPE tokenizer:

- **Mamba/SSM** — selective state-space model (optional Triton scan kernel on GPU)
- **Transformer** — decoder-only with RoPE, gradient checkpointing, and optional weight tying
  (the largest config is a 235M-param model: `d_model=1024`, 16 layers, 16 heads)

### Accelerators

- **GPU** — single T4, or **2× T4 via DistributedDataParallel (DDP/NCCL)** for ~1.6× throughput
- **TPU v3-8** — single-core, or multi-core data-parallel (`torch_xla`)

### Setup

1. Add notebook secrets:
   - `GITHUB_TOKEN` (PAT with repo read access) and `GITHUB_USERNAME` — to clone the repo
   - On Kaggle also add `KAGGLE_USERNAME` and `KAGGLE_KEY` — for checkpoint persistence
2. Pick a notebook (`colab_*` for Colab, `kaggle_*` for Kaggle):
   - `colab_pretrain_medium.ipynb` / `kaggle_pretrain_medium.ipynb` — Mamba pretraining on Wikipedia + SlimPajama
   - `colab_chat_multisession_medium.ipynb` / `kaggle_chat_multisession_medium.ipynb` — chat fine-tuning on OASST1 + Dolly
   - `colab_pretrain_tpu.ipynb` / `kaggle_pretrain_tpu.ipynb` — Transformer pretraining on TPU
   - `kaggle_pretrain_gpu_large.ipynb` — resume the 235M Transformer on 2× T4 GPU (DDP)
   - `colab_chat.ipynb` — lightweight single-session chat fine-tune
3. Select the matching runtime (GPU or TPU) and run all cells.

### Checkpoint persistence & resume

Training resumes automatically across sessions from the last saved checkpoint:

- **Colab** — checkpoints are saved to Google Drive
- **Kaggle** — checkpoints are pushed to a Kaggle Dataset. Each session trains a bounded
  number of steps (`STEPS_PER_SESSION`) and a background watcher banks every checkpoint
  mid-run, so progress survives a hard session stop or timeout.

Checkpoints are device-portable: a model trained on TPU can be resumed on GPU (and vice versa).

## Project Structure

```
src/
├── backend/          # FastAPI server (Python 3.12)
├── frontend/         # Vue 3 + Vite chat UI
├── bff/              # Backend-for-frontend (Node, standard mode)
├── llm_lab/          # Training pipeline
│   ├── models/       #   Mamba & transformer model definitions
│   ├── data/         #   Dataset download, merge, instruction generation
│   ├── tokenizer/    #   BPE tokenizer training
│   ├── training/     #   Trainer, config, model factory
│   ├── eval/         #   Evaluation metrics
│   └── configs/      #   Training YAML configs
├── infra/            # Infrastructure configs
└── docker-compose.yml
```

## Configuration

Copy `infra/.env.example` to `src/.env` and adjust:

| Variable | Default | Description |
|----------|---------|-------------|
| `AION_MODE` | `nano` | Operating mode: `nano`, `standard`, `lab` |
| `AION_PROVIDER` | `deterministic` | Model provider: `deterministic`, `local-mamba`, `hf-local` |
| `AION_PORT` | `8900` | Backend API port |
| `AION_MAMBA_MODEL_PATH` | — | Path to trained Mamba checkpoint |
| `AION_MAMBA_TOKENIZER_PATH` | — | Path to tokenizer.json |
| `AION_HF_MODEL_ID` | `SmolLM2-360M-Instruct` | HuggingFace model for hf-local provider |

## Pretrained Models

No pretrained weights are published yet. Use the Colab notebooks above to train your own.

## License

[MIT](LICENSE)
