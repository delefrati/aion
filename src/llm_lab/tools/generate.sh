#!/usr/bin/env bash
# Generate from a trained run folder (best.pt + run.yaml + tokenizer.json) in one command.
#
# Usage:
#   ./generate.sh "What is machine learning?"                 # chat model, chat template
#   ./generate.sh -m aion-transformer-tpu-large --raw "Water is made of"   # base model, plain text
#   ./generate.sh -m ../some/other/run_dir -n 200 -t 0.6 "Explain TPUs"
#
# Options:
#   -m DIR   run folder: a name under training-data/ or a path (default: aion-transformer-chat-large)
#   -c NAME  checkpoint file inside DIR (default: best.pt)
#   -n N     max new tokens (default: 128)
#   -t T     temperature (default: 0.7)
#   -k K     top-k (default: 40)
#   -r R     repetition penalty (default: 1.1)
#   --raw    send the prompt as-is. Without it the prompt is wrapped in the chat template
#            (<|user|>...<|end|>\n<|assistant|>), which chat models need; base models want --raw.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../.." && pwd)"              # src/
DATA="$(cd "$SRC/.." && pwd)/training-data"

MODEL="aion-transformer-chat-large"
CKPT="best.pt"
MAX_TOKENS=128
TEMP=0.7
TOP_K=40
REP=1.1
RAW=0
PROMPT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m) MODEL="$2"; shift 2 ;;
    -c) CKPT="$2"; shift 2 ;;
    -n) MAX_TOKENS="$2"; shift 2 ;;
    -t) TEMP="$2"; shift 2 ;;
    -k) TOP_K="$2"; shift 2 ;;
    -r) REP="$2"; shift 2 ;;
    --raw) RAW=1; shift ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    -*) echo "unknown option: $1 (see --help)"; exit 2 ;;
    *) PROMPT="$1"; shift ;;
  esac
done

[[ -n "$PROMPT" ]] || { echo "missing prompt (see --help)"; exit 2; }

# A bare name resolves under training-data/; anything else is taken as a path.
if [[ -d "$MODEL" ]]; then DIR="$(cd "$MODEL" && pwd)"; else DIR="$DATA/$MODEL"; fi
for f in "$CKPT" run.yaml tokenizer.json; do
  [[ -f "$DIR/$f" ]] || { echo "missing $DIR/$f"; exit 1; }
done

if [[ "$RAW" == 0 ]]; then
  PROMPT=$'<|user|>'"$PROMPT"$'<|end|>\n<|assistant|>'
fi

# Prefer the lab's venv (has torch/tokenizers); fall back to whatever python3 is on PATH.
PY="$SRC/llm_lab/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

cd "$SRC"
exec "$PY" -m llm_lab.cli generate \
  --checkpoint "$DIR/$CKPT" --config "$DIR/run.yaml" --tokenizer "$DIR/tokenizer.json" \
  --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMP" --top-k "$TOP_K" --repetition-penalty "$REP"
