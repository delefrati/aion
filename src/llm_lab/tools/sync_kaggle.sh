#!/usr/bin/env bash
# Push repo notebooks to Kaggle kernels — the right way, no manual web upload.
#
# One-time setup:
#   pip install kaggle
#   export KAGGLE_USERNAME=... KAGGLE_KEY=...     # or put ~/.kaggle/kaggle.json in place
#   Fill the kernel short-names in kernels.map  (find them: kaggle kernels list --mine)
#
# Usage:
#   ./sync_kaggle.sh                              # push every mapped kernel
#   ./sync_kaggle.sh kaggle_chat_gpu_large.ipynb # push just one
#
# The Kaggle owner is read from your local creds (KAGGLE_USERNAME or
# ~/.kaggle/kaggle.json), so no username is hardcoded in the repo. For each kernel
# it PULLS the live metadata first, so the accelerator (GPU/TPU) and dataset
# attachments you configured in the Kaggle UI are preserved — only the notebook
# code is replaced with the current repo version.
set -euo pipefail
cd "$(dirname "$0")"

NB_DIR="$(cd .. && pwd)"        # llm_lab/
MAP="kernels.map"
BUILD=".build"
FILTER="${1:-}"

command -v kaggle >/dev/null || { echo "kaggle CLI not found — run: pip install kaggle"; exit 1; }

# Kaggle owner from local creds — never hardcoded in the repo.
OWNER="${KAGGLE_USERNAME:-}"
if [[ -z "$OWNER" && -f "$HOME/.kaggle/kaggle.json" ]]; then
  OWNER="$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])')"
fi
[[ -n "$OWNER" ]] || { echo "No Kaggle username — set KAGGLE_USERNAME or ~/.kaggle/kaggle.json"; exit 1; }

while read -r nb name _; do
  [[ -z "${nb:-}" || "$nb" == \#* ]] && continue
  [[ -n "$FILTER" && "$nb" != "$FILTER" ]] && continue
  if [[ "$name" == *REPLACE-ME* ]]; then
    echo "skip $nb — kernel name not set in $MAP (run: kaggle kernels list --mine)"; continue
  fi
  [[ -f "$NB_DIR/$nb" ]] || { echo "skip $nb — file missing"; continue; }
  slug="$OWNER/$name"

  d="$BUILD/${nb%.ipynb}"; rm -rf "$d"; mkdir -p "$d"
  # Pull live metadata so we keep the UI-set accelerator + dataset sources.
  kaggle kernels pull "$slug" -p "$d" -m >/dev/null
  cp "$NB_DIR/$nb" "$d/"
  python3 - "$d/kernel-metadata.json" "$nb" <<'PY'
import json, sys
p, nb = sys.argv[1], sys.argv[2]
m = json.load(open(p))
m["code_file"] = nb
m["language"] = "python"
m["kernel_type"] = "notebook"
json.dump(m, open(p, "w"), indent=2)
PY
  echo "==> pushing $nb -> $slug"
  kaggle kernels push -p "$d"
done < "$MAP"

echo "Done. New versions are DRAFTS — open each kernel and Save & Run to execute."
