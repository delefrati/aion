#!/usr/bin/env bash
# Push repo notebooks to Kaggle kernels — the right way, no manual web upload.
#
# One-time setup:
#   pip install kaggle
#   export KAGGLE_USERNAME=... KAGGLE_KEY=...     # or put ~/.kaggle/kaggle.json in place
#   Fill the kernel short-names in kernels.map  (find them: kaggle kernels list --mine)
#
# Usage:
#   ./sync_kaggle.sh                                  # DRY-RUN: list what would push
#   ./sync_kaggle.sh -y                               # push+run every mapped kernel
#   ./sync_kaggle.sh kaggle_chat_gpu_large.ipynb -y   # push+run just one
#
# WARNING: `kaggle kernels push` has no draft-only mode — it also QUEUES A RUN with the
# notebook's DEFAULT cell flags. For a specific run (MODEL_SIZE/ADD_DATA), run in the UI.
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

# Push ALSO runs the kernel, so dry-run by default; -y/--yes actually pushes+runs.
DO_PUSH=0
FILTER=""
PUSHED=()
FAILED=()
for a in "$@"; do
  case "$a" in
    -y|--yes) DO_PUSH=1 ;;
    *.ipynb)  FILTER="$a" ;;
    *) echo "unknown arg: $a (expected a notebook name and/or -y)"; exit 2 ;;
  esac
done

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
  python3 - "$d/kernel-metadata.json" "$nb" "$OWNER" "$NB_DIR/runs.yaml" <<'PY'
import json, sys
p, nb, owner, runs = sys.argv[1:5]
m = json.load(open(p))
m["code_file"] = nb
m["language"] = "python"
m["kernel_type"] = "notebook"
# Attach the active phase's tokenized data (runs.yaml) as an input, so the session links it
# instead of downloading it. Inputs attached in the UI are kept.
import yaml
r = yaml.safe_load(open(runs))
run = (r.get("notebooks") or {}).get(nb[:-len(".ipynb")])
if run:
    src = f"{owner}/{r['phases'][run['phase']]['data']}"
    if src not in m.setdefault("dataset_sources", []):
        m["dataset_sources"].append(src)
        print(f"    attaching input {src} (runs.yaml: phase {run['phase']})")
json.dump(m, open(p, "w"), indent=2)
PY
  if [[ "$DO_PUSH" == 1 ]]; then
    echo "==> pushing (and RUNNING) $nb -> $slug"
    # The CLI exits 0 even when Kaggle refuses the push (e.g. "Maximum batch TPU session
    # count of 1 reached"), so success is read from its output, not its exit code.
    out="$(kaggle kernels push -p "$d" 2>&1)" || true
    echo "$out" | grep -v "outdated \`kaggle\` version" || true
    if grep -q "successfully pushed" <<<"$out"; then
      PUSHED+=("$nb")
    else
      FAILED+=("$nb")
    fi
  else
    echo "[dry-run] would push+run $nb -> $slug   (add -y to do it)"
  fi
done < "$MAP"

if [[ "$DO_PUSH" == 1 ]]; then
  if (( ${#PUSHED[@]} )); then
    echo "Pushed: ${PUSHED[*]}. Each push also QUEUED A RUN, which can't read Kaggle Secrets and"
    echo "fails in Cell 1 (after waiting in the queue): stop it under View Active Events, and start"
    echo "real runs from the UI (Save & Run All)."
  fi
  if (( ${#FAILED[@]} )); then
    echo "NOT pushed (Kaggle refused; the live kernel is unchanged): ${FAILED[*]}"
    exit 1
  fi
  (( ${#PUSHED[@]} )) || echo "Nothing pushed: no notebook matched."
else
  echo "Dry-run only — nothing pushed. Re-run with -y to push+run."
fi
