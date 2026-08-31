#!/usr/bin/env bash
# Print "Open in Colab" links for every colab_*.ipynb, straight from GitHub.
#
# Colab has no upload/push: the right way is to open the notebook FROM GitHub so it
# always runs the committed version. Push your changes first (git push), then open a
# link below. The aion repo is PRIVATE, so the first time Colab will ask you to
# authorize GitHub access (File > Open notebook > GitHub tab > authorize).
# To save edits back: File > Save a copy in GitHub.
set -euo pipefail
cd "$(dirname "$0")"

OWNER_REPO="delefrati/aion"
BRANCH="main"
NB_DIR_REL="src/llm_lab"
NB_DIR="$(cd .. && pwd)"

for nb in "$NB_DIR"/colab_*.ipynb; do
  base="$(basename "$nb")"
  echo "https://colab.research.google.com/github/${OWNER_REPO}/blob/${BRANCH}/${NB_DIR_REL}/${base}"
done
