#!/usr/bin/env bash
set -euo pipefail

TH02_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TH02_PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="${TH02_GPU:-3}"
export TMPDIR="$TH02_PROJECT_DIR/.cache/tmp"
export HF_HOME="$TH02_PROJECT_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR"

exec "$TH02_PROJECT_DIR/.venv/bin/python" -u "$TH02_PROJECT_DIR/scripts/run_server.py" \
  --data-dir "$TH02_PROJECT_DIR/data" \
  --output-dir "$TH02_PROJECT_DIR/artifacts" \
  --model-path "$TH02_PROJECT_DIR/models/all-MiniLM-L6-v2" \
  --device cuda \
  --seed 20260916 \
  "$@"
