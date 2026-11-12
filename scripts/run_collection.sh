#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")/.."
: "${SERPER_API_KEY:?Set SERPER_API_KEY privately in the environment}"
checkpoint="${SFT_CHECKPOINT:-./checkpoints/qwen3-vl-8b-sft}"
work="${WORK:-./runs/collection}"
devices="${DEVICES:-cuda:0}"
py="${COLLECTION_PYTHON:-.venv/bin/python}"
"$py" collect.py prepare --work "$work"
exec "$py" -u collect.py run --checkpoint "$checkpoint" --work "$work" --devices "$devices" "$@"
