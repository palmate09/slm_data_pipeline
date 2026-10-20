#!/usr/bin/env bash
# Convenience runner for the SLM data pipeline.
#
# Usage:
#   ./run_pipeline.sh                  # resume from clean (no re-download)
#   ./run_pipeline.sh full             # full run: extract -> gate -> ... -> output
#   ./run_pipeline.sh resume gate      # resume from a specific stage
#   ./run_pipeline.sh test             # run offline tests
#   ./run_pipeline.sh audit            # audit final docs/chunks
#   ./run_pipeline.sh preview --kind chunks
#   ./run_pipeline.sh validate --strict
#   ./run_pipeline.sh --stage quality  # pass arguments through to the CLI
#
# Override the config:
#   CONFIG=configs/physics.yaml ./run_pipeline.sh full

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
CONFIG="${CONFIG:-configs/machine_learning.yaml}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Error: Python environment not found at $PYTHON"
  echo "Create it with: uv venv && uv pip install -e '.[dev]'"
  exit 1
fi

command="${1:-resume}"
if [[ $# -eq 0 ]]; then
  set -- resume
fi

case "$command" in
  full)
    shift
    exec "$PYTHON" -m slm_pipeline.cli --config "$CONFIG" --stage all "$@"
    ;;
  resume)
    shift
    stage="${1:-clean}"
    shift || true
    exec "$PYTHON" -m slm_pipeline.cli --config "$CONFIG" --stage all --resume-from "$stage" "$@"
    ;;
  test)
    shift
    exec "$PYTHON" -m pytest -q "$@"
    ;;
  audit|preview|validate)
    shift
    exec "$PYTHON" -m slm_pipeline.cli "$command" --config "$CONFIG" "$@"
    ;;
  *)
    exec "$PYTHON" -m slm_pipeline.cli --config "$CONFIG" "$@"
    ;;
esac
