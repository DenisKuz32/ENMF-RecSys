#!/usr/bin/env bash
# Обучение в один процесс.
#   ./scripts/run_train.sh [конфиг] [дополнительные --set переопределения]
set -euo pipefail
CONFIG="${1:-configs/ml1m.yaml}"
shift || true
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python -m enmf.cli train --config "$CONFIG" "$@"
