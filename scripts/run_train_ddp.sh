#!/usr/bin/env bash
# Обучение в режиме DDP.
#   ./scripts/run_train_ddp.sh 4 configs/ml1m.yaml
# Backend выбирается автоматически: nccl при CUDA, gloo на CPU и Intel XPU.
set -euo pipefail
NPROC="${1:-2}"
CONFIG="${2:-configs/ml1m.yaml}"
shift 2 || true
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29500}" \
    -m enmf.cli train --config "$CONFIG" "$@"
