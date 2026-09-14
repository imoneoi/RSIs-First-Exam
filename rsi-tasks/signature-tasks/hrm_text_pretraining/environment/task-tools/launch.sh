#!/usr/bin/env bash
# Profile-aware launcher. --dry-run prints the command without starting torchrun.
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export WANDB_MODE=offline
exec python "$(dirname "${BASH_SOURCE[0]}")/train.py" --launch "$@"
