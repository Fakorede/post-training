#!/bin/bash
#SBATCH --job-name=smoke-gpu
#SBATCH --output=logs/%x-%j.out

# resource options are passed to sbatch as flags (see ../README.md); submit from the smoke-gpu folder
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../env/activate.sh
nvidia-smi
uv run --no-sync python smoke.py
