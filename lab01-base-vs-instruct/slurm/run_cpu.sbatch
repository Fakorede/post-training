#!/bin/bash
#SBATCH --job-name=lab01-cpu
#SBATCH --output=logs/%x-%j.out

# resource options are passed to sbatch as flags (see ../README.md); submit from the lab folder
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../env/activate.sh
uv run --no-sync python generate.py
