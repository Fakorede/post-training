#!/bin/bash
#SBATCH --job-name=lab01-gpu
#SBATCH --output=logs/%x-%j.out

# resource options are passed to sbatch as flags (see ../README.md); submit from the lab folder
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ../env/activate.sh
nvidia-smi
uv run --no-sync python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'"
uv run --no-sync python generate.py
