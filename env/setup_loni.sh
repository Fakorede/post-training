#!/bin/bash
# Run on a LONI compute node, from the repo root: bash env/setup_loni.sh
set -eo pipefail
source env/activate.sh
command -v uv >/dev/null || pip install --user uv
uv sync --locked
uv run --no-sync python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
