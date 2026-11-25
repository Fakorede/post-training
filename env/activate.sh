# source env/activate.sh
module load python/3.11.5-anaconda
export PATH="$HOME/.local/bin:$PATH"
# keep the environment and caches off the 10 GB /home
export UV_PROJECT_ENVIRONMENT=/work/$USER/venvs/post-training-labs
export UV_CACHE_DIR=/work/$USER/uv-cache
export UV_PYTHON_INSTALL_DIR=/work/$USER/uv-python
export HF_HOME=/work/$USER/hf-cache
