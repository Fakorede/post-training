# GPU smoke test

Checks the LONI GPU stack: trains a LoRA adapter on base Llama 3.2 1B for a few steps with TRL, saves it, reloads it and compares the outputs. The training data is made up, so this checks the setup, not model quality. The adapter is discarded.

Needs the environment from the [root README](../README.md), access to the gated model and a saved Hugging Face read token.

From this folder:

    sbatch -A YOUR_ALLOCATION -p YOUR_GPU_PARTITION -N NODES -n NTASKS -c CPUS_PER_TASK --gres=GPU_REQUEST -t TIME_LIMIT slurm/run_gpu.sbatch

The log is `logs/smoke-gpu-<jobid>.out`. A pass ends with `SMOKE TEST PASSED`. Note the GPU name and peak memory printed above it.

The memory reading applies to the default batch size 8, sequence length 512 and LoRA rank 8 on attention layers.
