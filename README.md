# Post-training labs


## Set up the environment on LONI (once)

Start a compute node session:

    srun -A YOUR_ALLOCATION -p YOUR_CPU_PARTITION -N NODES -n NTASKS -c CPUS_PER_TASK -t TIME_LIMIT --pty bash

From the repo root, install the environment and run the tests:

    bash env/setup_loni.sh
    source env/activate.sh
    cd lab01-base-vs-instruct
    uv run --no-sync python -m unittest discover -s tests
    exit

## Later sessions

From the repo root:

    source env/activate.sh

If `uv.lock` changed or `/work` was purged, rerun `bash env/setup_loni.sh` on a compute node.

## Notes

- The environment, caches and model weights live on `/work`, not the 10 GB `/home`.
- Do not set `#SBATCH --mem`; the QB4 submit filter warns that it is unsupported.
