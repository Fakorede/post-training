# Lab 1: base vs. instruct, preference vs. correctness

Compares `SmolLM2-135M` with `SmolLM2-135M-Instruct`, then checks whether your preferred
response is also a correct one. The two checkpoints differ by all of their post-training
combined, so do not attribute a difference to any single stage.

## Steps

1. From this folder, submit the job. The scripts set only the job name and log path, so pass
   the resources as flags, **before** the script name:

       sbatch -A YOUR_ALLOCATION -p YOUR_CPU_PARTITION -N NODES -n NTASKS -c CPUS_PER_TASK \
         -t TIME_LIMIT slurm/run_cpu.sbatch

   To run on a GPU instead:

       sbatch -A YOUR_ALLOCATION -p YOUR_GPU_PARTITION -N NODES -n NTASKS -c CPUS_PER_TASK \
         --gres=gpu:GPU_COUNT -t TIME_LIMIT slurm/run_gpu.sbatch

   Record the flags you used with your results; `run_meta.json` does not save all of them.

2. First read `results/run-<jobid>/rank_sheet.md` and fill in `rankings.json`, best to
   worst, for example `"e1": [3, 1, 4, 2]`. Do not look at the answers in `prompts.json` first.
3. Then read `comparison.md` (base vs. instruct, which includes more answers).
4. Score:

       uv run --no-sync python score.py results/run-<jobid>

5. Read each response yourself and note where the checker is wrong, then record the results
   in your Week 1 notes.

## Limits

- The checker takes the number after "answer is" or "answer:", otherwise the last number.
  It can pass a wrong response or fail a right one. It does not understand number words.
- The base model gets the raw prompt; the instruct model gets its chat template.
- A seed does not guarantee identical samples across hardware or library versions.
- Do not execute model-generated code.

## Tests (no model needed)

    uv run --no-sync python -m unittest discover -s tests
