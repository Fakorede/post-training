import argparse
import json
import os
import platform
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

HERE = Path(__file__).parent
BASE = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load(name, device):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    return tok, model


def generate(tok, model, prompt, chat, max_new_tokens, **kwargs):
    if chat:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    # a chat template already includes its special tokens
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=not chat).to(model.device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=pad, **kwargs)
    new = out[:, inputs["input_ids"].shape[1]:]
    return [t.strip() for t in tok.batch_decode(new, skip_special_tokens=True)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=BASE)
    p.add_argument("--instruct", default=INSTRUCT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    run = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or HERE / "results" / f"run-{run}"
    out.mkdir(parents=True, exist_ok=False)
    prompts = json.loads((HERE / "prompts.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    meta = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "models": {},
    }

    greedy = {}
    for label, name, chat in (("base", args.base, False), ("instruct", args.instruct, True)):
        tok, model = load(name, device)
        meta["models"][label] = {
            "name": name,
            "revision": getattr(model.config, "_commit_hash", None),
            "dtype": str(model.dtype),
            "parameters": sum(w.numel() for w in model.parameters()),
        }
        greedy[label] = {
            q["id"]: generate(tok, model, q["prompt"], chat, args.max_new_tokens, do_sample=False)[0]
            for q in prompts["open"] + prompts["exact"]
        }

    # tok and model are still the instruct pair here
    samples = {
        q["id"]: generate(
            tok, model, q["prompt"], True, args.max_new_tokens, do_sample=True,
            temperature=args.temperature, top_p=args.top_p, num_return_sequences=args.samples,
        )
        for q in prompts["exact"]
    }

    comparison = ["# Base vs. instruct (greedy decoding)", ""]
    for q in prompts["open"] + prompts["exact"]:
        comparison += [f"## {q['id']}: {q['prompt']}", ""]
        for label in ("base", "instruct"):
            comparison += [f"**{label}**", "", "````text", greedy[label][q["id"]], "````", ""]

    sheet = ["# Rank these before running score.py", "",
             "In rankings.json, list response numbers from best to worst.", ""]
    for q in prompts["exact"]:
        sheet += [f"## {q['id']}: {q['prompt']}", ""]
        for i, text in enumerate(samples[q["id"]], 1):
            sheet += [f"**Response {i}**", "", "````text", text, "````", ""]

    def write(name, data):
        (out / name).write_text(data if isinstance(data, str) else json.dumps(data, indent=2) + "\n")

    write("run_meta.json", meta)
    write("greedy.json", greedy)
    write("samples.json", samples)
    write("comparison.md", "\n".join(comparison))
    write("rank_sheet.md", "\n".join(sheet))
    write("rankings.json", {q["id"]: [] for q in prompts["exact"]})
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
