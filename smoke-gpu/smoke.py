"""Check that the TRL + PEFT stack can train, save and reload a LoRA adapter on one GPU."""
import argparse
import gc
import tempfile

import datasets
import peft
import torch
import transformers
import trl
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--allow-cpu", action="store_true", help="for local testing only")
    args = parser.parse_args()
    revision = args.revision or (DEFAULT_REVISION if args.model == DEFAULT_MODEL else None)

    for module in (torch, transformers, trl, peft, datasets):
        print(module.__name__, module.__version__)

    cuda = torch.cuda.is_available()
    assert cuda or args.allow_cpu, "PyTorch cannot see a GPU"
    if cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"gpu: {props.name}, {props.total_memory / 2**30:.1f} GiB")
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float32

    def load_base():
        model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, dtype=dtype)
        return model.to("cuda" if cuda else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # long made-up completions so every batch reaches max_length, for a realistic memory reading
    rows = {
        "prompt": ["Say hello.\n"] * args.batch_size,
        "completion": ["hello " * args.max_length] * args.batch_size,
    }

    with tempfile.TemporaryDirectory() as tmp:
        config = SFTConfig(
            output_dir=f"{tmp}/trainer",
            max_steps=args.steps,
            per_device_train_batch_size=args.batch_size,
            max_length=args.max_length,
            learning_rate=2e-4,
            bf16=bf16,
            fp16=False,
            use_cpu=not cuda,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            seed=0,
        )
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        trainer = SFTTrainer(
            model=load_base(),
            args=config,
            train_dataset=Dataset.from_dict(rows),
            processing_class=tokenizer,
            peft_config=lora,
        )
        trainer.model.print_trainable_parameters()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        loss = trainer.train().training_loss
        assert torch.isfinite(torch.tensor(loss)), f"loss is not finite: {loss}"
        if cuda:
            print(f"peak training memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated, "
                  f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved")

        model = trainer.model.eval()
        assert any(p.abs().sum() > 0 for n, p in model.named_parameters() if "lora_B" in n), "adapter did not update"
        inputs = tokenizer("Say hello.\n", return_tensors="pt").to(model.device)
        with torch.no_grad():
            before = model(**inputs).logits.float().cpu()
        model.save_pretrained(f"{tmp}/adapter")

        del trainer, model
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()

        reloaded = PeftModel.from_pretrained(load_base(), f"{tmp}/adapter").eval()
        with torch.no_grad():
            after = reloaded(**inputs).logits.float().cpu()
        diff = (before - after).abs().max().item()
        assert diff < 1e-2, f"reloaded adapter gives different logits (max difference {diff})"

    print(f"SMOKE TEST PASSED (mean loss {loss:.3f}, reload max logit difference {diff:.2e})")


if __name__ == "__main__":
    main()
