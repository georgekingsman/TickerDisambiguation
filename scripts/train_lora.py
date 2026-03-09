"""
train_lora.py  –  LoRA fine-tuning for Ticker Disambiguation
                  股票代码消歧 LoRA 微调训练脚本

Usage / 用法:
    # Full run (default 3 epochs)
    python scripts/train_lora.py --config configs/lora_v1.yaml --seed 42

    # Smoke run (quick sanity check)
    python scripts/train_lora.py --config configs/lora_v1.yaml --seed 42 --smoke

Requirements / 依赖:
    pip install transformers torch peft pyyaml
"""

import argparse
import json
import re
import time
import yaml
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model


# ── Ticker normalizer (identical to run_zero_shot.py) ──
TICKER_SET = {
    "AAPL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "NFLX",
    "GOOGL", "GOOG", "BRK-A", "BRK-B", "JPM", "BAC",
    "AMD", "INTC", "PLTR", "DIS", "TSM", "BABA", "PDD",
}
_TICKER_RE = re.compile(r"\b([A-Z]{1,5}(?:[.\-/][A-Z0-9]{1,3})?)\b")


def normalize_ticker(raw: str) -> tuple[str, bool]:
    text = raw.strip().upper()
    clean = text.replace(".", "-").replace("/", "-")
    if clean in TICKER_SET:
        return clean, clean != text
    candidates = _TICKER_RE.findall(text)
    for candidate in candidates:
        normalized = candidate.replace(".", "-").replace("/", "-")
        if normalized in TICKER_SET:
            return normalized, True
    for ticker in TICKER_SET:
        if ticker in text:
            return ticker, True
    return "UNKNOWN", True


# ── Data helpers ───────────────────────────────────────
def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TickerDataset(Dataset):
    """Tokenize instruction-tuning samples with prompt masking."""

    def __init__(self, data: list[dict], tokenizer, prompt_template: str, max_length: int = 128):
        self.samples = []
        for item in data:
            prompt = prompt_template.replace("{input}", item["input"])
            full_text = prompt + item["output"] + tokenizer.eos_token

            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=True,
            )
            prompt_enc = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=True,
            )

            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
            prompt_len = len(prompt_enc["input_ids"])

            # Mask prompt tokens in labels (only compute loss on answer)
            labels = list(input_ids)
            labels[:prompt_len] = [-100] * prompt_len

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {k: list(v) for k, v in item.items()}


class PadCollator:
    """Dynamic padding to the longest sample in the batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ── Quick accuracy evaluator ──────────────────────────
def quick_accuracy(model, tokenizer, data: list[dict], prompt_template: str,
                   device, max_new_tokens: int = 8) -> dict:
    """Run greedy inference on a dataset and return accuracy stats."""
    model.eval()
    correct = 0
    total = len(data)
    errors = []

    for item in data:
        prompt = prompt_template.replace("{input}", item["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = outputs[0][prompt_len:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predicted, _ = normalize_ticker(raw_output)
        gold = item["output"]
        if predicted == gold:
            correct += 1
        else:
            errors.append({"input": item["input"], "gold": gold,
                           "predicted": predicted, "raw": raw_output})

    model.train()
    return {"accuracy": correct / total if total else 0, "correct": correct,
            "total": total, "errors": errors}


# ── Main ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Ticker Disambiguation")
    parser.add_argument("--config", type=str, default="configs/lora_v1.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a quick smoke test (50 train samples, 0.2 epochs)")
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Load data ──────────────────────────────────────
    train_data = load_jsonl(cfg["data"]["train"])
    val_data = load_jsonl(cfg["data"]["val"])
    dev_hard_path = cfg["data"].get("dev_hard")
    dev_hard_data = load_jsonl(dev_hard_path) if dev_hard_path else []
    prompt_template = Path(cfg["data"]["prompt_template"]).read_text().strip()

    print(f"Seed:     {args.seed}")
    print(f"Mode:     {'SMOKE' if args.smoke else 'FULL'}")
    print(f"Train:    {len(train_data)} samples")
    print(f"Val:      {len(val_data)} samples")
    print(f"Dev-hard: {len(dev_hard_data)} samples")
    print(f"Prompt:   {cfg['data']['prompt_template']}")

    # ── Load model & tokenizer ─────────────────────────
    model_name = cfg["model"]["name"]
    max_length = cfg["model"]["max_length"]

    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    dtype_str = cfg["model"].get("dtype", "fp32")
    if dtype_str == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading model: {model_name}")
    print(f"Device: {device} | Dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)  # type: ignore[arg-type]

    # ── Apply LoRA ─────────────────────────────────────
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Build datasets ─────────────────────────────────
    print("\nTokenizing datasets...")
    train_dataset = TickerDataset(train_data, tokenizer, prompt_template, max_length)
    val_dataset = TickerDataset(val_data, tokenizer, prompt_template, max_length)

    # Smoke: use only 50 train samples
    if args.smoke:
        n_smoke = min(50, len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(n_smoke)))
        print(f"SMOKE: using {n_smoke} train samples")

    print(f"Train dataset: {len(train_dataset)} samples tokenized")
    print(f"Val dataset:   {len(val_dataset)} samples tokenized")

    # ── Output dir with seed suffix ────────────────────
    t_cfg = cfg["training"]
    base_output_dir = cfg["output"]["dir"]
    output_dir = f"{base_output_dir}_seed{args.seed}"

    # Smoke run overrides
    epochs = 1 if args.smoke else t_cfg["epochs"]
    max_steps = 50 if args.smoke else -1

    training_args = TrainingArguments(
        output_dir=output_dir,
        seed=args.seed,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=t_cfg["learning_rate"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        warmup_ratio=t_cfg["warmup_ratio"],
        weight_decay=t_cfg["weight_decay"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        logging_steps=t_cfg["logging_steps"],
        eval_strategy=t_cfg["eval_strategy"],
        save_strategy=t_cfg["save_strategy"],
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        metric_for_best_model=t_cfg["metric_for_best_model"],
        greater_is_better=False,
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=3,
        dataloader_pin_memory=False,
    )

    # ── Train ──────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    print("\n" + "=" * 50)
    print(f"  Starting LoRA training (seed={args.seed})...")
    print("=" * 50 + "\n")

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # ── Post-training accuracy evaluation ──────────────
    print("\n" + "=" * 50)
    print("  Post-training accuracy evaluation")
    print("=" * 50 + "\n")

    val_result = quick_accuracy(model, tokenizer, val_data, prompt_template, device)
    print(f"  val_v2 accuracy:   {val_result['accuracy']:.2%} ({val_result['correct']}/{val_result['total']})")

    dev_hard_result = {"accuracy": 0, "correct": 0, "total": 0, "errors": []}
    if dev_hard_data:
        dev_hard_result = quick_accuracy(model, tokenizer, dev_hard_data, prompt_template, device)
        print(f"  dev_hard accuracy: {dev_hard_result['accuracy']:.2%} ({dev_hard_result['correct']}/{dev_hard_result['total']})")

    # Composite model selection score
    composite = 0.4 * val_result["accuracy"] + 0.6 * dev_hard_result["accuracy"]
    print(f"\n  Composite score (0.4*val + 0.6*dev_hard): {composite:.4f}")

    # ── Save adapter ───────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Save training log ──────────────────────────────
    train_log = {
        "seed": args.seed,
        "smoke": args.smoke,
        "model": model_name,
        "config": args.config,
        "train_samples": len(train_dataset),
        "val_samples": len(val_data),
        "dev_hard_samples": len(dev_hard_data),
        "epochs": epochs,
        "max_steps": max_steps,
        "train_time_seconds": round(train_time, 1),
        "val_accuracy": round(val_result["accuracy"], 4),
        "dev_hard_accuracy": round(dev_hard_result["accuracy"], 4),
        "composite_score": round(composite, 4),
        "val_errors": val_result["errors"],
        "dev_hard_errors": dev_hard_result["errors"],
        "adapter_path": output_dir,
    }

    log_dir = Path("results")
    log_dir.mkdir(exist_ok=True)
    mode_tag = "smoke" if args.smoke else "full"
    log_path = log_dir / f"lora_v1_seed{args.seed}_{mode_tag}_metrics.json"
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2, ensure_ascii=False)

    print(f"\n✓ LoRA adapter saved to {output_dir}")
    print(f"✓ Training log saved to {log_path}")
    print(f"  Training time: {train_time:.0f}s")
    print(f"\n  To run inference:")
    print(f"  python scripts/run_lora_infer.py \\")
    print(f"      --model {model_name} \\")
    print(f"      --adapter {output_dir} \\")
    print(f"      --gold data/test.jsonl \\")
    print(f"      --prompt prompts/zero_shot_plain.txt \\")
    print(f"      --output data/test_lora_v1_preds.jsonl")


if __name__ == "__main__":
    main()
