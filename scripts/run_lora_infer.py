"""
run_lora_infer.py  –  LoRA inference for Ticker Disambiguation
                      股票代码消歧 LoRA 推理脚本

Usage / 用法:
    python scripts/run_lora_infer.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --adapter checkpoints/lora_v1 \
        --gold data/test.jsonl \
        --prompt prompts/zero_shot_plain.txt \
        --output data/test_lora_v1_preds.jsonl

Requirements / 依赖:
    pip install transformers torch peft
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Valid ticker universe | 有效代码范围 ──────────────────────────
TICKER_SET = {
    "AAPL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "NFLX",
    "GOOGL", "GOOG", "BRK-A", "BRK-B", "JPM", "BAC",
    "AMD", "INTC", "PLTR", "DIS", "TSM", "BABA", "PDD",
}

_TICKER_RE = re.compile(r"\b([A-Z]{1,5}(?:[.\-/][A-Z0-9]{1,3})?)\b")


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_ticker(raw: str) -> tuple[str, bool]:
    """Normalize raw model output to a canonical ticker symbol.
    Identical logic to run_zero_shot.py for fair comparison."""
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


def run_inference(
    model_name: str,
    adapter_path: str,
    gold_path: str,
    prompt_path: str,
    output_path: str,
    max_new_tokens: int = 8,
):
    """Run LoRA inference on a gold dataset."""

    gold_data = load_jsonl(gold_path)
    prompt_template = Path(prompt_path).read_text().strip()

    print(f"Model:    {model_name}")
    print(f"Adapter:  {adapter_path}")
    print(f"Gold:     {gold_path} ({len(gold_data)} samples)")
    print(f"Prompt:   {prompt_path}")
    print(f"Output:   {output_path}")
    print()

    # ── Load model ─────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()

    print(f"Device: {device} | Dtype: {dtype}")
    print()

    # ── Run inference ──────────────────────────────────
    predictions = []
    for i, item in enumerate(gold_data):
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
        predicted, was_normalized = normalize_ticker(raw_output)

        pred_record = {
            "input": item["input"],
            "predicted": predicted,
            "raw_output": raw_output,
            "normalized": was_normalized,
            "model": f"{model_name}+LoRA({adapter_path})",
        }
        predictions.append(pred_record)

        gold_label = item.get("output", "?")
        status = "✓" if predicted == gold_label else "✗"
        print(f"  [{i+1:3d}/{len(gold_data)}] {status}  {predicted:<8s}  (raw: {raw_output!r:<20s})  ← {item['input'][:60]}")

    # ── Write predictions ──────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ── Quick summary ──────────────────────────────────
    correct = sum(
        p["predicted"] == item.get("output", "")
        for p, item in zip(predictions, gold_data)
    )
    print()
    print(f"Quick accuracy: {correct}/{len(gold_data)} = {correct / len(gold_data):.2%}")
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LoRA ticker disambiguation inference."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace base model name (same as training)")
    parser.add_argument("--adapter", type=str, required=True,
                        help="Path to LoRA adapter checkpoint directory")
    parser.add_argument("--gold", type=str, required=True,
                        help="Path to gold JSONL file")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Path to prompt template file (.txt)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write predictions JSONL")
    parser.add_argument("--max-new-tokens", type=int, default=8,
                        help="Max new tokens to generate (default: 8)")
    args = parser.parse_args()

    run_inference(
        model_name=args.model,
        adapter_path=args.adapter,
        gold_path=args.gold,
        prompt_path=args.prompt,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
    )
