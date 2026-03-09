"""
run_zero_shot.py  –  Zero-shot inference for Ticker Disambiguation
                     股票代码消歧 Zero-shot 推理脚本

Usage / 用法:
    python scripts/run_zero_shot.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --gold data/test.jsonl \
        --prompt prompts/zero_shot_plain.txt \
        --output data/test_zeroshot_plain_preds.jsonl

    python scripts/run_zero_shot.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --gold data/ambiguous_eval.jsonl \
        --prompt prompts/zero_shot_prompt.txt \
        --output data/ambiguous_eval_zeroshot_policy_preds.jsonl

Requirements / 依赖:
    pip install transformers torch
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Valid ticker universe | 有效代码范围 ──────────────────────────
TICKER_SET = {
    "AAPL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "NFLX",
    "GOOGL", "GOOG", "BRK-A", "BRK-B", "JPM", "BAC",
    "AMD", "INTC", "PLTR", "DIS", "TSM", "BABA", "PDD",
}

# Regex pattern to extract a ticker-like token from model output
# Matches uppercase letters, digits, dots, slashes, dashes (e.g. BRK-B, BRK.B)
_TICKER_RE = re.compile(r"\b([A-Z]{1,5}(?:[.\-/][A-Z0-9]{1,3})?)\b")

# Historical ticker aliases → current canonical symbol
# 历史代码别名 → 当前规范代码
TICKER_ALIASES = {
    "FB": "META",
}


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_prompt_template(path: str) -> str:
    """Load prompt template from a text file."""
    return Path(path).read_text().strip()


def normalize_ticker(raw: str) -> tuple[str, bool]:
    """
    Normalize raw model output to a canonical ticker symbol.
    Returns (normalized_ticker, was_normalized).

    Rules:
    - Strip whitespace, convert to uppercase
    - BRK.B / BRK/B → BRK-B  (and same for A)
    - If output contains extra text, extract the first valid ticker
    - If extraction fails, return UNKNOWN
    """
    text = raw.strip().upper()

    # If the entire output is a clean ticker, use it directly
    clean = text.replace(".", "-").replace("/", "-")
    # Apply historical ticker aliases (e.g. FB → META)
    if clean in TICKER_ALIASES:
        return TICKER_ALIASES[clean], True
    if clean in TICKER_SET:
        return clean, clean != text

    # Try to find a valid ticker anywhere in the output
    candidates = _TICKER_RE.findall(text)
    for candidate in candidates:
        normalized = candidate.replace(".", "-").replace("/", "-")
        if normalized in TICKER_ALIASES:
            return TICKER_ALIASES[normalized], True
        if normalized in TICKER_SET:
            return normalized, True

    # Last resort: check if any known ticker appears as a substring
    for ticker in TICKER_SET:
        if ticker in text:
            return ticker, True

    return "UNKNOWN", True


def run_inference(
    model_name: str,
    gold_path: str,
    prompt_path: str,
    output_path: str,
    max_new_tokens: int = 8,
    batch_size: int = 1,
):
    """Run zero-shot inference on a gold dataset."""

    # ── Load data & prompt ─────────────────────────────
    gold_data = load_jsonl(gold_path)
    prompt_template = load_prompt_template(prompt_path)

    print(f"Model:    {model_name}")
    print(f"Gold:     {gold_path} ({len(gold_data)} samples)")
    print(f"Prompt:   {prompt_path}")
    print(f"Output:   {output_path}")
    print()

    # ── Load model ─────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model = model.to(device)  # type: ignore[arg-type]
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
            "model": model_name,
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
    print(f"Quick accuracy: {correct}/{len(gold_data)} = {correct/len(gold_data):.2%}")
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run zero-shot ticker disambiguation inference."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name or local path")
    parser.add_argument("--gold", type=str, required=True,
                        help="Path to gold JSONL file (test.jsonl, ambiguous_eval.jsonl, etc.)")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Path to prompt template file (.txt)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write predictions JSONL")
    parser.add_argument("--max-new-tokens", type=int, default=8,
                        help="Max new tokens to generate (default: 8)")
    args = parser.parse_args()

    run_inference(
        model_name=args.model,
        gold_path=args.gold,
        prompt_path=args.prompt,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
    )
