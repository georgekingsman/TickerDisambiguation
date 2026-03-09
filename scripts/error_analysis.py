"""
error_analysis.py  –  Error Delta Analysis across E1/E2/E3
                      跨实验错误差值分析

Usage / 用法:
    python scripts/error_analysis.py

Compares prediction files from:
    E1: Base + plain prompt
    E2: Base + policy-aware prompt
    E3: LoRA v1 + plain prompt

Outputs a structured error delta report showing which errors
were fixed / introduced / persisted by LoRA training.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ── Config ─────────────────────────────────────────────
TICKER_SET = {
    "AAPL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "NFLX",
    "GOOGL", "GOOG", "BRK-A", "BRK-B", "JPM", "BAC",
    "AMD", "INTC", "PLTR", "DIS", "TSM", "BABA", "PDD",
}

EVAL_SETS = [
    ("test",      "data/test.jsonl"),
    ("ambiguous", "data/ambiguous_eval.jsonl"),
    ("hard",      "data/hard_eval.jsonl"),
]

EXPERIMENTS = {
    "E1_plain": {
        "test":      "data/test_zeroshot_plain_preds.jsonl",
        "ambiguous": "data/ambiguous_eval_zeroshot_plain_preds.jsonl",
        "hard":      "data/hard_eval_zeroshot_plain_preds.jsonl",
    },
    "E2_policy": {
        "test":      "data/test_zeroshot_policy_preds.jsonl",
        "ambiguous": "data/ambiguous_eval_zeroshot_policy_preds.jsonl",
        "hard":      "data/hard_eval_zeroshot_policy_preds.jsonl",
    },
    "E3_lora": {
        "test":      "data/test_lora_v1_preds.jsonl",
        "ambiguous": "data/ambiguous_eval_lora_v1_preds.jsonl",
        "hard":      "data/hard_eval_lora_v1_preds.jsonl",
    },
}


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def classify_error(gold: str, predicted: str, raw_output: str = "") -> str:
    """Classify an error into one of 5 categories."""
    # 1. Share-class confusion
    if {gold, predicted} <= {"GOOG", "GOOGL"} or {gold, predicted} <= {"BRK-A", "BRK-B"}:
        return "share_class_confusion"
    # 3. Hallucination (predicted not in universe)
    if predicted not in TICKER_SET and predicted != "UNKNOWN":
        return "hallucination"
    # 4. Verbose output (raw has multiple tokens)
    if raw_output and len(raw_output.split()) > 1:
        return "verbose_output"
    # 5. Format error (UNKNOWN usually means extraction failed)
    if predicted == "UNKNOWN":
        return "format_error"
    # 2. Alias / general failure
    return "alias_failure"


def get_errors(gold_path: str, pred_path: str) -> dict:
    """Return {input_text: {gold, predicted, raw_output, error_type}}."""
    gold_data = load_jsonl(gold_path)
    pred_data = load_jsonl(pred_path)

    pred_map = {}
    raw_map = {}
    for item in pred_data:
        pred_map[item["input"]] = item["predicted"].strip().upper()
        raw_map[item["input"]] = item.get("raw_output", item["predicted"])

    errors = {}
    for item in gold_data:
        inp = item["input"]
        gold = item["output"]
        pred = pred_map.get(inp, "MISSING")
        raw = raw_map.get(inp, "")
        if pred != gold:
            errors[inp] = {
                "gold": gold,
                "predicted": pred,
                "raw_output": raw,
                "error_type": classify_error(gold, pred, raw),
            }
    return errors


def main():
    # Check that E3 files exist
    for eval_name, _ in EVAL_SETS:
        path = EXPERIMENTS["E3_lora"][eval_name]
        if not Path(path).exists():
            print(f"⚠  E3 file not found: {path}")
            print("   Run LoRA inference first, then re-run this script.")
            sys.exit(1)

    print("=" * 70)
    print("  ERROR DELTA ANALYSIS – E1 vs E2 vs E3")
    print("  错误差值分析 – 基础模型 vs 策略提示 vs LoRA")
    print("=" * 70)

    all_summary = {}

    for eval_name, gold_path in EVAL_SETS:
        print(f"\n{'─' * 70}")
        print(f"  Dataset: {eval_name} ({gold_path})")
        print(f"{'─' * 70}")

        errors_by_exp = {}
        for exp_name, paths in EXPERIMENTS.items():
            pred_path = paths[eval_name]
            if Path(pred_path).exists():
                errors_by_exp[exp_name] = get_errors(gold_path, pred_path)
            else:
                print(f"  ⚠ Skipping {exp_name}: {pred_path} not found")

        if "E3_lora" not in errors_by_exp:
            continue

        e1_errors = errors_by_exp.get("E1_plain", {})
        e2_errors = errors_by_exp.get("E2_policy", {})
        e3_errors = errors_by_exp.get("E3_lora", {})

        # Error counts by type
        print(f"\n  Error counts by type:")
        print(f"  {'Error Type':<25} {'E1 Plain':>10} {'E2 Policy':>10} {'E3 LoRA':>10}")
        print(f"  {'─' * 55}")

        all_types = sorted(set(
            [e["error_type"] for e in e1_errors.values()] +
            [e["error_type"] for e in e2_errors.values()] +
            [e["error_type"] for e in e3_errors.values()]
        ))

        type_counts = {}
        for etype in all_types:
            c1 = sum(1 for e in e1_errors.values() if e["error_type"] == etype)
            c2 = sum(1 for e in e2_errors.values() if e["error_type"] == etype)
            c3 = sum(1 for e in e3_errors.values() if e["error_type"] == etype)
            print(f"  {etype:<25} {c1:>10} {c2:>10} {c3:>10}")
            type_counts[etype] = {"E1": c1, "E2": c2, "E3": c3}

        print(f"  {'─' * 55}")
        print(f"  {'TOTAL':<25} {len(e1_errors):>10} {len(e2_errors):>10} {len(e3_errors):>10}")

        # Delta: errors fixed by LoRA (were in E1 but not E3)
        fixed_vs_e1 = set(e1_errors.keys()) - set(e3_errors.keys())
        new_vs_e1 = set(e3_errors.keys()) - set(e1_errors.keys())
        fixed_vs_e2 = set(e2_errors.keys()) - set(e3_errors.keys())
        new_vs_e2 = set(e3_errors.keys()) - set(e2_errors.keys())

        print(f"\n  Delta vs E1 (plain baseline):")
        print(f"    Fixed by LoRA: {len(fixed_vs_e1)}")
        print(f"    New in LoRA:   {len(new_vs_e1)}")
        print(f"    Net change:    {len(e1_errors) - len(e3_errors):+d}")

        print(f"\n  Delta vs E2 (policy-aware):")
        print(f"    Fixed by LoRA: {len(fixed_vs_e2)}")
        print(f"    New in LoRA:   {len(new_vs_e2)}")
        print(f"    Net change:    {len(e2_errors) - len(e3_errors):+d}")

        # Show specific fixes
        if fixed_vs_e1:
            print(f"\n  Errors FIXED by LoRA (vs E1):")
            for inp in sorted(fixed_vs_e1):
                e = e1_errors[inp]
                print(f"    ✓ [{e['gold']}←{e['predicted']}] \"{inp[:70]}\"")

        if new_vs_e1:
            print(f"\n  Errors INTRODUCED by LoRA (vs E1):")
            for inp in sorted(new_vs_e1):
                e = e3_errors[inp]
                print(f"    ✗ [{e['gold']}←{e['predicted']}] \"{inp[:70]}\"")

        all_summary[eval_name] = {
            "e1_errors": len(e1_errors),
            "e2_errors": len(e2_errors),
            "e3_errors": len(e3_errors),
            "fixed_vs_e1": len(fixed_vs_e1),
            "new_vs_e1": len(new_vs_e1),
            "fixed_vs_e2": len(fixed_vs_e2),
            "new_vs_e2": len(new_vs_e2),
            "type_counts": type_counts,
        }

    # Final summary table
    print(f"\n{'=' * 70}")
    print("  SUMMARY TABLE (for docs/lora_v1_results.md)")
    print(f"{'=' * 70}")
    print()
    print("| Error Type | Base Plain | Base Policy-aware | LoRA v1 Plain |")
    print("|---|---:|---:|---:|")

    # Aggregate across all eval sets
    agg = defaultdict(lambda: {"E1": 0, "E2": 0, "E3": 0})
    for eval_name, summary in all_summary.items():
        for etype, counts in summary["type_counts"].items():
            agg[etype]["E1"] += counts["E1"]
            agg[etype]["E2"] += counts["E2"]
            agg[etype]["E3"] += counts["E3"]

    for etype in sorted(agg.keys()):
        c = agg[etype]
        print(f"| {etype} | {c['E1']} | {c['E2']} | {c['E3']} |")

    total_e1 = sum(c["E1"] for c in agg.values())
    total_e2 = sum(c["E2"] for c in agg.values())
    total_e3 = sum(c["E3"] for c in agg.values())
    print(f"| **TOTAL** | **{total_e1}** | **{total_e2}** | **{total_e3}** |")

    # Save summary JSON
    summary_path = Path("results/error_delta_summary.json")
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
