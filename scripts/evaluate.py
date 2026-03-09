"""
evaluate.py  –  Evaluation script for Ticker Disambiguation
               股票代码消歧评估脚本

Usage / 用法:
    python scripts/evaluate.py --predictions predictions.jsonl --gold data/test.jsonl
    python scripts/evaluate.py --predictions predictions.jsonl --gold data/ambiguous_eval.jsonl

Input format (predictions.jsonl) / 输入格式（预测文件）:
    Each line / 每行: {"input": "...", "predicted": "AAPL"}
Gold format (*.jsonl) / 标签格式:
    Each line / 每行: {"instruction": "...", "input": "...", "output": "GOOGL"}

Metrics reported / 报告指标:
    - Exact Match Accuracy | 精确匹配准确率
    - Macro-F1 (across all labels) | 宏平均 F1（跨所有标签）
    - Per-label precision / recall / F1 | 每个标签的准确率 / 召回率 / F1
    - Confusion pairs (predicted vs gold for errors) | 混淆对（错误的预测 vs 标签）
"""

import argparse
import json
import sys
from collections import Counter, defaultdict


def load_jsonl(path):
    """Load a JSONL file into a list of dicts. | 将 JSONL 文件加载为字典列表。"""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def exact_match_accuracy(gold_labels, pred_labels):
    """Calculate exact match accuracy. | 计算精确匹配准确率。"""
    assert len(gold_labels) == len(pred_labels)
    correct = sum(g == p for g, p in zip(gold_labels, pred_labels))
    return correct / len(gold_labels) if gold_labels else 0.0


def macro_f1(gold_labels, pred_labels):
    """Calculate macro-averaged F1 and per-label details. | 计算宏平均 F1 和每个标签的详情。"""
    labels = sorted(set(gold_labels) | set(pred_labels))
    f1_scores = []
    details = {}

    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold_labels, pred_labels))
        fp = sum(g != label and p == label for g, p in zip(gold_labels, pred_labels))
        fn = sum(g == label and p != label for g, p in zip(gold_labels, pred_labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_scores.append(f1)
        details[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(g == label for g in gold_labels)}

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0, details


def error_analysis(gold_labels, pred_labels, inputs):
    """Collect mismatched predictions for error analysis. | 收集不匹配的预测用于错误分析。"""
    errors = []
    for inp, g, p in zip(inputs, gold_labels, pred_labels):
        if g != p:
            errors.append({"input": inp, "gold": g, "predicted": p})
    return errors


def hallucination_and_verbosity(pred_data, ticker_set):
    """Compute hallucination rate and verbosity rate from raw predictions."""
    n = len(pred_data)
    if n == 0:
        return 0.0, 0.0
    hallucinations = 0
    verbose = 0
    for item in pred_data:
        pred = item.get("predicted", "").strip().upper()
        raw = item.get("raw_output", pred)
        if pred not in ticker_set and pred != "UNKNOWN":
            hallucinations += 1
        if len(raw.split()) > 1:
            verbose += 1
    return hallucinations / n, verbose / n


def confusion_pair_count(errors, pair):
    """Count how many times a specific (gold, predicted) pair appears."""
    a, b = pair
    return sum(1 for e in errors if (e["gold"] == a and e["predicted"] == b)
               or (e["gold"] == b and e["predicted"] == a))


def evaluate(gold_path, pred_path):
    """Run full evaluation and print report. | 运行完整评估并打印报告。"""
    gold_data = load_jsonl(gold_path)
    pred_data = load_jsonl(pred_path)

    # Build lookup: input text → predicted label | 构建查找表：输入文本 → 预测标签
    pred_map = {item["input"]: item["predicted"].strip().upper() for item in pred_data}
    raw_map = {item["input"]: item.get("raw_output", item["predicted"]) for item in pred_data}

    gold_labels = []
    pred_labels = []
    inputs = []
    missing = []
    matched_preds = []  # raw prediction records for hallucination/verbosity

    for item in gold_data:
        inp = item["input"]
        gold_label = item["output"]
        if inp in pred_map:
            gold_labels.append(gold_label)
            pred_labels.append(pred_map[inp])
            inputs.append(inp)
            matched_preds.append({"predicted": pred_map[inp], "raw_output": raw_map[inp]})
        else:
            missing.append(inp)

    if missing:
        print(f"\n⚠  {len(missing)} gold inputs not found in predictions:")
        for m in missing[:5]:
            print(f"   - {m}")
        if len(missing) > 5:
            print(f"   ... and {len(missing) - 5} more")

    if not gold_labels:
        print("No matching samples found. Check that prediction inputs match gold inputs exactly.")
        sys.exit(1)

    # ── Metrics | 计算指标 ─────────────────────────────
    acc = exact_match_accuracy(gold_labels, pred_labels)
    mf1, per_label = macro_f1(gold_labels, pred_labels)
    errors = error_analysis(gold_labels, pred_labels, inputs)
    hall_rate, verb_rate = hallucination_and_verbosity(matched_preds, TICKER_SET)

    # Key confusion pairs
    goog_confusion = confusion_pair_count(errors, ("GOOG", "GOOGL"))
    brk_confusion = confusion_pair_count(errors, ("BRK-A", "BRK-B"))

    # ── Report | 输出报告 ─────────────────────────────
    print("=" * 60)
    print("  TICKER DISAMBIGUATION EVALUATION REPORT")
    print("  股票代码消歧评估报告")
    print("=" * 60)
    print(f"  Gold file:       {gold_path}")
    print(f"  Prediction file: {pred_path}")
    print(f"  Matched samples: {len(gold_labels)}")
    print()
    print(f"  Exact Match Accuracy:  {acc:.4f}  ({sum(g == p for g, p in zip(gold_labels, pred_labels))}/{len(gold_labels)})")
    print(f"  Macro-F1:              {mf1:.4f}")
    print(f"  Hallucination Rate:    {hall_rate:.4f}  ({int(hall_rate * len(gold_labels))}/{len(gold_labels)})")
    print(f"  Verbosity Rate:        {verb_rate:.4f}  ({int(verb_rate * len(gold_labels))}/{len(gold_labels)})")
    print(f"  GOOG↔GOOGL confusion:  {goog_confusion}")
    print(f"  BRK-A↔BRK-B confusion: {brk_confusion}")
    print()

    # Per-label breakdown | 每个标签的详细指标
    print("  Per-Label Breakdown / 分标签明细:")
    print(f"  {'Label':<10} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print("  " + "-" * 40)
    for label in sorted(per_label.keys()):
        d = per_label[label]
        print(f"  {label:<10} {d['precision']:>6.3f} {d['recall']:>6.3f} {d['f1']:>6.3f} {d['support']:>8}")
    print()

    # Errors | 错误分析
    if errors:
        print(f"  Errors / 错误 ({len(errors)}):")
        for e in errors:
            print(f"    [{e['gold']} → {e['predicted']}]  \"{e['input']}\"")
    else:
        print("  No errors – perfect score! | 无错误——满分！")

    print("=" * 60)

    # Return results as dict for programmatic use
    # 返回结果字典，便于程序化调用
    return {
        "accuracy": acc,
        "macro_f1": mf1,
        "hallucination_rate": hall_rate,
        "verbosity_rate": verb_rate,
        "goog_googl_confusion": goog_confusion,
        "brk_ab_confusion": brk_confusion,
        "per_label": per_label,
        "errors": errors,
        "n_matched": len(gold_labels),
        "n_missing": len(missing),
    }


# ── Rule baseline | 规则基线 ───────────────────────────────────────

# Alias mapping: common names → ticker | 别名映射：常用名称 → 代码
ALIAS_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN",
    "meta": "META", "facebook": "META", "fb": "META",
    "tesla": "TSLA", "nvidia": "NVDA", "netflix": "NFLX",
    "jpmorgan": "JPM", "jp morgan": "JPM", "chase": "JPM",
    "bank of america": "BAC", "bofa": "BAC",
    "amd": "AMD", "advanced micro devices": "AMD",
    "intel": "INTC", "palantir": "PLTR",
    "disney": "DIS", "walt disney": "DIS",
    "tsmc": "TSM", "taiwan semiconductor": "TSM",
    "alibaba": "BABA", "pinduoduo": "PDD", "temu": "PDD",
}

# Valid ticker universe | 有效的股票代码范围
TICKER_SET = {
    "AAPL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "NFLX",
    "GOOGL", "GOOG", "BRK-A", "BRK-B", "JPM", "BAC",
    "AMD", "INTC", "PLTR", "DIS", "TSM", "BABA", "PDD",
}


def rule_baseline_predict(text: str) -> str:
    """Simple rule-based ticker resolution."""
    t = text.lower()

    # ── Ticker format normalization ────────────────────
    for sep in [".", "/", " "]:
        for cls in ["a", "b"]:
            if f"brk{sep}{cls}" in t:
                return f"BRK-{cls.upper()}"

    # ── Direct ticker mentions ─────────────────────────
    for ticker in TICKER_SET:
        if ticker.lower() in t.split() or ticker in text:
            return ticker

    # ── Google / Alphabet with class ───────────────────
    is_google = "google" in t or "alphabet" in t
    if is_google:
        if "class c" in t or "non-voting" in t:
            return "GOOG"
        return "GOOGL"  # default

    # ── Berkshire with class ───────────────────────────
    is_berk = "berkshire" in t or "buffett" in t
    if is_berk:
        if "class a" in t or "expensive" in t:
            return "BRK-A"
        return "BRK-B"  # default

    # ── Alias lookup ───────────────────────────────────
    for alias, ticker in sorted(ALIAS_MAP.items(), key=lambda x: -len(x[0])):
        if alias in t:
            return ticker

    return "UNKNOWN"


def run_rule_baseline(gold_path, output_path):
    """Run rule baseline on a gold file and write predictions.
       在标签文件上运行规则基线并写入预测结果。"""
    gold_data = load_jsonl(gold_path)
    predictions = []
    for item in gold_data:
        pred = rule_baseline_predict(item["input"])
        predictions.append({"input": item["input"], "predicted": pred})

    with open(output_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Rule baseline predictions written to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ticker disambiguation predictions.")
    parser.add_argument("--predictions", type=str, help="Path to predictions JSONL")
    parser.add_argument("--gold", type=str, help="Path to gold JSONL")
    parser.add_argument("--run-rule-baseline", action="store_true",
                        help="Run the rule baseline on --gold and evaluate it")
    parser.add_argument("--json-output", type=str, default=None,
                        help="Path to write evaluation results as JSON (optional)")
    args = parser.parse_args()

    results = None
    if args.run_rule_baseline:
        if not args.gold:
            print("Error: --gold is required with --run-rule-baseline")
            sys.exit(1)
        pred_path = args.gold.replace(".jsonl", "_rule_preds.jsonl")
        run_rule_baseline(args.gold, pred_path)
        results = evaluate(args.gold, pred_path)
    elif args.predictions and args.gold:
        results = evaluate(args.gold, args.predictions)
    else:
        print("Usage:")
        print("  python evaluate.py --predictions preds.jsonl --gold gold.jsonl")
        print("  python evaluate.py --run-rule-baseline --gold gold.jsonl")
        sys.exit(1)

    if results and args.json_output:
        # Serialize per_label (has nested dicts) but skip errors for brevity
        out = {k: v for k, v in results.items() if k != "per_label"}
        out["per_label"] = {k: v for k, v in results["per_label"].items()}
        from pathlib import Path
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Results JSON saved to {args.json_output}")
