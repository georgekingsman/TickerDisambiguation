#!/bin/bash
# ──────────────────────────────────────────────────────────────
# run_experiment.sh – Complete LoRA v1 Experiment Pipeline
# LoRA v1 完整实验流水线
#
# Usage:
#   # Smoke run (quick sanity check, ~2 min)
#   bash scripts/run_experiment.sh smoke
#
#   # Full run with seed 42
#   bash scripts/run_experiment.sh full 42
#
#   # Full run with seed 7
#   bash scripts/run_experiment.sh full 7
#
#   # Evaluate only (skip training, use existing adapter)
#   bash scripts/run_experiment.sh eval 42
#
#   # Error delta analysis (after E3 predictions exist)
#   bash scripts/run_experiment.sh analysis
# ──────────────────────────────────────────────────────────────

set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE

MODE="${1:-smoke}"
SEED="${2:-42}"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
CONFIG="configs/lora_v1.yaml"
PROMPT="prompts/zero_shot_plain.txt"
ADAPTER="checkpoints/lora_v1_seed${SEED}"

echo "═══════════════════════════════════════════════════"
echo "  LoRA v1 Experiment Pipeline"
echo "  Mode: ${MODE} | Seed: ${SEED}"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Step 1: Training ──────────────────────────────────
if [[ "$MODE" == "smoke" ]]; then
    echo "▶ Step 1: Smoke run training (50 steps)..."
    python scripts/train_lora.py --config "$CONFIG" --seed "$SEED" --smoke
    echo ""
    echo "✓ Smoke run complete. Check results/ for metrics."
    echo "  If loss decreased and outputs look reasonable, proceed with:"
    echo "  bash scripts/run_experiment.sh full $SEED"
    exit 0

elif [[ "$MODE" == "full" ]]; then
    echo "▶ Step 1: Full LoRA v1 training (seed=$SEED)..."
    python scripts/train_lora.py --config "$CONFIG" --seed "$SEED"
    echo ""

elif [[ "$MODE" == "eval" ]]; then
    echo "▶ Skipping training, using existing adapter: $ADAPTER"
    if [[ ! -d "$ADAPTER" ]]; then
        echo "✗ Adapter not found: $ADAPTER"
        echo "  Run full training first: bash scripts/run_experiment.sh full $SEED"
        exit 1
    fi
    echo ""

elif [[ "$MODE" == "analysis" ]]; then
    echo "▶ Running error delta analysis..."
    python scripts/error_analysis.py
    exit 0

else
    echo "✗ Unknown mode: $MODE"
    echo "Usage: bash scripts/run_experiment.sh {smoke|full|eval|analysis} [seed]"
    exit 1
fi

# ── Step 2: LoRA Inference on all eval sets ───────────
echo "▶ Step 2: Running LoRA inference on all evaluation sets..."
echo ""

echo "  [1/5] val_v2..."
python scripts/run_lora_infer.py \
    --model "$MODEL" --adapter "$ADAPTER" \
    --gold data/val_v2.jsonl --prompt "$PROMPT" \
    --output "data/val_v2_lora_v1_seed${SEED}_preds.jsonl"

echo ""
echo "  [2/5] dev_hard..."
python scripts/run_lora_infer.py \
    --model "$MODEL" --adapter "$ADAPTER" \
    --gold data/dev_hard.jsonl --prompt "$PROMPT" \
    --output "data/dev_hard_lora_v1_seed${SEED}_preds.jsonl"

echo ""
echo "  [3/5] test..."
python scripts/run_lora_infer.py \
    --model "$MODEL" --adapter "$ADAPTER" \
    --gold data/test.jsonl --prompt "$PROMPT" \
    --output data/test_lora_v1_preds.jsonl

echo ""
echo "  [4/5] ambiguous_eval..."
python scripts/run_lora_infer.py \
    --model "$MODEL" --adapter "$ADAPTER" \
    --gold data/ambiguous_eval.jsonl --prompt "$PROMPT" \
    --output data/ambiguous_eval_lora_v1_preds.jsonl

echo ""
echo "  [5/5] hard_eval..."
python scripts/run_lora_infer.py \
    --model "$MODEL" --adapter "$ADAPTER" \
    --gold data/hard_eval.jsonl --prompt "$PROMPT" \
    --output data/hard_eval_lora_v1_preds.jsonl

# ── Step 3: Official Evaluation ───────────────────────
echo ""
echo "▶ Step 3: Official evaluation (E3 results)..."
echo ""

echo "═══ TEST SET ═══"
python scripts/evaluate.py \
    --predictions data/test_lora_v1_preds.jsonl \
    --gold data/test.jsonl \
    --json-output "results/e3_test_seed${SEED}.json"

echo ""
echo "═══ AMBIGUOUS EVAL ═══"
python scripts/evaluate.py \
    --predictions data/ambiguous_eval_lora_v1_preds.jsonl \
    --gold data/ambiguous_eval.jsonl \
    --json-output "results/e3_ambiguous_seed${SEED}.json"

echo ""
echo "═══ HARD EVAL ═══"
python scripts/evaluate.py \
    --predictions data/hard_eval_lora_v1_preds.jsonl \
    --gold data/hard_eval.jsonl \
    --json-output "results/e3_hard_seed${SEED}.json"

# ── Step 4: Error Delta Analysis ──────────────────────
echo ""
echo "▶ Step 4: Error delta analysis (E1 vs E2 vs E3)..."
python scripts/error_analysis.py

# ── Done ──────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ LoRA v1 Experiment Complete (seed=$SEED)"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  Outputs:"
echo "    Adapter:      $ADAPTER"
echo "    Predictions:  data/*_lora_v1_preds.jsonl"
echo "    Eval JSON:    results/e3_*_seed${SEED}.json"
echo "    Train log:    results/lora_v1_seed${SEED}_full_metrics.json"
echo "    Error delta:  results/error_delta_summary.json"
echo ""
echo "  Next steps:"
echo "    1. Review results/ and update comparison_matrix.csv"
echo "    2. Run second seed:  bash scripts/run_experiment.sh full 7"
echo "    3. Write docs/lora_v1_results.md"
