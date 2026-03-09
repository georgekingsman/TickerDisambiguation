#!/bin/bash
# ──────────────────────────────────────────────────────────────
# run_v1p1_experiment.sh – LoRA v1.1 Targeted Patch Experiment
# LoRA v1.1 定向修复实验流水线
#
# Usage:
#   bash scripts/run_v1p1_experiment.sh full 42
#   bash scripts/run_v1p1_experiment.sh eval 42
# ──────────────────────────────────────────────────────────────

set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE

MODE="${1:-full}"
SEED="${2:-42}"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
CONFIG="configs/lora_v1p1.yaml"
PROMPT="prompts/zero_shot_plain.txt"
ADAPTER="checkpoints/lora_v1p1_seed${SEED}"
TAG="lora_v1p1_seed${SEED}"

echo "═══════════════════════════════════════════════════"
echo "  LoRA v1.1 Experiment Pipeline (Targeted Patch)"
echo "  Mode: ${MODE} | Seed: ${SEED}"
echo "  Train: train_v2p1.jsonl (392 + 96 patch)"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Step 1: Training ──────────────────────────────────
if [[ "$MODE" == "full" ]]; then
    echo "▶ Step 1: Full LoRA v1.1 training (seed=$SEED)..."
    python scripts/train_lora.py --config "$CONFIG" --seed "$SEED"
    echo ""

elif [[ "$MODE" == "eval" ]]; then
    echo "▶ Skipping training, using existing adapter: $ADAPTER"
    if [[ ! -d "$ADAPTER" ]]; then
        echo "✗ Adapter not found: $ADAPTER"
        exit 1
    fi
    echo ""
else
    echo "✗ Unknown mode: $MODE"
    echo "Usage: bash scripts/run_v1p1_experiment.sh {full|eval} [seed]"
    exit 1
fi

# ── Step 2: Inference on all eval sets ────────────────
echo "▶ Step 2: Running inference on all evaluation sets..."
echo ""

EVAL_SETS=(
    "val_v2:data/val_v2.jsonl"
    "dev_hard:data/dev_hard.jsonl"
    "test:data/test.jsonl"
    "ambiguous_eval:data/ambiguous_eval.jsonl"
    "hard_eval:data/hard_eval.jsonl"
    "google_shareclass_eval:data/google_shareclass_eval.jsonl"
    "meta_alias_eval:data/meta_alias_eval.jsonl"
)

TOTAL=${#EVAL_SETS[@]}
IDX=0

for entry in "${EVAL_SETS[@]}"; do
    NAME="${entry%%:*}"
    GOLD="${entry##*:}"
    IDX=$((IDX + 1))
    OUTPUT="data/${NAME}_${TAG}_preds.jsonl"

    echo "  [${IDX}/${TOTAL}] ${NAME}..."
    python scripts/run_lora_infer.py \
        --model "$MODEL" --adapter "$ADAPTER" \
        --gold "$GOLD" --prompt "$PROMPT" \
        --output "$OUTPUT"
    echo ""
done

# ── Step 3: Official Evaluation ───────────────────────
echo "▶ Step 3: Official evaluation..."
echo ""

echo "═══ TEST SET ═══"
python scripts/evaluate.py \
    --predictions "data/test_${TAG}_preds.jsonl" \
    --gold data/test.jsonl \
    --json-output "results/v1p1_test_seed${SEED}.json"

echo ""
echo "═══ AMBIGUOUS EVAL ═══"
python scripts/evaluate.py \
    --predictions "data/ambiguous_eval_${TAG}_preds.jsonl" \
    --gold data/ambiguous_eval.jsonl \
    --json-output "results/v1p1_ambiguous_seed${SEED}.json"

echo ""
echo "═══ HARD EVAL ═══"
python scripts/evaluate.py \
    --predictions "data/hard_eval_${TAG}_preds.jsonl" \
    --gold data/hard_eval.jsonl \
    --json-output "results/v1p1_hard_seed${SEED}.json"

# ── Step 4: Diagnostic Evaluation ─────────────────────
echo ""
echo "▶ Step 4: Diagnostic evaluation (targeted slices)..."
echo ""

echo "═══ GOOGLE SHARE-CLASS EVAL ═══"
python scripts/evaluate.py \
    --predictions "data/google_shareclass_eval_${TAG}_preds.jsonl" \
    --gold data/google_shareclass_eval.jsonl \
    --json-output "results/v1p1_google_shareclass_seed${SEED}.json"

echo ""
echo "═══ META ALIAS EVAL ═══"
python scripts/evaluate.py \
    --predictions "data/meta_alias_eval_${TAG}_preds.jsonl" \
    --gold data/meta_alias_eval.jsonl \
    --json-output "results/v1p1_meta_alias_seed${SEED}.json"

# ── Done ──────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ LoRA v1.1 Experiment Complete (seed=$SEED)"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  Outputs:"
echo "    Adapter:       $ADAPTER"
echo "    Predictions:   data/*_${TAG}_preds.jsonl"
echo "    Main eval:     results/v1p1_{test,ambiguous,hard}_seed${SEED}.json"
echo "    Diagnostics:   results/v1p1_{google_shareclass,meta_alias}_seed${SEED}.json"
echo "    Train log:     results/lora_v1p1_seed${SEED}_full_metrics.json"
