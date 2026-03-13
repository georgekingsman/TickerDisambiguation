#!/usr/bin/env bash
# run_demo.sh — One-liner demo runner for the Research Copilot
#
# Usage:
#   bash run_demo.sh
#
# What it does:
#   1. Checks Python is available
#   2. Runs 3 annotated demo cases through the full pipeline
#   3. Saves all inputs + formatted outputs to demo_assets/
#   4. Prints a summary of which files to open next
#
# Demo cases:
#   1. "Research Alphabet class A" → resolves to GOOGL (Google share class fix)
#   2. "Research FB for 3 months"  → resolves FB → META (alias normalization)
#   3. "Analyze Apple over 1 year" → resolves to AAPL (standard case)

set -euo pipefail

# Allow duplicate OpenMP libs on macOS (common with PyTorch + conda)
export KMP_DUPLICATE_LIB_OK=TRUE

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Preflight check ───────────────────────────────────────────────
if ! command -v python &>/dev/null; then
    echo "ERROR: python not found. Activate your virtualenv first:"
    echo "  source .venv/bin/activate"
    exit 1
fi

echo ""
echo "=================================================="
echo "  Investment Research Copilot — Demo Runner"
echo "  LoRA v1.2 + normalize(FB→META, BRK.B→BRK-B)"
echo "=================================================="
echo ""
echo "  Running 3 annotated demo cases..."
echo "  Outputs will be saved to demo_assets/"
echo ""

# ── Run pipeline + save demo_assets ──────────────────────────────
python app.py --batch

# ── Post-run guidance ─────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Demo complete. Files saved to demo_assets/"
echo "=================================================="
echo ""
echo "  Best files to review:"
echo ""
echo "  1. demo_assets/demo_case_1_output.md"
echo "     → Google Alphabet class A  (GOOGL vs GOOG disambiguation)"
echo ""
echo "  2. demo_assets/demo_case_2_output.md"
echo "     → FB alias  (normalize: FB → META)"
echo ""
echo "  3. demo_assets/demo_case_3_output.md"
echo "     → Apple standard case  (baseline stability)"
echo ""
echo "  For the full evaluation numbers, see:"
echo "  results/comparison_matrix.csv"
echo ""
echo "  For a no-run overview, see:"
echo "  SUBMISSION.md"
echo "=================================================="
echo ""
