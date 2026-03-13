# Submission — Investment Research Copilot

> **Reviewer: if you have 3 minutes, follow this order:**
> 1. Read the [final numbers](#3-final-numbers) below (30 seconds)
> 2. Open [demo_assets/demo_case_1_output.md](demo_assets/demo_case_1_output.md) — Google share class case (1 min)
> 3. Open [demo_assets/demo_case_2_output.md](demo_assets/demo_case_2_output.md) — FB→META alias case (1 min)
> 4. Scan [results/comparison_matrix.csv](results/comparison_matrix.csv) for the full 7-system comparison (30 sec)
>
> No code execution required. All outputs are pre-captured.

---

## 1. What This Project Is

A **lightweight investment research copilot prototype** where ticker disambiguation is the entry-point module. The system converts ambiguous natural-language company references (*"Facebook"*, *"Alphabet class A"*, *"Berkshire B"*) into canonical US equity ticker symbols, then feeds the resolved symbol into a live-data research memo pipeline.

Ticker disambiguation is the critical first step: if routing fails, every downstream tool (data fetch, memo, alerts) produces results for the wrong stock. This project builds and benchmarks a hybrid resolver — LoRA fine-tuning + a normalize layer — against 7 system variants across 5 evaluation splits.

---

## 2. What Was Delivered

| Requirement | Deliverable | Location |
|---|---|---|
| LoRA fine-tuning (3 iterations) | v1 → v1.1 → v1.2, each with documented patches | [scripts/train_lora.py](scripts/train_lora.py) · [configs/](configs/) |
| Before/after evaluation | Full comparison matrix: 7 systems × 5 eval splits | [results/comparison_matrix.csv](results/comparison_matrix.csv) |
| Ambiguity handling | Share-class data patch + alias normalize layer | [data/train_v2p2.jsonl](data/train_v2p2.jsonl) |
| End-to-end copilot workflow | Query → Resolver → Market Data → Research Memo | [app.py](app.py) |
| Final frozen predictions | 3 eval splits, locked at submission | [data/test_final_preds.jsonl](data/test_final_preds.jsonl) · [data/ambiguous_final_preds.jsonl](data/ambiguous_final_preds.jsonl) · [data/hard_final_preds.jsonl](data/hard_final_preds.jsonl) |
| Pre-captured demo outputs | 3 annotated cases with full memos + citation markers | [demo_assets/](demo_assets/) |
| One-liner demo runner | Regenerates demo_assets in one command | [run_demo.sh](run_demo.sh) |
| **MCP server** | Yahoo Finance endpoints wrapping data_fetch.py | [src/mcp_server.py](src/mcp_server.py) |
| **n8n workflow** | Importable 7-node topology: Webhook → Resolve → MCP × 2 → Memo | [n8n_workflow.json](n8n_workflow.json) |

---

## 2b. Workflow Integration — Simplifications

The RA spec requires n8n orchestration, a Yahoo Finance MCP server, budget limits, response caching, and citation markers.

| Requirement | Status | Location |
|---|---|---|
| Budget limit | ✅ Implemented | `budget_limit` param in `src/data_fetch.py` |
| Response caching | ✅ Implemented | `_CACHE` dict in `src/data_fetch.py` |
| Citation markers `[1]`/`[2]` | ✅ Implemented | All memo outputs (see `demo_assets/`) |
| MCP server endpoints | ✅ Reference impl | `src/mcp_server.py` |
| n8n workflow JSON | ✅ Exportable | `n8n_workflow.json` |
| **Live n8n orchestration** | ⚠️ Simplified | Demo runs via direct `app.py`; see note below |

> **Simplification note (per RA clause):** Primary development time was invested in LoRA training quality — zero-hallucination, zero-verbosity, and 2× hard-eval accuracy. Accordingly, end-to-end demonstration runs through the native Python pipeline (`app.py`) rather than a live n8n instance. The complete n8n + MCP topology is fully specified in `n8n_workflow.json` and `src/mcp_server.py` and can be deployed directly into any n8n instance.

---

## 3. Final Numbers

**Final system: LoRA v1.2 + normalize(FB→META, BRK.B→BRK-B)**

| Eval Split | Rule Baseline | **LoRA v1.2 + normalize** | Delta |
|---|---:|---:|---:|
| Test (29 samples) | 96.55% | **93.10%** | −3.45pp |
| Ambiguous (20 samples) | 95.00% | **95.00%** | 0pp |
| **Hard / colloquial (30 samples)** | **30.00%** | **60.00%** | **+30pp** |
| Google ShareClass (24 samples) | — | **100.00%** | — |
| Meta Alias (12 samples) | — | 58.33% | — |
| Hallucination | 0% | **0%** | — |
| Verbosity | 0% | **0%** | — |

**Hard eval is the decisive metric.** Real users say "Facebook" not "META", "Alphabet class A" not "GOOGL". The rule baseline collapses to 30% on these colloquial queries; LoRA v1.2 doubles it to 60%.

---

## 4. Final Locked Configuration

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Final adapter | `checkpoints/lora_v1p2_seed42/` |
| Training set | `data/train_v2p2.jsonl` (520 samples) |
| Training config | `configs/lora_v1p2.yaml` |
| Normalize rules | `FB → META`, `BRK.B → BRK-B` |
| Decoding | Greedy (deterministic) |
| Final eval seed | 42 |

---

## 5. How to Review in 3 Minutes (No Code Required)

**Option A — read the pre-captured outputs:**

| File | What it shows |
|---|---|
| [demo_assets/demo_case_1_output.md](demo_assets/demo_case_1_output.md) | GOOGL vs GOOG disambiguation + market memo |
| [demo_assets/demo_case_2_output.md](demo_assets/demo_case_2_output.md) | FB alias → META normalization + market memo |
| [demo_assets/demo_case_3_output.md](demo_assets/demo_case_3_output.md) | Apple standard case — baseline stability |
| [results/comparison_matrix.csv](results/comparison_matrix.csv) | Full 7-system × 5-split benchmark table |

**Option B — run the demo yourself:**

```bash
pip install -r requirements.txt
bash run_demo.sh
```

Outputs are regenerated to `demo_assets/` and also printed to the terminal.

**Option C — single query:**

```bash
python app.py --query "Research Alphabet class A for me"
python app.py --query "Research FB for the past 3 months"
```

---

## 6. Key Files Reference

| File | Purpose |
|---|---|
| `app.py` | Main entry point — `--demo`, `--batch`, `--query`, interactive |
| `run_demo.sh` | One-liner runner: runs all 3 cases + saves to `demo_assets/` |
| `requirements.txt` | Python dependencies |
| `src/resolver.py` | LoRA v1.2 inference + normalize_ticker layer |
| `src/data_fetch.py` | yfinance market data fetch with stub fallback |
| `src/memo.py` | Research memo generation + flywheel JSONL logging |
| `src/config.py` | TICKER_SET, ALIASES, model/adapter paths |
| `scripts/train_lora.py` | LoRA fine-tuning script |
| `scripts/evaluate.py` | Evaluation harness + rule baseline |
| `configs/lora_v1p2.yaml` | **[FINAL]** training config |
| `checkpoints/lora_v1p2_seed42/` | **[FINAL]** trained LoRA adapter weights |
| `data/train_v2p2.jsonl` | **[FINAL]** 520-sample training set |
| `data/test_final_preds.jsonl` | **[FINAL]** frozen predictions — test split |
| `data/ambiguous_final_preds.jsonl` | **[FINAL]** frozen predictions — ambiguous split |
| `data/hard_final_preds.jsonl` | **[FINAL]** frozen predictions — hard split |
| `results/comparison_matrix.csv` | Full 7-system benchmark comparison |
| `demo_assets/` | Pre-captured demo outputs (readable without running code) |
| `docs/experiment_protocol.md` | Frozen experiment protocol |
| `docs/label_policy.md` | Labeling rules and ticker defaults |
