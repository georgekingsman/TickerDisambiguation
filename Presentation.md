---
marp: true
theme: default
class: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    background: #ffffff;
    color: #2c3e50;
  }
  h1 {
    color: #1a252f;
    font-size: 2.2em;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
  }
  h2 {
    color: #2c3e50;
    font-size: 1.55em;
    margin-bottom: 0.3em;
  }
  h3 {
    color: #3498db;
    font-size: 1.1em;
    margin: 0.3em 0;
  }
  table {
    width: 100%;
    font-size: 0.85em;
    border-collapse: collapse;
  }
  th {
    background-color: #2c3e50;
    color: #ffffff;
    padding: 6px 10px;
  }
  td {
    padding: 5px 10px;
    border-bottom: 1px solid #e8ecef;
  }
  tr:nth-child(even) td { background-color: #f8f9fa; }
  blockquote {
    border-left: 5px solid #3498db;
    background: #eaf4fb;
    padding: 10px 20px;
    margin: 10px 0;
    font-size: 0.9em;
  }
  code {
    background: #f0f3f4;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.88em;
    color: #c0392b;
  }
  .highlight { color: #27ae60; font-weight: bold; }
  .warn { color: #e74c3c; font-weight: bold; }
---

<!-- _class: lead -->

# Investment Research Copilot
## Ticker Disambiguation Module

**Lightweight LLM Fine-Tuning · Hybrid Resolver · End-to-End n8n Workflow**

---
**Presenter:** George · **Date:** March 2026

---

## Agenda

1. **Problem** — Why ticker disambiguation is critical
2. **Solution Architecture** — Three-layer hybrid resolver
3. **Data Engineering** — Curated training data & patches
4. **Fine-Tuning Strategy** — LoRA on Qwen2.5-0.5B
5. **Model Iteration** — v1 → v1.1 → v1.2 (ablation)
6. **Full Benchmark Results** — 7 systems × 5 evaluation splits
7. **Error Analysis** — Remaining hard cases
8. **End-to-End Workflow** — n8n + MCP + Data Flywheel
9. **Live Demo Showcase** — 3 annotated cases
10. **Deliverables & Next Steps**

---

## 1. The Problem: "Silent Failure" in Financial NLP

**Investment research tools fail silently on ambiguous inputs.**

| User Says | Intended Stock | Naïve System Returns | Consequence |
|:---|:---|:---|:---|
| *"Research Google class C"* | `GOOG` | `GOOGL` ❌ | Wrong share class, pricing divergence |
| *"Research Facebook earnings"* | `META` | API error ❌ | `FB` no longer trades |
| *"How is Berkshire B doing?"* | `BRK-B` | `BRK-A` ❌ | Price off by ~1500× ($400 vs $600k) |
| *"Analyze Alphabet's voting shares"* | `GOOGL` | hallucination ❌ | Completely wrong ticker |

> **The problem is not just incorrect — it is invisible.** The pipeline continues with bad data, and the user never knows.

**Target:** Convert any natural-language company reference → single canonical US equity ticker, with zero hallucination and zero verbosity.

---

## 2. Solution Architecture: Three-Layer Hybrid Resolver

```
User Query (natural language)
        │
        ▼
┌───────────────────────────────────┐
│  Layer 1: LoRA Fine-Tuned Model   │  "voting shares of Alphabet" → GOOGL
│  Qwen2.5-0.5B + LoRA adapter      │  Handles semantics, aliases, context
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  Layer 2: Normalize / Canonicalize│  FB → META   BRK.B → BRK-B
│  TICKER_ALIASES dict + regex      │  Handles format variants & stale tickers
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  Layer 3: Guardrail / Validation  │  Out-of-universe? → UNKNOWN
│  20-symbol universe check         │  Hallucination blocker
└───────────────────────────────────┘
        │
        ▼
  Canonical Ticker (e.g., GOOGL)
        │
        ▼
  Yahoo Finance MCP → Research Memo
```

---

## 3. Data Engineering: Building the Training Corpus

**Symbol universe: 20 US equities covering 4 key ambiguity groups**

```
AAPL  MSFT  AMZN  META  TSLA  NVDA  NFLX  GOOGL  GOOG
BRK-A  BRK-B  JPM  BAC  AMD  INTC  PLTR  DIS  TSM  BABA  PDD
```

**Training data evolution:**

| Version | Samples | Key Change |
|:---|:---:|:---|
| `train_v1` (baseline) | ~350 | Initial instruction-tuning dataset |
| `train_v2` | ~450 | Harder phrasings, colloquial aliases added |
| `train_v2p1` | ~490 | Expanded Google share-class examples |
| **`train_v2p2` (final)** | **520** | +Patch: `FB→META` alias · +Patch: `GOOG`/`GOOGL` bias fix |

**Label policy highlights:**

| Rule | Example |
|:---|:---|
| Default "Google/Alphabet" → `GOOGL` | Class A is the more-referenced share |
| Default "Berkshire" → `BRK-B` | Retail-accessible (~$400 vs ~$600k) |
| Explicit override wins | "Google class C" → `GOOG` |
| Legacy alias maps | `FB` → `META` (canonical, not historical) |

---

## 4. Fine-Tuning Strategy: High Performance, Near-Zero Cost

**PEFT / LoRA on Qwen2.5-0.5B-Instruct**

| Parameter | Value |
|:---|:---|
| **Base Model** | `Qwen/Qwen2.5-0.5B-Instruct` |
| **Method** | LoRA (Low-Rank Adaptation) — ~1% trainable parameters |
| **LoRA Rank / Alpha** | r=16, α=32 |
| **Target Modules** | `q_proj`, `v_proj` |
| **Training Samples** | 520 (final: `train_v2p2`) |
| **Epochs** | 3 (early versions: 1–2) |
| **Train Time** | ~128 seconds on single GPU |
| **Decode** | Temperature=0 (greedy), max_new_tokens=8 |
| **Output Format** | Single ticker token — no verbosity, no explanation |

> **Why Qwen2.5-0.5B?** Instruction-following capability in a 500M parameter footprint → deployable on CPU / edge with tuned adapter only (~50MB).

---

## 5. Model Iteration: v1 → v1.1 → v1.2

**How each version improved and what problem it solved:**

| Version | Test (29) | Ambiguous (20) | Hard (30) | Key Fix |
|:---|:---:|:---:|:---:|:---|
| Rule baseline | 96.55% | 95.00% | 30.00% | Pattern matching only |
| Base + plain prompt | 51.72% | 40.00% | 13.33% | No fine-tuning, poor generalization |
| Base + policy-aware | 75.86% | 70.00% | 23.33% | Better prompting, still limited |
| **LoRA v1** | 68.97% | 55.00% | 50.00% | First LoRA; hard-eval +2× vs rule |
| **LoRA v1.1 + normalize** | 79.31% | 75.00% | 50.00% | +Normalize layer; better aliases |
| **LoRA v1.2 + normalize** | **93.10%** | **95.00%** | **60.00%** | +Data patches; GOOG/GOOGL confusion → 0 |

**The key turning point — GOOG ↔ GOOGL confusion rate:**

| Version | GOOG ↔ GOOGL errors (test set) |
|:---|:---:|
| LoRA v1 | 6 |
| LoRA v1.1 + normalize | 3 |
| **LoRA v1.2 + normalize** | **0** ✅ |

---

## 6. Full Benchmark Results (7 Systems × 5 Splits)

| System | Test (29) | Ambiguous (20) | Hard (30) | Google ShareClass (24) | Meta Alias (12) | Hallucination | Verbosity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Rule baseline | 96.55% | 95.00% | 30.00% | — | — | 0% | 0% |
| Base + plain prompt | 51.72% | 40.00% | 13.33% | — | — | — | — |
| Base + policy-aware | 75.86% | 70.00% | 23.33% | — | — | — | — |
| LoRA v1 (seed42) | 68.97% | 55.00% | 50.00% | — | — | 0% | 0% |
| LoRA v1.1 + normalize | 79.31% | 75.00% | 50.00% | 70.83% | 66.67% | 0% | 0% |
| **LoRA v1.2 + normalize** | **93.10%** | **95.00%** | **60.00%** | **100.00%** | 58.33% | **0%** | **0%** |

**Key takeaways:**
- LoRA v1.2 achieves **100% on Google share-class** — the hardest structural disambiguation task
- Hard eval **60% = 2× the rule baseline** (30%) — robust on colloquial real-user phrasing
- **0% hallucination** maintained throughout all LoRA versions
- Meta alias (58.33%) remains the **only remaining weakness** — partially addressed by normalize layer (`FB→META`)

> Source: `results/comparison_matrix.csv`

---

## 7. Error Analysis: Where the Model Still Struggles

**Remaining failure categories in LoRA v1.2:**

| Category | Example Input | Gold | Predicted | Root Cause |
|:---|:---|:---:|:---:|:---|
| Indirect product reference | *"company that runs Instagram"* | `META` | `DIS` | Indirect association not seen in training |
| Legacy ticker not normalized | *"What's happening with FB?"* | `META` | `UNKNOWN` | `FB` not direct; normalize catches this post-model |
| Colloquial brand reference | *"company building the Cybertruck"* | `TSLA` | `INTC` | Product→brand→ticker is a multi-hop reasoning |
| BRK-A vs BRK-B nuance | *"Berkshire shares for everyday investors"* | `BRK-B` | `BRK-A` | Subtle phrasing, insufficient training signal |

**The data flywheel is designed to fix these:** every misclassification generates a new JSONL training pair for the next training cycle.

> **Hard eval at 60% is a floor, not a ceiling** — the architecture is designed for continuous improvement via the flywheel.

---

## 8. End-to-End Workflow: n8n + MCP + Data Flywheel

```
┌─────────────────────────────────────────────────────────────┐
│                    n8n Orchestration Layer                    │
│                                                               │
│  [Webhook] → [LoRA Resolver] → [MCP × 2] → [Memo Builder]   │
│                                    ↑                          │
│              Yahoo Finance: /get_price_history                │
│              Yahoo Finance: /get_ticker_info                  │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
  Budget Guard                        Research Memo
  (default: 10 calls)           with [1][2] citation markers
         │                                    │
         ▼                                    ▼
  Response Cache                    ┌──────────────────┐
  (repeated calls = free)           │  Data Flywheel   │
                                    │  JSONL artifact  │
                                    │  Prompt↔Response │
                                    └──────────────────┘
```

| Feature | Implementation |
|:---|:---|
| **Budget limiting** | `budget_limit` param (default: 10); halts before overspend |
| **Response caching** | `_CACHE` dict — (symbol, months) → no repeat API calls |
| **Citation markers** | `[1]`/`[2]` on all numeric data lines + Data Sources footer |
| **MCP endpoints** | `GET /mcp/get_price_history` · `GET /mcp/get_ticker_info` · `POST /resolve` |

---

## 9. Live Demo Showcase

### Demo 1: Google Share-Class Disambiguation
```
Input:  "Research Alphabet class A for me"
Model:  GOOGL  ✅  (Class A, voting shares — not GOOG)
Memo:   Current Price $174.32 [1] · Period Return ▲8.15% [1]
```

### Demo 2: Legacy Alias Normalization
```
Input:  "Research FB for the past 3 months"
Model:  FB  →  Normalize: FB → META  ✅
Memo:   Current Price $589.14 [1] · Period Return ▲4.97% [1]
```

### Demo 3: Standard Ticker (Smoke Test)
```
Input:  "What's the outlook on Apple?"
Model:  AAPL  ✅  (direct match, no normalization needed)
Memo:   Standard research memo generated with full citation trail
```

> All three cases produce research memos with strict `[1]`/`[2]` citation markers. Source data: `demo_assets/`.

---

## 10. Deliverables & Next Steps

**All artifacts committed and ready:**

| Type | Artifact | Location |
|:---|:---|:---|
| **Model Training** | `train_lora.py` · `lora_v1p2.yaml` | `scripts/` · `configs/` |
| **Inference** | `run_lora_infer.py` · `app.py` · `workflow.py` | `scripts/` · root |
| **Workflow** | `n8n_workflow.json` · `src/mcp_server.py` | root · `src/` |
| **Evaluation** | `comparison_matrix.csv` (7 systems × 5 splits) | `results/` |
| **Demo** | 3 annotated cases (input + full memo output) | `demo_assets/` |
| **Final Predictions** | `*_final_preds.jsonl` for all 3 frozen eval sets | `data/` |
| **Data Flywheel** | `flywheel_log.jsonl` — auto-generated JSONL training pairs | `data/` |

**To run the full demo:**
```bash
./run_demo.sh
# or individually:
python app.py --query "Research Alphabet class A for me"
```

**Next iteration targets:**
- Indirect product-reference cases (Instagram→META, Cybertruck→TSLA)
- Meta alias coverage beyond FB (e.g., "Zuckerberg's company")
- Extend symbol universe beyond 20 tickers