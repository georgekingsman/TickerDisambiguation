# Ticker Disambiguation for Investment Research📈 | 投资研究中的股票代码消歧🤔

> **Resolve ambiguous company names and aliases to standardized US equity ticker symbols.**
> **将模糊的公司名称和别名解析为标准化的美股股票代码。**

---

## Final Results | 最终结果 🏆

**Final system: LoRA v1.2 + normalize(FB→META, BRK.B→BRK-B)**

| System 系统 | Test (29) | Ambiguous (20) | Hard (30) | Google ShareClass (24) | Meta Alias (12) | Hallucination | Verbosity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule baseline | 96.55% | 95.00% | 30.00% | — | — | 0% | 0% |
| Base + plain prompt | 51.72% | 40.00% | 13.33% | — | — | — | — |
| Base + policy-aware | 75.86% | 70.00% | 23.33% | — | — | — | — |
| LoRA v1 | 68.97% | 55.00% | 50.00% | — | — | 0% | 0% |
| LoRA v1.1 + normalize | 79.31% | 75.00% | 50.00% | 70.83% | 66.67% | 0% | 0% |
| **LoRA v1.2 + normalize** | **93.10%** | **95.00%** | **60.00%** | **100.00%** | 58.33% | **0%** | **0%** |

**Key wins | 关键突破:**
- GOOG↔GOOGL confusion: **completely eliminated** (was 6 → 3 → 0) | GOOG↔GOOGL 混淆：**彻底消除**
- Google share class accuracy: **100%** (was 70.83%) | Google 股类准确率：**100%**
- Hard eval: **60%** — 2x the rule baseline | 困难集：**60%** — 是规则基线的两倍
- Hallucination & verbosity: **0%** throughout | 幻觉和冗余：始终 **0%**

---

## Project Overview | 项目概述

This project builds a **hybrid ticker symbol resolver** for investment research — combining LoRA fine-tuning for natural language understanding with a lightweight normalize layer for canonicalization.

本项目构建了一个**混合股票代码解析器**用于投资研究 — 结合 LoRA 微调进行自然语言理解，以及轻量级标准化层进行代码规范化。

### Architecture | 架构

```
User Query → [LoRA v1.2 Resolver] → [Normalize Layer] → [Market Data] → [Research Memo]
                                        FB→META
                                        BRK.B→BRK-B
```

### Why This Task? | 为什么选择这个任务？

1. **Clear boundaries | 边界清晰** — Well-defined classification task with measurable accuracy.
2. **Hard edge cases | 困难边界情况** — Google/Alphabet share classes, Meta/FB rebrand, BRK-A/B require real understanding.
3. **Natural pipeline entry point | 流水线入口** — Symbol resolution is the first step in any research copilot workflow.

---

## Project Structure | 项目结构

```
TickerDisambiguation/
├── workflow.py                    # End-to-end copilot (resolve → data → memo)
├── configs/                       # Experiment configs | 实验配置
│   ├── lora_v1.yaml               # LoRA v1 config | LoRA v1 配置
│   ├── lora_v1p1.yaml             # LoRA v1.1 config (+ Google/Meta patch)
│   └── lora_v1p2.yaml             # LoRA v1.2 config (+ GOOGL bias fix) [FINAL]
├── checkpoints/                   # Trained LoRA adapters | 训练好的 LoRA 适配器
│   └── lora_v1p2_seed42/          # Final adapter [FINAL]
├── data/                          # Datasets & predictions | 数据集与预测
│   ├── train_v2p2.jsonl           # Final training set (520 samples) [FINAL]
│   ├── val_v2.jsonl               # Validation set (69 samples)
│   ├── test.jsonl                 # Frozen test set (29 samples)
│   ├── ambiguous_eval.jsonl       # Frozen ambiguity eval (20 samples)
│   ├── hard_eval.jsonl            # Frozen hard/colloquial eval (30 samples)
│   ├── google_shareclass_eval.jsonl  # Google share class diagnostic (24)
│   ├── meta_alias_eval.jsonl      # Meta/FB alias diagnostic (12)
│   ├── test_final_preds.jsonl     # Final predictions — test
│   ├── ambiguous_final_preds.jsonl   # Final predictions — ambiguous
│   └── hard_final_preds.jsonl     # Final predictions — hard
├── docs/                          # Documentation | 文档
│   ├── task_spec.md               # Task definition & metrics
│   ├── label_policy.md            # Labeling rules & defaults
│   ├── baseline_notes.md          # Baseline design & results
│   ├── experiment_protocol.md     # Frozen experiment protocol
│   └── zero_shot_results.md       # Zero-shot experiment results
├── prompts/                       # Prompt templates | 提示词模板
│   ├── zero_shot_plain.txt        # Plain prompt (used in final) [FINAL]
│   └── zero_shot_prompt.txt       # Policy-aware prompt
├── results/                       # Evaluation results | 评估结果
│   ├── comparison_matrix.csv      # Full comparison matrix
│   └── v1p2_*.json               # Final v1.2 eval results
└── scripts/                       # Code | 代码
    ├── train_lora.py              # LoRA fine-tuning
    ├── run_lora_infer.py          # LoRA inference (with normalize)
    ├── evaluate.py                # Evaluation + rule baseline
    ├── run_zero_shot.py           # Zero-shot inference
    ├── run_v1p2_experiment.sh     # v1.2 experiment pipeline [FINAL]
    ├── build_dataset.py           # Dataset v1 generation
    ├── build_dataset_v2.py        # Dataset v2 generation
    ├── error_analysis.py          # Error delta analysis
    └── compare_v1_v1p1.py         # v1 vs v1.1 comparison
```

---

## Task Definition | 任务定义

| Item | Description |
|---|---|
| **Input 输入** | A natural-language investment research request (1–2 sentences) 一句自然语言投资研究请求 |
| **Output 输出** | A single standardized ticker symbol (e.g. `GOOGL`, `BRK-B`) 单个标准化股票代码 |
| **Symbol Universe 代码范围** | 20 common US equities (see below) 20 个常见美股 |

### Symbol Universe (v1) | 代码范围（第一版）

```
AAPL  MSFT  AMZN  META  TSLA  NVDA  NFLX  GOOGL  GOOG
BRK-A  BRK-B  JPM  BAC  AMD  INTC  PLTR  DIS  TSM  BABA  PDD
```

### Key Ambiguity Cases | 核心歧义场景

| Ambiguity 歧义 | Symbols 代码 | Trigger 触发条件 |
|---|---|---|
| Google / Alphabet share classes | `GOOGL` (A类) vs `GOOG` (C类) | "class A", "class C", "voting", "non-voting" |
| Berkshire Hathaway share classes | `BRK-A` (A类) vs `BRK-B` (B类) | "class A", "class B" |
| Meta / Facebook rebrand | `META` | "Facebook", "FB", "Meta" |
| Ticker format variants | `BRK-B` | "BRK.B", "BRK/B", "BRK B" |

---

## Label Policy (Key Defaults) | 标注策略（关键默认规则）

> Full details in [docs/label_policy.md](docs/label_policy.md) | 完整规则见 label_policy.md

| Expression 表达 | Default Symbol 默认代码 | Rationale 理由 |
|---|---|---|
| "Google" / "Alphabet" (no class) | `GOOGL` | Class A is more commonly referenced A类是更常用的引用 |
| "Google class C" | `GOOG` | Explicit class C override 显式 C 类覆盖 |
| "Berkshire" (no class) | `BRK-B` | Class B is the retail-accessible share B股是散户可买的股 |
| "Facebook" / "FB" | `META` | Former name → current ticker 旧名称映射到当前代码 |

---

## Data Schema | 数据格式

JSONL, instruction-tuning style | JSONL 指令微调格式：

```json
{
  "instruction": "Resolve the stock ticker symbol from the user request. Return only the ticker.",
  "input": "Give me a quick memo on Google class C over the last 6 months.",
  "output": "GOOG"
}
```

### Dataset Composition | 数据集组成

| Category 类别 | Proportion 占比 | Description 描述 |
|---|---|---|
| A. Direct mapping 直接映射 | ~30% | "Analyze Apple" → `AAPL` |
| B. Alias / colloquial 别名/口语 | ~30% | "Facebook" → `META` |
| C. Ambiguity / tricky 歧义/陷阱 | ~40% | "Google class C" → `GOOG` |

---

## Evaluation Metrics | 评估指标

| Metric 指标 | Description 描述 |
|---|---|
| **Exact Match Accuracy 精确匹配准确率** | % of predictions exactly matching the gold label 预测与标签完全匹配的比例 |
| **Macro-F1** | Macro-averaged F1 across all symbols 所有代码的宏平均 F1 |
| **Ambiguity Subset Accuracy 歧义子集准确率** | Exact match on the curated ambiguous eval set 在歧义评估集上的精确匹配（核心指标） |

---

## Experiment Results | 实验结果

### Full Comparison Matrix | 完整对比矩阵

| ID | System 系统 | Test Acc (29) | Test F1 | Ambiguity (20) | Hard (30) | GOOG↔GOOGL | Halluc. | Verbose |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E0 | Rule baseline | **96.55%** | 0.914 | **95.00%** | 30.00% | — | 0% | 0% |
| E1 | Base + plain prompt | 51.72% | 0.447 | 40.00% | 13.33% | — | — | — |
| E2 | Base + policy-aware | 75.86% | 0.757 | 70.00% | 23.33% | — | — | — |
| E3 | LoRA v1 | 68.97% | 0.696 | 55.00% | 50.00% | 6 | 0% | 0% |
| E4 | LoRA v1.1 + normalize | 79.31% | 0.806 | 75.00% | 50.00% | 3 | 0% | 0% |
| **E5** | **LoRA v1.2 + normalize** | **93.10%** | **0.942** | **95.00%** | **60.00%** | **0** | **0%** | **0%** |

### Diagnostic Slices | 诊断切片

| Slice | LoRA v1.1+norm | **LoRA v1.2** |
|---|---:|---:|
| Google ShareClass (24) | 70.83% | **100.00%** |
| Meta Alias (12) | 66.67% | 58.33% |

> Full results in [results/](results/) | 完整结果见 results/ 目录

**Key Insight | 关键发现**: The rule baseline tops test/ambiguous sets but **collapses to 30% on hard_eval** (colloquial, indirect references). LoRA v1.2 achieves **60% on hard** while matching the rule baseline on standard sets, with zero hallucination. The hybrid architecture (model + normalize) delivers the best of both worlds.

规则基线在标准集上最强，但在困难集（口语化、间接表达）上**崩溃至 30%**。LoRA v1.2 在困难集上达到 **60%**，同时在标准集上匹配规则基线，且零幻觉。混合架构（模型 + 规范化层）兼得两者之长。

---

## Quick Start | 快速开始

### 1. Run the Research Copilot (Demo) | 运行研究 Copilot（演示）

```bash
# Run 3 demo cases: Google class A, FB alias, Apple standard
python workflow.py --demo

# Single query
python workflow.py --query "Research Alphabet class A for me"

# Interactive mode
python workflow.py
```

### 2. Train & Evaluate (Reproduce Results) | 训练与评估（复现结果）

```bash
export KMP_DUPLICATE_LIB_OK=TRUE

# Train LoRA v1.2 and run full evaluation
bash scripts/run_v1p2_experiment.sh full 42

# Evaluate only (skip training, use existing adapter)
bash scripts/run_v1p2_experiment.sh eval 42
```

### 3. Step-by-Step (Manual) | 分步手动执行

```bash
# Train
python scripts/train_lora.py --config configs/lora_v1p2.yaml --seed 42

# Inference
python scripts/run_lora_infer.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter checkpoints/lora_v1p2_seed42 \
    --gold data/test.jsonl \
    --prompt prompts/zero_shot_plain.txt \
    --output data/test_lora_v1p2_seed42_preds.jsonl

# Evaluate
python scripts/evaluate.py \
    --predictions data/test_lora_v1p2_seed42_preds.jsonl \
    --gold data/test.jsonl
```

### 4. Run Rule Baseline | 运行规则基线

```bash
python scripts/evaluate.py --run-rule-baseline --gold data/test.jsonl
```

---

## Roadmap | 路线图

- [x] Task definition & label policy | 任务定义与标注策略
- [x] Dataset v1 → v2 → v2.2 (520 final training samples) | 数据集迭代
- [x] Rule baseline + evaluation script | 规则基线 + 评估脚本
- [x] Zero-shot inference (plain + policy-aware prompts) | Zero-shot 推理
- [x] LoRA v1 training & multi-seed stability check | LoRA v1 训练与多种子验证
- [x] LoRA v1.1 — Google/Meta targeted patch | LoRA v1.1 — Google/Meta 定向修复
- [x] Normalize layer (FB→META, BRK.B→BRK-B) | 规范化层
- [x] LoRA v1.2 — GOOGL bias correction | LoRA v1.2 — GOOGL 偏置修正
- [x] Final comparison matrix & version freeze | 最终对比矩阵与版本冻结
- [x] End-to-end workflow (resolve → data → memo) | 端到端工作流
- [x] 3 demo cases (Google class A, FB alias, Apple standard) | 3 个 Demo 案例
- [ ] Slide deck & oral presentation prep | 演示幻灯片与口头准备

---

## Method Overview | 方法概述

### Hybrid Resolver Architecture | 混合解析器架构

| Layer | Responsibility | Examples |
|---|---|---|
| **Layer 1: LoRA model** | Natural language understanding | "voting shares of Alphabet" → GOOGL |
| **Layer 2: Normalize** | Historical alias canonicalization | FB → META, BRK.B → BRK-B |
| **Layer 3: Guardrail** | (Future) High-risk pattern alerts | Log + flag conflicting signals |

### Training Data Evolution | 训练数据演变

| Version | Samples | Key Addition |
|---|---:|---|
| train_v1 | 80 | Base dataset |
| train_v2 | 392 | 6-category augmentation |
| train_v2p1 | 488 | +96 Google/Meta patch |
| **train_v2p2** | **520** | **+32 GOOGL bias correction** |

### Key Design Decisions | 关键设计决策

1. **Small model, targeted data** — Qwen2.5-0.5B + 520 samples beats large zero-shot models on this task
2. **Normalize ≠ cheating** — FB→META is canonicalization, same as BRK.B→BRK-B
3. **Iterative patching** — Small targeted patches (32–96 samples) fix specific biases without destabilizing other metrics

---

## License

MIT
