# Ticker Disambiguation for Investment Research📈 | 投资研究中的股票代码消歧🤔

> **Resolve ambiguous company names and aliases to standardized US equity ticker symbols.**
> **将模糊的公司名称和别名解析为标准化的美股股票代码。**

---

## Project Overview | 项目概述

This project builds a lightweight fine-tuning pipeline for **ticker symbol disambiguation** — mapping natural-language references like "Google class C" or "Berkshire B" to their correct standardized tickers (`GOOG`, `BRK-B`).

本项目构建了一个轻量级微调流水线，用于**股票代码消歧** —— 将自然语言中的公司引用（如 "Google class C" 或 "Berkshire B"）映射到正确的标准代码（`GOOG`、`BRK-B`）。

### Why This Task? | 为什么选择这个任务？

1. **Clear boundaries | 边界清晰** — Unlike open-ended financial Q&A, this is a well-defined classification task.  
   与开放式金融问答不同，这是一个定义明确的分类任务。

2. **Easy before/after comparison | 易于做前后对比** — Perfect for demonstrating zero-shot vs. LoRA fine-tuned model improvements.  
   非常适合展示 zero-shot 基础模型与 LoRA 微调后的差距。

3. **Natural pipeline entry point | 自然的流水线入口** — Symbol resolution is the first step in any research copilot workflow.  
   代码解析是研究 copilot 工作流的第一步。

---

## Project Structure | 项目结构

```
TickerDisambiguation/
├── configs/                       # Experiment configs | 实验配置
│   └── lora_v1.yaml               # LoRA v1 hyperparameters | LoRA v1 超参数
├── data/                          # Datasets | 数据集
│   ├── train.jsonl                # Training set v1 (80 samples) | 训练集 v1
│   ├── train_v2.jsonl             # Training set v2 (392 samples) | 训练集 v2（含6类增强）
│   ├── val.jsonl                  # Validation set v1 (20 samples) | 验证集 v1
│   ├── val_v2.jsonl               # Validation set v2 (69 samples) | 验证集 v2
│   ├── dev_hard.jsonl             # Hard dev set for checkpoint selection (40 samples) | 困难开发集
│   ├── test.jsonl                 # Frozen test set (29 samples) | 冻结测试集
│   ├── ambiguous_eval.jsonl       # Frozen ambiguity eval (20 samples) | 冻结歧义评估集
│   └── hard_eval.jsonl            # Frozen hard/colloquial eval (30 samples) | 冻结困难评估集
├── docs/                          # Documentation | 文档
│   ├── task_spec.md               # Task definition & metrics | 任务定义与指标
│   ├── label_policy.md            # Labeling rules & defaults | 标注规则与默认映射
│   ├── baseline_notes.md          # Baseline design & results | 基线方案与结果
│   ├── experiment_protocol.md     # Frozen experiment protocol | 冻结实验协议
│   └── zero_shot_results.md       # Zero-shot experiment results | Zero-shot 实验结果
├── prompts/                       # Prompt templates | 提示词模板
│   ├── zero_shot_plain.txt        # Plain zero-shot prompt (no rules) | 纯 zero-shot 提示词
│   └── zero_shot_prompt.txt       # Policy-aware zero-shot prompt | 含规则的 zero-shot 提示词
├── results/                       # Evaluation results | 评估结果
│   └── comparison_matrix.csv      # E0–E4 comparison matrix | E0–E4 对比矩阵
└── scripts/                       # Code | 代码
    ├── build_dataset.py           # Dataset v1 generation | 数据集 v1 生成
    ├── build_dataset_v2.py        # Dataset v2 generation (6-category augmented) | 数据集 v2 生成
    ├── evaluate.py                # Evaluation + rule baseline | 评估脚本 + 规则基线
    ├── run_zero_shot.py           # Zero-shot model inference | Zero-shot 模型推理
    ├── train_lora.py              # LoRA fine-tuning (multi-seed, smoke/full) | LoRA 微调训练
    ├── run_lora_infer.py          # LoRA inference | LoRA 推理
    ├── run_experiment.sh          # One-shot experiment pipeline | 一键实验流水线
    └── error_analysis.py          # Error delta analysis (E1 vs E2 vs E3) | 错误差值分析
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

## Baseline Results | 基线结果

### Comparison Matrix | 对比矩阵

| ID | System 系统 | Test Acc (29) | Test F1 | Ambiguity Acc (20) | Hard Acc (30) | Hallucination | Verbosity |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 | Rule baseline 规则基线 | **96.55%** | **0.914** | **95.00%** | 30.00% | 0.00% | 0.00% |
| E1 | Base + plain prompt 基础+纯提示 | 51.72% | 0.447 | 40.00% | 13.33% | TBD | TBD |
| E2 | Base + policy-aware 基础+策略提示 | 75.86% | 0.757 | 70.00% | 23.33% | TBD | TBD |
| E3 | LoRA v1 + plain prompt | TBD | TBD | TBD | TBD | TBD | TBD |

> Full results and error analysis in [docs/zero_shot_results.md](docs/zero_shot_results.md)  
> 完整结果和错误分析见 [docs/zero_shot_results.md](docs/zero_shot_results.md)

**Key Insight 关键发现**: The rule baseline scores 96.55% on test but **only 30% on hard_eval** (colloquial/indirect references). The base model (zero-shot) shows learnable error structure — class confusion and alias failures are systematic patterns that LoRA can fix.  
规则基线在测试集上 96.55%，但在困难集上**仅 30%**。基础模型的错误是有结构的——股类混淆和别名失败是可通过 LoRA 学习修复的系统性模式。

---

## Quick Start | 快速开始

### 1. LoRA Training & Evaluation (One Command) | LoRA 训练与评估（一条命令）

```bash
export KMP_DUPLICATE_LIB_OK=TRUE

# Smoke run first (sanity check, ~2 min) | 先跑 smoke run 验证链路
bash scripts/run_experiment.sh smoke

# Full run with seed 42 | 正式训练 seed 42
bash scripts/run_experiment.sh full 42

# Full run with seed 7 (stability check) | 换种子验证稳定性
bash scripts/run_experiment.sh full 7

# Error delta analysis only | 仅做错误分析
bash scripts/run_experiment.sh analysis
```

### 2. Step-by-Step (Manual) | 分步手动执行

```bash
# Train | 训练
python scripts/train_lora.py --config configs/lora_v1.yaml --seed 42

# Inference | 推理
python scripts/run_lora_infer.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --adapter checkpoints/lora_v1_seed42 \
    --gold data/test.jsonl \
    --prompt prompts/zero_shot_plain.txt \
    --output data/test_lora_v1_preds.jsonl

# Evaluate | 评估
python scripts/evaluate.py --predictions data/test_lora_v1_preds.jsonl --gold data/test.jsonl

# Error analysis | 错误分析
python scripts/error_analysis.py
```

### 3. Regenerate Dataset | 重新生成数据集

```bash
python scripts/build_dataset.py      # v1 (80+20 train/val)
python scripts/build_dataset_v2.py   # v2 (392+69 train/val, 40 dev_hard)
```

### 4. Run Rule Baseline Evaluation | 运行规则基线评估

```bash
# Evaluate on test set | 在测试集上评估
python scripts/evaluate.py --run-rule-baseline --gold data/test.jsonl

# Evaluate on ambiguity set | 在歧义集上评估
python scripts/evaluate.py --run-rule-baseline --gold data/ambiguous_eval.jsonl
```

### Evaluate Custom Predictions | 评估自定义预测

```bash
python scripts/evaluate.py --predictions your_preds.jsonl --gold data/test.jsonl
```

Prediction file format | 预测文件格式：
```json
{"input": "Research Google class C.", "predicted": "GOOG"}
```

---

## Roadmap | 路线图

- [x] Task definition & label policy | 任务定义与标注策略
- [x] Dataset v1 (129 samples + 20 ambiguity eval + 30 hard eval) | 数据集 v1
- [x] Rule baseline + evaluation script | 规则基线 + 评估脚本
- [x] Zero-shot inference script + dual prompt variants | Zero-shot 推理脚本 + 双提示词变体
- [x] Zero-shot experiments & comparison matrix | Zero-shot 实验并填写对比矩阵
- [x] Dataset v2 (392 train + 69 val + 40 dev_hard) | 数据集 v2（6 类增强）
- [x] LoRA fine-tuning script (multi-seed, smoke/full) | LoRA 微调脚本（多种子，快速/完整）
- [x] LoRA inference script | LoRA 推理脚本
- [x] One-shot experiment pipeline | 一键实验流水线
- [x] Error delta analysis script | 错误差值分析脚本
- [ ] LoRA v1 training & E3 results | LoRA v1 训练与 E3 结果
- [ ] Before/after comparison report | 微调前后对比报告
- [ ] LoRA v2 data augmentation (if needed) | LoRA v2 数据增强（按需）
- [ ] Integration with research copilot workflow | 集成到研究 copilot 工作流

---

## License

MIT
