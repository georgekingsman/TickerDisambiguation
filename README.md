# Ticker Disambiguation for Investment Research | 投资研究中的股票代码消歧

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
├── data/                          # Datasets | 数据集
│   ├── train.jsonl                # Training set (80 samples) | 训练集（80 条）
│   ├── val.jsonl                  # Validation set (20 samples) | 验证集（20 条）
│   ├── test.jsonl                 # Test set (29 samples) | 测试集（29 条）
│   └── ambiguous_eval.jsonl       # Ambiguity-focused eval (20 samples) | 歧义专项评估（20 条）
├── docs/                          # Documentation | 文档
│   ├── task_spec.md               # Task definition & metrics | 任务定义与指标
│   ├── label_policy.md            # Labeling rules & defaults | 标注规则与默认映射
│   └── baseline_notes.md          # Baseline design & results | 基线方案与结果
├── prompts/                       # Prompt templates | 提示词模板
│   └── zero_shot_prompt.txt       # Zero-shot inference prompt | Zero-shot 推理提示词
└── scripts/                       # Code | 代码
    ├── build_dataset.py           # Dataset generation & split | 数据集生成与切分
    └── evaluate.py                # Evaluation + rule baseline | 评估脚本 + 规则基线
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

### Rule Baseline | 规则基线

| Eval Set 评估集 | Accuracy 准确率 | Macro-F1 | Typical Error 典型错误 |
|---|---|---|---|
| test.jsonl | 96.55% | 0.914 | "Bezos company" → UNKNOWN |
| ambiguous_eval.jsonl | 95.00% | 0.953 | "cheaper Google share class" → GOOGL (should be GOOG 应为 GOOG) |

**Key Insight 关键发现**: The rule baseline fails on colloquial references and indirect class descriptions — exactly the cases where fine-tuning adds value.  
规则基线在口语化引用和间接类别描述上失败——这正是微调能发挥价值的地方。

---

## Quick Start | 快速开始

### Regenerate Dataset | 重新生成数据集

```bash
python scripts/build_dataset.py
```

### Run Rule Baseline Evaluation | 运行规则基线评估

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
- [x] Dataset v1 (129 samples + 20 ambiguity eval) | 数据集 v1
- [x] Rule baseline + evaluation script | 规则基线 + 评估脚本
- [ ] Zero-shot base model baseline | Zero-shot 基础模型基线
- [ ] LoRA fine-tuning script | LoRA 微调脚本
- [ ] Before/after comparison report | 微调前后对比报告
- [ ] Integration with research copilot workflow | 集成到研究 copilot 工作流

---

## License

MIT
