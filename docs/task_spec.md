# Task Specification: Ticker Disambiguation for Investment Research Requests
# 任务规格说明：投资研究请求中的股票代码消歧

## 1. Task Name | 任务名称

**Ticker Disambiguation / Symbol Normalization**
**股票代码消歧 / 代码标准化**

Given a natural-language investment research request, extract and resolve the target company or stock reference to a single, standardized US equity ticker symbol.
给定一句自然语言的投资研究请求，提取并解析目标公司或股票引用，输出单个标准化的美股代码。

## 2. Input | 输入

A single user utterance (1–2 sentences) that contains a reference to a publicly traded company. Examples:
一句包含上市公司引用的用户请求（1–2 句）。示例：

| User Request 用户请求 | Expected Symbol 期望代码 |
|---|---|
| "Give me a quick memo on Google class C over the last 6 months." | `GOOG` |
| "Research Alphabet class A for me." | `GOOGL` |
| "How is Berkshire B doing recently?" | `BRK-B` |
| "Analyze Meta's earnings trend." | `META` |
| "Pull up the latest on Tesla." | `TSLA` |

## 3. Output | 输出

A single standardized ticker symbol string (e.g. `GOOG`, `BRK-B`). No extra text, no explanation — just the symbol.
单个标准化股票代码字符串（如 `GOOG`、`BRK-B`）。无额外文字，无解释——只返回代码。

## 4. Task Boundary | 任务边界

### In Scope (v1) | 范围内（第一版）

| Category 类别 | Description 描述 | Example 示例 |
|---|---|---|
| Company alias mapping 公司别名映射 | Common names / former names → canonical ticker 常用名/旧名 → 标准代码 | "Facebook" → `META`, "Google" → `GOOGL` |
| Share-class disambiguation 股类消歧 | Distinguish class A / B / C when explicitly mentioned 明确提及股类时进行区分 | "Google class C" → `GOOG` |
| Ticker format normalization 代码格式标准化 | Dot / slash variants → standard dash format 点/斜杠变体转换为标准破折号格式 | `BRK.B` → `BRK-B` |

### Out of Scope (v1) | 范围外（第一版）

- Multi-ticker comparison requests | 多股票对比请求
- Investment advice / scoring | 投资建议 / 评分
- Long-form research summarization | 长文研究摘要
- News sentiment analysis | 新闻情感分析
- Non-US equities | 非美股
- Crypto / forex / commodity symbols | 加密货币 / 外汇 / 商品代码

## 5. Symbol Universe (v1) | 代码范围（第一版）

20 symbols selected to balance coverage and ambiguity density:

```
AAPL  MSFT  AMZN  META  TSLA  NVDA  NFLX  GOOGL  GOOG
BRK-A  BRK-B  JPM  BAC  AMD  INTC  PLTR  DIS  TSM  BABA  PDD
```

### Key Ambiguity Groups | 核心歧义组

| Ambiguity Group 歧义组 | Symbols 代码 | Trigger Expressions 触发表达 |
|---|---|---|
| Google / Alphabet share classes | `GOOGL` (class A), `GOOG` (class C) | "Google", "Alphabet", "class A", "class C" |
| Berkshire Hathaway share classes | `BRK-A` (class A), `BRK-B` (class B) | "Berkshire", "class A", "class B" |
| Meta / Facebook rebrand | `META` | "Facebook", "Meta", "FB" |
| Ticker format variants | `BRK-B` | "BRK.B", "BRK/B", "BRK B" |

## 6. Data Format | 数据格式

JSONL, instruction-tuning style | JSONL 指令微调格式：

```json
{"instruction": "Resolve the stock ticker symbol from the user request. Return only the ticker.", "input": "Give me a quick memo on Google class C over the last 6 months.", "output": "GOOG"}
```

## 7. Evaluation Metrics | 评估指标

| Metric 指标 | Description 描述 |
|---|---|
| **Exact Match Accuracy** | % of predictions that exactly equal the gold label |
| **Macro-F1** | Macro-averaged F1 across all symbols in the label space |
| **Ambiguity Subset Accuracy** | Exact match on the hand-curated ambiguous eval set |

The ambiguity subset accuracy is the headline metric for demonstrating fine-tuning value: it directly shows improvement on the hardest cases (share-class disambiguation, alias resolution).

## 8. Why This Task Is Worth Fine-Tuning

1. **Base models are inconsistent** on share-class disambiguation — they cannot reliably distinguish `GOOG` vs `GOOGL` or `BRK-A` vs `BRK-B` without explicit guidance.
2. **Rule-based systems are brittle** — they fail on novel phrasings and colloquial language.
3. **The task is small and well-bounded** — perfect for lightweight LoRA fine-tuning with minimal compute.
4. **It's a natural entry point** for a research copilot pipeline — symbol resolution is the first step before any downstream data retrieval or analysis.

## 9. Demo Narrative

> The base model handles common tickers well but struggles with ambiguous references like share classes and rebranded companies. After lightweight LoRA fine-tuning on ~100 examples, the model resolves these edge cases reliably, unlocking accurate downstream research automation.
