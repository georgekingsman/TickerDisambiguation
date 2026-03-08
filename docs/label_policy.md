# Label Policy: Ticker Disambiguation v1
# 标注策略：股票代码消歧 v1

This document defines the deterministic mapping rules used to assign gold labels in the training and evaluation datasets. Any ambiguity in user requests is resolved by the rules below — there is no "either is acceptable" case.

本文档定义了用于在训练和评估数据集中分配标签的确定性映射规则。用户请求中的任何歧义都由以下规则解决——不存在“两个都可以”的情况。

---

## 1. Default Mapping Rules (No Explicit Share-Class Mention)
## 1. 默认映射规则（未明确提及股类时）

When the user mentions a company **without** specifying a share class, the following defaults apply:
当用户提到公司但**未指定股类**时，适用以下默认规则：

| User Expression 用户表达 | Default Symbol 默认代码 | Rationale 理由 |
|---|---|---|
| "Google" | `GOOGL` | Class A is the more commonly traded / referenced share |
| "Alphabet" | `GOOGL` | Parent company name defaults to class A |
| "Berkshire" / "Berkshire Hathaway" | `BRK-B` | Class B is the retail-accessible share (~$400 vs ~$600k) |
| "Facebook" | `META` | Former name maps to current ticker |
| "Meta" | `META` | Direct match |
| "FB" | `META` | Legacy ticker maps to current ticker |

## 2. Explicit Share-Class Override Rules
## 2. 显式股类覆盖规则

When the user **explicitly** mentions a share class, it overrides the default:
当用户**明确提到**股类时，覆盖默认规则：

| User Expression 用户表达 | Symbol 代码 | Rule 规则 |
|---|---|---|
| "Google class C" / "GOOG" | `GOOG` | Explicit class C → GOOG |
| "Google class A" / "Alphabet class A" | `GOOGL` | Explicit class A → GOOGL |
| "Alphabet class C" | `GOOG` | Explicit class C → GOOG |
| "Berkshire class A" / "Berkshire Hathaway class A" | `BRK-A` | Explicit class A → BRK-A |
| "Berkshire class B" / "Berkshire Hathaway class B" | `BRK-B` | Explicit class B → BRK-B |

## 3. Ticker Format Normalization | 代码格式标准化

All ticker symbols use the **dash format** as canonical:
所有股票代码使用**破折号格式**作为标准：

| Input Variant 输入变体 | Normalized Output 标准化输出 |
|---|---|
| `BRK.B` | `BRK-B` |
| `BRK/B` | `BRK-B` |
| `BRK B` | `BRK-B` |
| `BRK.A` | `BRK-A` |

## 4. Direct Ticker Mentions | 直接提到的股票代码

If the user directly uses a valid ticker symbol (correctly formatted or not), resolve it:
如果用户直接使用有效的股票代码（无论格式是否正确），直接解析：

| User Expression 用户表达 | Symbol 代码 |
|---|---|
| "What's happening with AAPL?" | `AAPL` |
| "Look up TSLA" | `TSLA` |
| "BRK.B performance" | `BRK-B` |

## 5. Common Aliases | 常用别名

| Alias 别名 | Symbol 代码 |
|---|---|
| Apple | `AAPL` |
| Microsoft | `MSFT` |
| Amazon | `AMZN` |
| Meta / Facebook / FB | `META` |
| Tesla | `TSLA` |
| Nvidia / NVIDIA | `NVDA` |
| Netflix | `NFLX` |
| Google / Alphabet (no class) | `GOOGL` |
| Berkshire / Berkshire Hathaway (no class) | `BRK-B` |
| JPMorgan / JP Morgan / Chase | `JPM` |
| Bank of America / BofA | `BAC` |
| AMD / Advanced Micro Devices | `AMD` |
| Intel | `INTC` |
| Palantir | `PLTR` |
| Disney / Walt Disney | `DIS` |
| TSMC / Taiwan Semiconductor | `TSM` |
| Alibaba | `BABA` |
| PDD / Pinduoduo / Temu | `PDD` |

## 6. Edge Case Decisions | 边界情况决策

| Scenario 场景 | Decision 决策 | Rationale 理由 |
|---|---|---|
| "Google stock" (no class) | `GOOGL` | Default rule applies |
| "Alphabet shares" (no class) | `GOOGL` | Default rule applies |
| "Berkshire stock" (no class) | `BRK-B` | Default rule applies |
| "voting shares of Alphabet" | `GOOGL` | Class A has voting rights → GOOGL |
| "non-voting Google shares" | `GOOG` | Class C is non-voting → GOOG |

## 7. Policy Version | 策略版本

- **Version 版本**: 1.0
- **Date 日期**: 2026-03-08
- **Symbol Universe Size 代码范围大小**: 20
- **Maintainer 维护者**: Project Author

> **Note | 注意**: This policy must be referenced in any presentation or README when discussing evaluation results, so reviewers understand the ground-truth mapping conventions.
> 在任何展示或 README 中讨论评估结果时，必须引用本策略，以便审阅者理解标签映射规则。
