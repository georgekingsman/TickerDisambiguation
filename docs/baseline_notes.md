# Baseline Design Notes | 基线方案设计记录

## Overview | 概述

Two baselines are used to establish the "before" performance level, against which the LoRA fine-tuned model will be compared.
两个基线用于建立“微调前”的性能水平，作为 LoRA 微调模型的对比基线。

---

## Baseline A: Rule-Based Keyword Matching | 基线 A：基于规则的关键词匹配

**Implementation | 实现**: `scripts/evaluate.py` → `rule_baseline_predict()`

**Approach | 方法**:
1. Check for ticker format variants (BRK.B, BRK/B, BRK B) → normalize to BRK-B | 检查代码格式变体并标准化
2. Check for direct ticker mentions (AAPL, GOOGL, etc.) in the text | 检查文本中直接提到的代码
3. Apply company-specific rules | 应用公司特定规则：
   - Google/Alphabet + "class C" / "non-voting" → GOOG, else → GOOGL
   - Berkshire/Buffett + "class A" / "expensive" → BRK-A, else → BRK-B
4. Fall back to alias dictionary lookup (longest-match-first) | 回退到别名字典查找（最长匹配优先）
5. Return UNKNOWN if no match | 无匹配时返回 UNKNOWN

**Strengths | 优点**: High precision on well-formatted inputs; deterministic. 在格式规范的输入上准确率高；确定性强。

**Weaknesses | 缺点**:
- Fails on colloquial references (“Bezos company”, “Mouse House”) | 在口语化引用上失败
- Fails on indirect class descriptions (“cheaper share class”) | 在间接类别描述上失败
- Cannot generalize to unseen phrasings | 无法泛化到未见过的表达
- Requires manual rule maintenance | 需要手动维护规则

**Results on test.jsonl (29 samples)**:
- Exact Match Accuracy: 96.55% (28/29)
- Macro-F1: 0.9143
- Error: "Research the Bezos company." → UNKNOWN (expected AMZN)

**Results on ambiguous_eval.jsonl (20 samples)**:
- Exact Match Accuracy: 95.00% (19/20)
- Macro-F1: 0.9532
- Error: "Research the cheaper Google share class." → GOOGL (expected GOOG)

---

## Baseline B: Base Model Zero-Shot | 基线 B：基础模型 Zero-Shot

**Model | 模型**: TBD (candidate: Qwen2.5-0.5B, Phi-3-mini, or similar small LLM suitable for LoRA)
待定（候选：Qwen2.5-0.5B、Phi-3-mini 或其他适合 LoRA 的小型 LLM）

**Prompt Template | 提示词模板**: See `prompts/zero_shot_prompt.txt` | 见 `prompts/zero_shot_prompt.txt`

```text
Resolve the correct stock ticker symbol from the user request below.
Return ONLY the ticker symbol (e.g. AAPL, GOOGL, BRK-B). Do not include any explanation.

Rules:
- "Google" or "Alphabet" without a class → GOOGL
- "Google class C" or "Alphabet class C" → GOOG
- "Berkshire" or "Berkshire Hathaway" without a class → BRK-B
- "Berkshire class A" → BRK-A
- "Facebook" or "FB" → META
- Normalize ticker formats: BRK.B → BRK-B

Request: {input}
Answer:
```

**How to Run | 运行方法**: 
```bash
# 1. Generate predictions (to be implemented in next phase)
# 1. 生成预测（下一阶段实现）
python scripts/run_zero_shot.py --model <model_name> --gold data/test.jsonl --output data/test_zeroshot_preds.jsonl

# 2. Evaluate | 评估
python scripts/evaluate.py --predictions data/test_zeroshot_preds.jsonl --gold data/test.jsonl
```

**Expected Behaviour | 预期表现**:
- Should handle common tickers well (AAPL, TSLA, MSFT) | 应能很好地处理常见代码
- Expected to struggle with: share-class disambiguation, format normalization, colloquial references | 预计在股类消歧、格式标准化、口语化引用上会困难
- This is the primary "before" in the before/after comparison | 这是前后对比中的主要“微调前”基线

---

## Evaluation Protocol | 评估协议

### Metrics | 指标

| Metric 指标 | Purpose 用途 |
|---|---|
| Exact Match Accuracy 精确匹配准确率 | Overall correctness — headline number | 整体正确率——标题指标 |
| Macro-F1 宏平均 F1 | Fairness across low-frequency labels | 低频标签的公平性 |
| Ambiguity Subset Accuracy 歧义子集准确率 | Targeted metric for the hard cases that justify fine-tuning | 针对难场景的核心指标 |

### Comparison Matrix (to be filled after training) | 对比矩阵（训练后填写）

| Model 模型 | Prompt 提示词 | Test Accuracy 测试准确率 | Test Macro-F1 | Ambiguity Accuracy 歧义准确率 | Hard Accuracy 困难集准确率 |
|---|---|---|---|---|---|
| Rule baseline | N/A | **96.55%** | **0.914** | **95.00%** | 30.00% |
| Qwen2.5-0.5B-Instruct (zero-shot) | Plain | 51.72% | 0.447 | 40.00% | 13.33% |
| Qwen2.5-0.5B-Instruct (zero-shot) | Policy-aware | 75.86% | 0.757 | 70.00% | 23.33% |
| LoRA fine-tuned | TBD | TBD | TBD | TBD | TBD |

### Error Analysis Checklist | 错误分析检查表

For each baseline and the fine-tuned model, examine:
对每个基线和微调模型，检查：

1. **Class confusion | 股类混淆**: Did it confuse GOOG ↔ GOOGL or BRK-A ↔ BRK-B? | 是否混淆了这些代码？
2. **Alias failure | 别名失败**: Did it fail on colloquial names? | 是否在口语化名称上失败？
3. **Format errors | 格式错误**: Did it output BRK.B instead of BRK-B? | 是否输出了非标准格式？
4. **Hallucination | 幻觉**: Did it invent a ticker not in the universe? | 是否编造了不在范围内的代码？
5. **Refusal / empty output | 拒绝/空输出**: Did it fail to produce any answer? | 是否未能产生任何答案？

---

## Next Steps | 下一步

1. Select base model for fine-tuning (small enough for LoRA on limited compute) | 选择基础模型（足够小以便在有限计算资源上进行 LoRA）
2. Run zero-shot baseline and fill in the comparison matrix | 运行 zero-shot 基线并填写对比矩阵
3. Implement LoRA training script | 实现 LoRA 训练脚本
4. Run fine-tuned model and compare | 运行微调模型并对比结果
