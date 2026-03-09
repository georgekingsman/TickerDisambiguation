# Experiment Protocol – Ticker Disambiguation LoRA

# 实验协议 – 股票代码消歧 LoRA 微调

> **Status**: Frozen as of 2026-03-08. Do not modify eval sets or decode parameters.

## 1. Base Model

- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- All experiments (zero-shot and LoRA) use this same base model.
- Changing the base model invalidates all before/after comparisons.

## 2. Frozen Evaluation Sets

| File | Samples | Purpose |
|------|---------|---------|
| `data/test.jsonl` | 29 | Main test set |
| `data/ambiguous_eval.jsonl` | 20 | Ambiguity-focused evaluation |
| `data/hard_eval.jsonl` | 30 | Hard / colloquial evaluation |

**Rules:**

- Do NOT add, remove, or modify any samples in these files.
- Do NOT use these samples for training or hyperparameter tuning.
- `data/dev_hard.jsonl` exists for tuning — same distribution, different sentences.

## 3. Decoding Parameters

| Parameter | Value |
|-----------|-------|
| `temperature` | 0 (greedy) |
| `do_sample` | `False` |
| `max_new_tokens` | 8 |

These must remain constant across all experiments.

## 4. Post-Processing Rules

1. Uppercase all output
2. Normalize: `BRK.B` / `BRK/B` / `BRK B` → `BRK-B` (same for A)
3. Extract first valid ticker via regex `\b([A-Z]{1,5}(?:[.\-/][A-Z0-9]{1,3})?)\b`
4. Fallback substring match against known ticker set
5. If no ticker found → `UNKNOWN`

## 5. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Exact match on each test set |
| Macro-F1 | Macro-averaged F1 across all labels |
| Ambiguity Accuracy | Accuracy on `ambiguous_eval.jsonl` |
| Hard Eval Accuracy | Accuracy on `hard_eval.jsonl` |
| Hallucination Rate | % predictions not in 20-symbol ticker universe |
| Verbosity Rate | % outputs with >1 token or explanation text |

## 6. Experiment Matrix

| ID | System | Purpose |
|----|--------|---------|
| E0 | Rule baseline | Engineering baseline |
| E1 | Base model + plain prompt | Raw model capability |
| E2 | Base model + policy-aware prompt | Prompt ceiling |
| E3 | Base model + LoRA(v1) + plain prompt | Main LoRA result |
| E4 | Base model + LoRA(v2, augmented) + plain prompt | Enhanced version |

**Critical**: LoRA experiments (E3, E4) must use **plain prompt only**.
This proves knowledge was internalized, not prompted.

## 7. LoRA Configuration (v1)

| Parameter | Value |
|-----------|-------|
| `r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Max sequence length | 128 |
| Precision | bf16 (fallback fp16/fp32) |

## 8. Ticker Universe (20 symbols)

```
AAPL  MSFT  AMZN  META  TSLA  NVDA  NFLX  GOOGL  GOOG
BRK-A  BRK-B  JPM  BAC  AMD  INTC  PLTR  DIS  TSM  BABA  PDD
```

## 9. Success Criteria

| Level | test | ambiguity | hard_eval | hallucination | verbosity |
|-------|------|-----------|-----------|---------------|-----------|
| Baseline (pass) | > 80% | > 80% | > 40% | ↓ significant | ↓ near 0 |
| Good | ≈ policy-aware | > 85% | > 50% | < 5% | < 3% |
| Strong | > policy-aware | > 90% | > 60% | < 2% | 0% |

## 10. Anti-Leak Policy

- `hard_eval.jsonl` sentences must NEVER appear in training data.
- `ambiguous_eval.jsonl` sentences must NEVER appear in training data.
- `test.jsonl` sentences must NEVER appear in training data.
- `dev_hard.jsonl` uses same-distribution but entirely different sentences.
- The build script includes an automated leakage check.
