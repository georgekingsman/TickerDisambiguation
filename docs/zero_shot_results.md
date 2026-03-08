# Zero-Shot Baseline Results | Zero-Shot 基线结果

> **Status 状态**: ✅ Complete — experiments run on 2026-03-08.
> 已完成——实验于 2026-03-08 运行。

---

## Experiment Setup | 实验设置

- **Base Model 基础模型**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Decoding 解码**: `temperature=0`, `do_sample=False`, `max_new_tokens=8`
- **Post-processing 后处理**: Uppercase, BRK.B→BRK-B normalization, first-valid-ticker extraction
- **Hardware 硬件**: Apple Silicon (MPS), float32

### Prompt Variants | 提示词变体

| Variant 变体 | File 文件 | Description 描述 |
|---|---|---|
| **Plain** | `prompts/zero_shot_plain.txt` | Task-only, no disambiguation rules 仅任务描述，无消歧规则 |
| **Policy-aware** | `prompts/zero_shot_prompt.txt` | Includes explicit class/alias rules 包含显式股类/别名规则 |

---

## Comparison Matrix | 对比矩阵

| Model 模型 | Prompt 提示词 | Test Acc (29) | Test Macro-F1 | Ambiguity Acc (20) | Hard Acc (30) |
|---|---|---:|---:|---:|---:|
| Rule baseline 规则基线 | N/A | **96.55%** | **0.914** | **95.00%** | 30.00% |
| Qwen2.5-0.5B-Instruct | Plain | 51.72% | 0.447 | 40.00% | 13.33% |
| Qwen2.5-0.5B-Instruct | Policy-aware | 75.86% | 0.757 | 70.00% | 23.33% |
| LoRA fine-tuned (future) | TBD | TBD | TBD | TBD | TBD |

### Key Takeaways from the Matrix | 矩阵关键发现

1. **Rule baseline dominates on test & ambiguity** — but only because those sets align with its hardcoded rules.
2. **Policy-aware prompt gives +24pp on test** over plain (75.86% vs 51.72%) — showing the model *can* follow rules when told.
3. **All methods collapse on hard_eval** — rule baseline (30%), plain (13.33%), policy-aware (23.33%). This is where LoRA must make the difference.
4. **The model has learnable error structure** — class confusion and alias failures are systematic, not random.

---

## How to Reproduce | 如何复现

### 1. Run zero-shot inference | 运行 zero-shot 推理

```bash
export KMP_DUPLICATE_LIB_OK=TRUE

# Plain prompt — all 3 datasets
python scripts/run_zero_shot.py --model Qwen/Qwen2.5-0.5B-Instruct --gold data/test.jsonl --prompt prompts/zero_shot_plain.txt --output data/test_zeroshot_plain_preds.jsonl
python scripts/run_zero_shot.py --model Qwen/Qwen2.5-0.5B-Instruct --gold data/ambiguous_eval.jsonl --prompt prompts/zero_shot_plain.txt --output data/ambiguous_eval_zeroshot_plain_preds.jsonl
python scripts/run_zero_shot.py --model Qwen/Qwen2.5-0.5B-Instruct --gold data/hard_eval.jsonl --prompt prompts/zero_shot_plain.txt --output data/hard_eval_zeroshot_plain_preds.jsonl

# Policy-aware prompt — all 3 datasets
python scripts/run_zero_shot.py --model Qwen/Qwen2.5-0.5B-Instruct --gold data/test.jsonl --prompt prompts/zero_shot_prompt.txt --output data/test_zeroshot_policy_preds.jsonl
python scripts/run_zero_shot.py --model Qwen/Qwen2.5-0.5B-Instruct --gold data/ambiguous_eval.jsonl --prompt prompts/zero_shot_prompt.txt --output data/ambiguous_eval_zeroshot_policy_preds.jsonl
python scripts/run_zero_shot.py --model Qwen/Qwen2.5-0.5B-Instruct --gold data/hard_eval.jsonl --prompt prompts/zero_shot_prompt.txt --output data/hard_eval_zeroshot_policy_preds.jsonl
```

### 2. Evaluate | 评估

```bash
python scripts/evaluate.py --predictions data/test_zeroshot_plain_preds.jsonl --gold data/test.jsonl
python scripts/evaluate.py --predictions data/ambiguous_eval_zeroshot_plain_preds.jsonl --gold data/ambiguous_eval.jsonl
python scripts/evaluate.py --predictions data/hard_eval_zeroshot_plain_preds.jsonl --gold data/hard_eval.jsonl

python scripts/evaluate.py --predictions data/test_zeroshot_policy_preds.jsonl --gold data/test.jsonl
python scripts/evaluate.py --predictions data/ambiguous_eval_zeroshot_policy_preds.jsonl --gold data/ambiguous_eval.jsonl
python scripts/evaluate.py --predictions data/hard_eval_zeroshot_policy_preds.jsonl --gold data/hard_eval.jsonl
```

---

## Error Analysis | 错误分析

Total errors across all 6 runs: **Plain: 52 errors**, **Policy-aware: 36 errors**

### 1. Class Confusion | 股类混淆

The most systematic error. The model confuses `GOOG` ↔ `GOOGL` in both directions, and `BRK-A` ↔ `BRK-B`.

| Input | Gold | Predicted | Prompt |
|---|---|---|---|
| "Give me a report on Google's class A shares." | GOOGL | GOOG | Policy |
| "Research Alphabet class A for me." | GOOGL | GOOG | Policy |
| "Summarize Google class C for me." | GOOG | GOOGL | Plain |
| "What's the performance of Alphabet class C?" | GOOG | GOOGL | Plain |
| "Quick analysis of GOOG please." | GOOG | GOOGL | Both |
| "Research Google's non-voting stock." | GOOG | GOOGL | Both |
| "Research the cheaper Google share class." | GOOG | GOOGL | Both |
| "Analyze the cheaper Google share class." | GOOG | GOOGL | Both |
| "Pull up the non-voting Google shares." | GOOG | GOOGL | Both |
| "Research the expensive Berkshire shares." | BRK-A | BRK-B | Plain |
| "Show me the premium Berkshire class." | BRK-A | BRK-B | Plain |

**Count**: ~15 instances across runs. This is the #1 error class.

**Key insight**: Even with policy-aware prompt, the model *reverses* GOOGL→GOOG for "class A" inputs — suggesting it learned a spurious association between "class" keyword and GOOG.

### 2. Alias Failure | 别名失败

The model cannot resolve colloquial / indirect references to companies.

| Input | Gold | Predicted | Prompt |
|---|---|---|---|
| "Research the Bezos company." | AMZN | UNKNOWN | Both |
| "Give me the old Facebook stock." | META | UNKNOWN | Both |
| "How is the Zuckerberg social media empire performing?" | META | UNKNOWN | Both |
| "What's the deal with Elon's EV company?" | TSLA | UNKNOWN | Both |
| "How is the Mouse House doing lately?" | DIS | UNKNOWN | Both |
| "Research Jensen's chip company." | NVDA | UNKNOWN | Both |
| "How is Satya's cloud company doing?" | MSFT | UNKNOWN/META | Both |
| "Research Lisa Su's chip company." | AMD | UNKNOWN/AAPL | Both |
| "Research the AWS parent company." | AMZN | UNKNOWN | Plain |
| "What's happening with the iPhone maker?" | AAPL | UNKNOWN | Plain |
| "Research the company that runs Instagram and WhatsApp." | META | UNKNOWN | Both |

**Count**: ~20+ instances. This is the largest error bucket on hard_eval.

### 3. Format Error | 格式错误

Post-processing normalization handles most format errors (BRK.B → BRK-B). Remaining:

| Input | Gold | Predicted (raw) | Prompt |
|---|---|---|---|
| "Give me a report on Facebook." | META | `FB.` (→ UNKNOWN) | Both |
| "Give me the old Facebook stock." | META | `FB` (→ UNKNOWN) | Policy |

**Count**: 2-3 instances. The model outputs `FB` which is not in our ticker set (it's `META`). This is actually an alias failure disguised as a format issue.

### 4. Hallucination | 幻觉

The model invented tickers not in the 20-symbol universe:

| Input | Gold | Raw Output | Prompt |
|---|---|---|---|
| "Give me a memo on Chase bank stock." | JPM | `CHX` | Plain |
| "Research the Chinese e-commerce giant behind Taobao." | BABA | `TAIWANSEMICOMMOD` | Plain |
| "Research the Chinese e-commerce giant behind Taobao." | BABA | `TACO` | Policy |
| "Show me data for the Temu parent." | PDD | `TMUS.001.0` | Policy |
| "How is the biggest semiconductor foundry performing?" | TSM | `900570.S` | Policy |
| "How is Satya's cloud company doing?" | MSFT | → META (wrong company) | Policy |
| "Pull up Buffett's six-figure stock." | BRK-A | → META | Policy |

**Count**: ~8 instances. The model generates plausible-looking but wrong symbols.

### 5. Refusal / Empty / Verbose Output | 拒答 / 空输出 / 废话输出

The model frequently outputs verbose refusals (especially with plain prompt):

| Input | Gold | Raw Output (truncated) | Prompt |
|---|---|---|---|
| "How is Netflix stock doing?" | NFLX | `Netflix\nticker-symbol: Netflix\n...` | Plain |
| "Pull up JPM financials." | JPM | `None of the provided options are...` | Plain |
| "How is AMD stock doing?" | AMD | `The answer depends on the specific...` | Plain |
| "How has BofA been performing?" | BAC | `None of the listed tickers are...` | Plain |
| "How's the world's biggest online retailer doing?" | AMZN | `None of the options provided are...` | Plain |

**Count**: ~12 instances (mostly plain prompt). The model often says "None of the provided options" — it seems to hallucinate a multiple-choice context that doesn't exist.

---

## Key Observations | 关键发现

1. **Plain vs Policy-aware gap**: Policy-aware prompt adds **+24pp on test** (51.72% → 75.86%) and **+30pp on ambiguity** (40% → 70%). This proves the model *can* follow explicit rules but lacks internalized knowledge for this task.

2. **Where the model struggles most**: 
   - **Class confusion** (GOOG↔GOOGL) is the most systematic error — it persists even with policy-aware prompt
   - **Colloquial aliases** ("Bezos company", "Mouse House") fail universally across all methods
   - **Verbose/refusal outputs** are common with plain prompt (the model doesn't understand it should output only a ticker)

3. **Why LoRA is justified**: 
   - The rule baseline is brittle: 96.55% on test but **30%** on hard_eval — it can't generalize
   - The zero-shot model has learnable error structure: class confusion and alias failures are systematic patterns that LoRA can learn to fix
   - Neither rules nor prompting can solve the full problem — the model needs to internalize company knowledge and output discipline
   - **The gap between policy-aware (75.86%) and rule baseline (96.55%) on test shows there's 20pp of "easy" improvements waiting for LoRA**
   - **The hard_eval collapse (13-30% across all methods) is the strongest argument for fine-tuning**

4. **Answer to "Why not just use rules?"**:
   - Rules score 96.55% on test but **30% on hard_eval** — they only work within their hardcoded patterns
   - The zero-shot model at least gets iPhone→AAPL right where rules fail, showing models can generalize
   - Real user requests look more like hard_eval than test — rules are not production-viable
