# Zero-Shot Baseline Results | Zero-Shot 基线结果

> **Status 状态**: Pending — run experiments and fill in tables below.
> 待完成——运行实验后填写以下表格。

---

## Experiment Setup | 实验设置

- **Base Model 基础模型**: `Qwen/Qwen2.5-0.5B-Instruct` (or chosen model)
- **Decoding 解码**: `temperature=0`, `do_sample=False`, `max_new_tokens=8`
- **Post-processing 后处理**: Uppercase, BRK.B→BRK-B normalization, first-valid-ticker extraction

### Prompt Variants | 提示词变体

| Variant 变体 | File 文件 | Description 描述 |
|---|---|---|
| **Plain** | `prompts/zero_shot_plain.txt` | Task-only, no disambiguation rules 仅任务描述，无消歧规则 |
| **Policy-aware** | `prompts/zero_shot_prompt.txt` | Includes explicit class/alias rules 包含显式股类/别名规则 |

---

## Comparison Matrix | 对比矩阵

| Model 模型 | Prompt 提示词 | Test Acc 测试准确率 | Test Macro-F1 | Ambiguity Acc 歧义准确率 | Hard Acc 困难集准确率 |
|---|---|---:|---:|---:|---:|
| Rule baseline 规则基线 | N/A | 96.55% | 0.914 | 95.00% | TBD |
| Base model 基础模型 | Plain | TBD | TBD | TBD | TBD |
| Base model 基础模型 | Policy-aware | TBD | TBD | TBD | TBD |
| LoRA fine-tuned (future) | Plain | TBD | TBD | TBD | TBD |

---

## How to Reproduce | 如何复现

### 1. Run zero-shot inference | 运行 zero-shot 推理

```bash
# Plain prompt — test set
python scripts/run_zero_shot.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --gold data/test.jsonl \
    --prompt prompts/zero_shot_plain.txt \
    --output data/test_zeroshot_plain_preds.jsonl

# Plain prompt — ambiguous eval set
python scripts/run_zero_shot.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --gold data/ambiguous_eval.jsonl \
    --prompt prompts/zero_shot_plain.txt \
    --output data/ambiguous_eval_zeroshot_plain_preds.jsonl

# Plain prompt — hard eval set
python scripts/run_zero_shot.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --gold data/hard_eval.jsonl \
    --prompt prompts/zero_shot_plain.txt \
    --output data/hard_eval_zeroshot_plain_preds.jsonl

# Policy-aware prompt — test set
python scripts/run_zero_shot.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --gold data/test.jsonl \
    --prompt prompts/zero_shot_prompt.txt \
    --output data/test_zeroshot_policy_preds.jsonl

# Policy-aware prompt — ambiguous eval set
python scripts/run_zero_shot.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --gold data/ambiguous_eval.jsonl \
    --prompt prompts/zero_shot_prompt.txt \
    --output data/ambiguous_eval_zeroshot_policy_preds.jsonl

# Policy-aware prompt — hard eval set
python scripts/run_zero_shot.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --gold data/hard_eval.jsonl \
    --prompt prompts/zero_shot_prompt.txt \
    --output data/hard_eval_zeroshot_policy_preds.jsonl
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

### Error Taxonomy | 错误分类

Errors are categorized into 5 types (fill in after running experiments):
错误按以下 5 类整理（实验后填写）：

### 1. Class Confusion | 股类混淆
Examples: `GOOG` ↔ `GOOGL`, `BRK-A` ↔ `BRK-B`

| Input | Gold | Predicted | Prompt |
|---|---|---|---|
| TBD | | | |

### 2. Alias Failure | 别名失败
Examples: "Bezos company", "old Facebook stock"

| Input | Gold | Predicted | Prompt |
|---|---|---|---|
| TBD | | | |

### 3. Format Error | 格式错误
Examples: output `BRK.B` instead of `BRK-B`

| Input | Gold | Predicted | Prompt |
|---|---|---|---|
| TBD | | | |

### 4. Hallucination | 幻觉
Examples: output a ticker not in the 20-symbol universe

| Input | Gold | Predicted | Prompt |
|---|---|---|---|
| TBD | | | |

### 5. Refusal / Empty / Verbose Output | 拒答 / 空输出 / 废话输出
Examples: "I'm not sure", "The ticker is GOOG"

| Input | Gold | Predicted | Prompt |
|---|---|---|---|
| TBD | | | |

---

## Key Observations | 关键发现

> Fill in after running experiments.
> 实验后填写。

1. **Plain vs Policy-aware gap**: TBD
2. **Where the model struggles most**: TBD
3. **Why LoRA is justified**: TBD
