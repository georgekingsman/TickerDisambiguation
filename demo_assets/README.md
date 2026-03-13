# Demo Assets

Pre-captured outputs from the 3 annotated demo cases. You can review these without running any code or having a GPU.

To regenerate fresh outputs (with live yfinance data):
```bash
python app.py --batch
```

---

## Files

| File | Contents |
|---|---|
| `demo_case_1_input.txt` | Query for demo 1 |
| `demo_case_1_output.md` | Full output: resolver trace + market data + memo |
| `demo_case_2_input.txt` | Query for demo 2 |
| `demo_case_2_output.md` | Full output: resolver trace + normalize trace + market data + memo |
| `demo_case_3_input.txt` | Query for demo 3 |
| `demo_case_3_output.md` | Full output: resolver trace + market data + memo |

---

## Demo 1 — Google Share Class ([demo_case_1_output.md](demo_case_1_output.md))

**Query:** `Research Alphabet class A for me`  
**Expected symbol:** `GOOGL`

**What this proves:** The model correctly distinguishes `GOOGL` (Class A, voting) from `GOOG` (Class C, non-voting). LoRA v1 made 6 errors on this; v1.2 makes 0. This is the hardest disambiguation challenge in the benchmark.

---

## Demo 2 — Legacy Alias Normalization ([demo_case_2_output.md](demo_case_2_output.md))

**Query:** `Research FB for the past 3 months`  
**Expected symbol:** `META` (via `FB → META` normalize)

**What this proves:** The normalize layer converts stale tickers to current ones (same operation as `BRK.B → BRK-B`). Without it, the downstream data fetch would fail on a ticker that no longer trades.

---

## Demo 3 — Standard Stable Case ([demo_case_3_output.md](demo_case_3_output.md))

**Query:** `Analyze Apple over the last 1 year`  
**Expected symbol:** `AAPL`

**What this proves:** Routine queries resolve cleanly. Robustness improvements on hard cases (demos 1 & 2) did not degrade performance on standard inputs.

