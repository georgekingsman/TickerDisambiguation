## Demo Case 1: Google share class disambiguation

**Input query:** `Research Alphabet class A for me`  
**Raw resolver output:** `GOOGL`  
**Resolved symbol:** `GOOGL`  
**Normalized:** No — direct model match  

### What this demonstrates

The model correctly returns `GOOGL` (Class A, voting shares) rather than `GOOG` (Class C, non-voting). This was the hardest disambiguation challenge in the benchmark:

| Version | GOOG↔GOOGL errors |
|---|---:|
| LoRA v1 | 6 |
| LoRA v1.1 + normalize | 3 |
| **LoRA v1.2 + normalize** | **0** |

Alphabet has two tickers because it issued non-voting Class C shares (`GOOG`) in 2014 while retaining the original Class A (`GOOGL`). Without explicit training signal, base LLMs conflate the two.

### Market Data

| Field | Value |
|---|---|
| Company | Alphabet Inc. (Class A) |
| Period | 6 months |
| Current Price | $174.32 |
| Period Return | ▲ 8.15% |
| Period High | $208.70 |
| Period Low | $155.49 |
| Avg Volume | 24,187,342 |
| Sector | Communication Services |

> Prices captured at demo time. Re-run `python app.py --query "Research Alphabet class A for me"` for live data.

### Research Memo

```
============================================================
  RESEARCH MEMO: Alphabet Inc. (Class A) (GOOGL)
============================================================
  Query:   Research Alphabet class A for me
  Period:  6 months

  Current Price:   $174.32
  Period Start:    $161.18
  Period Return:   ▲ 8.15%
  Period High:     $208.70
  Period Low:      $155.49
  Avg Volume:      24,187,342
  Market Cap:      2138000000000
  P/E Ratio:       21.8
  Sector:          Communication Services

============================================================
```
