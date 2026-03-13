## Demo Case 3: Standard stable case

**Input query:** `Analyze Apple over the last 1 year`  
**Raw resolver output:** `AAPL`  
**Resolved symbol:** `AAPL`  
**Normalized:** No — direct model match  

### What this demonstrates

Unambiguous queries resolve cleanly and immediately. This case verifies that robustness work on hard edge cases (demos 1 & 2) did not degrade performance on routine queries. The system handles the full spectrum from trivial to highly ambiguous inputs.

### Market Data

| Field | Value |
|---|---|
| Company | Apple Inc. |
| Period | 12 months |
| Current Price | $213.49 |
| Period Return | ▲ 25.49% |
| Period High | $260.10 |
| Period Low | $164.08 |
| Avg Volume | 62,341,205 |
| Sector | Technology |

> Prices captured at demo time. Re-run `python app.py --query "Analyze Apple over the last 1 year" --months 12` for live data.

### Research Memo

```
============================================================
  RESEARCH MEMO: Apple Inc. (AAPL)
============================================================
  Query:   Analyze Apple over the last 1 year
  Period:  12 months

  Current Price:   $213.49
  Period Start:    $170.12
  Period Return:   ▲ 25.49%
  Period High:     $260.10
  Period Low:      $164.08
  Avg Volume:      62,341,205
  Market Cap:      3210000000000
  P/E Ratio:       33.1
  Sector:          Technology

============================================================
```
