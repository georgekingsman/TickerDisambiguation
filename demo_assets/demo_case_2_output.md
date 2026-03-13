## Demo Case 2: Legacy alias normalization (FB → META)

**Input query:** `Research FB for the past 3 months`  
**Raw resolver output:** `FB`  
**Resolved symbol:** `META`  
**Normalized:** Yes — alias/format fix applied  

### What this demonstrates

The LoRA model outputs `FB` (the pre-2021 ticker it learned from training data). The normalize layer maps `FB → META` — the same class of operation as `BRK.B → BRK-B`. This is canonicalization, not cheating: the user asked about the company, not a specific historical ticker.

### Normalize trace

```
raw model output  →  "FB"
TICKER_ALIASES lookup  →  FB → META
final symbol  →  META
normalized flag  →  True
```

This prevents the downstream data fetch from failing on a stale ticker that no longer trades.

### Market Data

| Field | Value |
|---|---|
| Company | Meta Platforms Inc. |
| Period | 3 months |
| Current Price | $589.14 |
| Period Return | ▲ 4.97% |
| Period High | $740.91 |
| Period Low | $521.10 |
| Avg Volume | 17,654,891 |
| Sector | Communication Services |

> Prices captured at demo time. Re-run `python app.py --query "Research FB for the past 3 months" --months 3` for live data.

### Research Memo

```
============================================================
  RESEARCH MEMO: Meta Platforms Inc. (META)
============================================================
  Query:   Research FB for the past 3 months
  Period:  3 months

  Current Price:   $589.14  [1]
  Period Start:    $561.22  [1]
  Period Return:   ▲ 4.97%  [1]
  Period High:     $740.91  [1]
  Period Low:      $521.10  [1]
  Avg Volume:      17,654,891  [1]
  Market Cap:      1489000000000  [2]
  P/E Ratio:       26.4  [2]
  Sector:          Communication Services  [2]

  Data Sources
  ──────────────────────────────────────────────────
  [1] get_price_history(META, start=2025-12-13, end=2026-03-13)
      → price history · period return · high · low · volume
  [2] get_ticker_info(META)
      → market cap · P/E ratio · sector

============================================================
```
