"""src/memo.py — Research memo generator and flywheel logger."""

import json
from datetime import datetime

from src.config import FLYWHEEL_LOG


def generate_memo(symbol: str, data: dict, query: str) -> str:
    """Generate a structured research memo from resolved symbol + market data."""
    company = data.get("company", symbol)
    period = data.get("period", "N/A")

    lines = [
        f"{'='*60}",
        f"  RESEARCH MEMO: {company} ({symbol})",
        f"{'='*60}",
        f"  Query:   {query}",
        f"  Period:  {period}",
        "",
    ]

    if data.get("data_source") == "yfinance":
        ret = data["period_return"]
        direction = "▲" if ret >= 0 else "▼"
        lines += [
            f"  Current Price:   ${data['current_price']}",
            f"  Period Start:    ${data['period_start_price']}",
            f"  Period Return:   {direction} {abs(ret):.2f}%",
            f"  Period High:     ${data['period_high']}",
            f"  Period Low:      ${data['period_low']}",
            f"  Avg Volume:      {data['avg_volume']:,}",
            f"  Market Cap:      {data.get('market_cap', 'N/A')}",
            f"  P/E Ratio:       {data.get('pe_ratio', 'N/A')}",
            f"  Sector:          {data.get('sector', 'N/A')}",
        ]
    else:
        lines += [
            f"  [Data source: stub — install yfinance for live data]",
        ]

    lines += ["", f"{'='*60}"]
    return "\n".join(lines)


def log_flywheel(record: dict):
    """Append a structured JSONL record for future training flywheel use."""
    record["timestamp"] = datetime.now().isoformat()
    with open(FLYWHEEL_LOG, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
