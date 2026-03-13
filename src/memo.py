"""src/memo.py — Research memo generator and flywheel logger."""

import json
from datetime import datetime

from src.config import FLYWHEEL_LOG


def generate_memo(symbol: str, data: dict, query: str) -> str:
    """Generate a structured research memo from resolved symbol + market data.

    Numeric data points are annotated with citation markers [1] / [2] that
    reference the MCP tool calls listed in the 'Data Sources' footer section.
    This satisfies the RA requirement for traceable data provenance.
    """
    company = data.get("company", symbol)
    period = data.get("period", "N/A")
    tool_calls: list = data.get("tool_calls", [])

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
            f"  Current Price:   ${data['current_price']}  [1]",
            f"  Period Start:    ${data['period_start_price']}  [1]",
            f"  Period Return:   {direction} {abs(ret):.2f}%  [1]",
            f"  Period High:     ${data['period_high']}  [1]",
            f"  Period Low:      ${data['period_low']}  [1]",
            f"  Avg Volume:      {data['avg_volume']:,}  [1]",
            f"  Market Cap:      {data.get('market_cap', 'N/A')}  [2]",
            f"  P/E Ratio:       {data.get('pe_ratio', 'N/A')}  [2]",
            f"  Sector:          {data.get('sector', 'N/A')}  [2]",
        ]
    else:
        lines += [
            "  [Data source: stub — install yfinance for live data]",
        ]

    # ── Citation markers footer ───────────────────────────────────
    if tool_calls:
        lines += [
            "",
            "  Data Sources",
            "  " + "─" * 50,
        ]
        for i, call in enumerate(tool_calls, 1):
            if "get_price_history" in call:
                note = "→ price history · period return · high · low · volume"
            else:
                note = "→ market cap · P/E ratio · sector"
            lines += [f"  [{i}] {call}", f"      {note}"]

    lines += ["", f"{'='*60}"]
    return "\n".join(lines)


def log_flywheel(record: dict):
    """Append a structured JSONL record for future training flywheel use."""
    record["timestamp"] = datetime.now().isoformat()
    with open(FLYWHEEL_LOG, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
