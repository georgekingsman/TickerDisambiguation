"""src/data_fetch.py — Market data fetcher with yfinance + stub fallback.

Supports:
  - budget_limit : max number of live yfinance calls per session (0 = unlimited)
  - response caching : repeated calls for the same (symbol, months) pair are
    served from _CACHE without consuming budget or hitting the network
  - tool_calls list : each result contains the MCP-style call signatures used,
    so generate_memo() can emit citation markers [1]/[2] in the output memo
"""

from datetime import datetime, timedelta

from src.config import COMPANY_NAMES

# ── In-memory response cache and session budget counter ──────────
_CACHE: dict = {}
_CALL_COUNTER: list = [0]  # mutable list so nested functions can mutate it


def reset_budget() -> None:
    """Reset the MCP call counter. Call once at the start of each request."""
    _CALL_COUNTER[0] = 0


def fetch_market_data(symbol: str, months: int = 6, budget_limit: int = 10) -> dict:
    """Fetch price history and basic fundamentals for a symbol.

    Parameters
    ----------
    symbol       : canonical ticker (e.g. 'GOOGL', 'BRK-B')
    months       : lookback window in months
    budget_limit : max live yfinance calls allowed this session; 0 = unlimited

    Returns a dict that always includes:
      tool_calls  — list of MCP-style call signatures e.g.
                    ['get_price_history(GOOGL, ...)', 'get_ticker_info(GOOGL)']
      cache_hit   — True if result was served from _CACHE
    """
    cache_key = (symbol, months)

    # ── Cache hit — skip budget check and network call ────────────
    if cache_key in _CACHE:
        cached = dict(_CACHE[cache_key])
        cached["cache_hit"] = True
        return cached

    # ── Budget guard ──────────────────────────────────────────────
    if budget_limit > 0 and _CALL_COUNTER[0] >= budget_limit:
        return {
            "symbol": symbol,
            "company": COMPANY_NAMES.get(symbol, symbol),
            "period": f"{months} months",
            "current_price": "N/A (budget limit reached)",
            "data_source": "stub",
            "tool_calls": [],
            "cache_hit": False,
            "note": f"MCP budget limit of {budget_limit} calls reached; cached responses only.",
        }

    yf_symbol = symbol.replace("-", ".")  # BRK-B → BRK.B for yfinance

    try:
        import yfinance as yf  # type: ignore[import-untyped]

        ticker = yf.Ticker(yf_symbol)
        end = datetime.now()
        start = end - timedelta(days=months * 30)
        hist = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )

        if hist.empty:
            raise ValueError("No data returned")

        info = ticker.info
        _CALL_COUNTER[0] += 1

        tool_calls = [
            f"get_price_history({symbol}, start={start.strftime('%Y-%m-%d')}, end={end.strftime('%Y-%m-%d')})",
            f"get_ticker_info({symbol})",
        ]

        result = {
            "symbol": symbol,
            "company": info.get("longName", COMPANY_NAMES.get(symbol, symbol)),
            "period": f"{months} months",
            "current_price": round(hist["Close"].iloc[-1], 2),
            "period_start_price": round(hist["Close"].iloc[0], 2),
            "period_return": round(
                (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100, 2
            ),
            "period_high": round(hist["High"].max(), 2),
            "period_low": round(hist["Low"].min(), 2),
            "avg_volume": int(hist["Volume"].mean()),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "sector": info.get("sector", "N/A"),
            "data_source": "yfinance",
            "tool_calls": tool_calls,
            "cache_hit": False,
        }
        _CACHE[cache_key] = result
        return result

    except Exception:
        # Graceful stub — always works, for demo/offline purposes
        return {
            "symbol": symbol,
            "company": COMPANY_NAMES.get(symbol, symbol),
            "period": f"{months} months",
            "current_price": "N/A (yfinance not installed)",
            "data_source": "stub",
            "tool_calls": [],
            "cache_hit": False,
            "note": "Install yfinance for live data: pip install yfinance",
        }
