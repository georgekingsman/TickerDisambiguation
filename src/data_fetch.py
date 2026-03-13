"""src/data_fetch.py — Market data fetcher with yfinance + stub fallback."""

from datetime import datetime, timedelta

from src.config import COMPANY_NAMES


def fetch_market_data(symbol: str, months: int = 6) -> dict:
    """Fetch price history and basic fundamentals for a symbol.

    Uses yfinance when available; falls back to a safe stub so the pipeline
    always produces output even without network access or yfinance installed.
    """
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
        return {
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
        }

    except Exception:
        # Graceful stub — always works, for demo/offline purposes
        return {
            "symbol": symbol,
            "company": COMPANY_NAMES.get(symbol, symbol),
            "period": f"{months} months",
            "current_price": "N/A (yfinance not installed)",
            "data_source": "stub",
            "note": "Install yfinance for live data: pip install yfinance",
        }
