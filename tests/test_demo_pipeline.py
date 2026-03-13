"""tests/test_demo_pipeline.py — Smoke tests for the end-to-end pipeline.

These tests run the resolver in stub/offline mode so no GPU or network is
required.  They verify that the pipeline produces a valid symbol, data dict,
and memo string without crashing.

Run with:
    python -m pytest tests/test_demo_pipeline.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.resolver import normalize_ticker


# ── normalize-only smoke tests (no model load) ────────────────────

def test_resolve_fb_returns_meta():
    """normalize_ticker("FB") must return META — core alias path."""
    symbol, normalized = normalize_ticker("FB")
    assert symbol == "META"
    assert normalized is True


def test_resolve_alphabet_class_a():
    """normalize_ticker("GOOGL") must return GOOGL without modification."""
    symbol, normalized = normalize_ticker("GOOGL")
    assert symbol == "GOOGL"


# ── pipeline smoke tests (model call mocked) ──────────────────────

def _make_mock_resolver(raw_output: str):
    """Return a SymbolResolver whose model inference is replaced by a stub."""
    from src.resolver import SymbolResolver

    resolver = SymbolResolver.__new__(SymbolResolver)
    resolver._model = MagicMock()
    resolver._tokenizer = None
    resolver._device = "cpu"

    # Patch resolve() to use only the normalize layer, bypassing model load
    def fake_resolve(query: str) -> dict:
        symbol, normalized = normalize_ticker(raw_output)
        return {"symbol": symbol, "raw_output": raw_output, "normalized": normalized}

    resolver.resolve = fake_resolve
    return resolver


def test_pipeline_fb_alias_end_to_end():
    """Full pipeline with FB input must produce META symbol and a memo."""
    from src.data_fetch import fetch_market_data
    from src.memo import generate_memo

    resolver = _make_mock_resolver("FB")
    result = resolver.resolve("Research FB for the past 3 months")

    assert result["symbol"] == "META", f"Expected META, got {result['symbol']!r}"
    assert result["normalized"] is True

    # data_fetch falls back to stub when yfinance is unavailable — always runs
    data = fetch_market_data(result["symbol"], months=3)
    assert isinstance(data, dict)
    assert "data_source" in data

    memo = generate_memo(result["symbol"], data, "Research FB for the past 3 months")
    assert isinstance(memo, str)
    assert len(memo) > 0


def test_pipeline_standard_case_end_to_end():
    """Full pipeline with AAPL must produce a memo without errors."""
    from src.data_fetch import fetch_market_data
    from src.memo import generate_memo

    resolver = _make_mock_resolver("AAPL")
    result = resolver.resolve("Analyze Apple over the last 1 year")

    assert result["symbol"] == "AAPL"

    data = fetch_market_data("AAPL", months=12)
    assert isinstance(data, dict)

    memo = generate_memo("AAPL", data, "Analyze Apple over the last 1 year")
    assert isinstance(memo, str)
    assert len(memo) > 0
