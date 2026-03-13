"""tests/test_normalize.py — Smoke tests for the normalize_ticker layer.

Run with:
    python -m pytest tests/test_normalize.py -v
"""

import sys
from pathlib import Path

# Ensure the project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.resolver import normalize_ticker


def test_fb_alias_resolves_to_meta():
    """FB is a known alias; must map to META with normalized=True."""
    symbol, was_normalized = normalize_ticker("FB")
    assert symbol == "META", f"Expected META, got {symbol!r}"
    assert was_normalized is True


def test_brk_dot_b_resolves_to_brk_dash_b():
    """BRK.B (yfinance format) must canonicalize to BRK-B (internal format)."""
    symbol, was_normalized = normalize_ticker("BRK.B")
    assert symbol == "BRK-B", f"Expected BRK-B, got {symbol!r}"
    assert was_normalized is True


def test_clean_ticker_passthrough():
    """A valid ticker already in canonical form should pass through unchanged."""
    symbol, was_normalized = normalize_ticker("AAPL")
    assert symbol == "AAPL", f"Expected AAPL, got {symbol!r}"
    assert was_normalized is False


def test_googl_not_confused_with_goog():
    """GOOGL and GOOG are both valid tickers; each must resolve to itself."""
    sym_l, _ = normalize_ticker("GOOGL")
    sym_c, _ = normalize_ticker("GOOG")
    assert sym_l == "GOOGL", f"Expected GOOGL, got {sym_l!r}"
    assert sym_c == "GOOG", f"Expected GOOG, got {sym_c!r}"


def test_unknown_ticker_returns_unknown():
    """Unrecognized input must return UNKNOWN, not crash."""
    symbol, _ = normalize_ticker("XYZZY_GARBAGE_123")
    assert symbol == "UNKNOWN", f"Expected UNKNOWN, got {symbol!r}"
