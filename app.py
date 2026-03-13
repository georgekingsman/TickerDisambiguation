"""
app.py  —  Investment Research Copilot
          Query → Resolve → Market Data → Research Memo

Usage:
    python app.py --demo                              # 3 annotated demo cases
    python app.py --query "Research Apple"            # single query
    python app.py --batch                             # save outputs to demo_assets/
    python app.py --batch path/to/cases.jsonl         # batch from custom JSONL
    python app.py                                     # interactive mode
"""

import argparse
import json
from pathlib import Path

from src.config import FINAL_MODEL, FINAL_ADAPTER, FLYWHEEL_LOG
from src.resolver import SymbolResolver
from src.data_fetch import fetch_market_data
from src.memo import generate_memo, log_flywheel

DEMO_ASSETS_DIR = Path("demo_assets")

# ── Demo Cases ────────────────────────────────────────────────────

DEMO_CASES = [
    {
        "query": "Research Alphabet class A for me",
        "months": 6,
        "story": "Google share class — model must distinguish GOOGL (A) from GOOG (C); "
                 "confusion fully eliminated in v1.2 (was 6 errors → 0)",
    },
    {
        "query": "Research FB for the past 3 months",
        "months": 3,
        "story": "Meta alias — model outputs 'FB', normalize layer canonicalizes to META",
    },
    {
        "query": "Analyze Apple over the last 1 year",
        "months": 12,
        "story": "Standard case — proves system handles routine queries reliably",
    },
]


# ── Pipeline ─────────────────────────────────────────────────────

def run_pipeline(resolver: SymbolResolver, query: str, months: int = 6) -> dict:
    """Full pipeline: query → resolve → fetch → memo → log."""
    print(f"\n{'─'*60}")
    print(f"  INPUT: {query}")
    print(f"{'─'*60}")

    result = resolver.resolve(query)
    symbol = result["symbol"]
    raw = result["raw_output"]
    norm_flag = " (normalized)" if result["normalized"] else ""
    print(f"  ✓ Resolver:  {symbol}  (raw: {raw!r}){norm_flag}")

    if symbol == "UNKNOWN":
        print("  ✗ Could not resolve symbol. Skipping data fetch.")
        return result

    data = fetch_market_data(symbol, months=months)
    print(f"  ✓ Data:      {data.get('data_source', 'unknown')} source")

    memo = generate_memo(symbol, data, query)
    print(f"\n{memo}")

    log_flywheel({
        "query": query,
        "raw_output": raw,
        "resolved_symbol": symbol,
        "normalized": result["normalized"],
        "data_source": data.get("data_source"),
        "model": f"{FINAL_MODEL}+LoRA({FINAL_ADAPTER})",
    })
    print(f"  ✓ Logged to {FLYWHEEL_LOG}")

    return {**result, "data": data, "memo": memo}


def run_demo(resolver: SymbolResolver):
    """Run all 3 annotated demo cases."""
    print("\n" + "═" * 60)
    print("  INVESTMENT RESEARCH COPILOT — DEMO")
    print("  LoRA v1.2 + normalize(FB→META, BRK.B→BRK-B)")
    print("═" * 60)
    print(f"  Base model: {FINAL_MODEL}")
    print(f"  Adapter:    {FINAL_ADAPTER}")

    for i, case in enumerate(DEMO_CASES, 1):
        print(f"\n\n{'▓'*60}")
        print(f"  DEMO {i}/3: {case['story']}")
        print(f"{'▓'*60}")
        run_pipeline(resolver, case["query"], months=case["months"])

    print(f"\n\n{'═'*60}")
    print(f"  ✓ Demo complete — {len(DEMO_CASES)} cases processed")
    print(f"  Flywheel log: {FLYWHEEL_LOG}")
    print(f"{'═'*60}\n")


# ── Batch Mode ────────────────────────────────────────────────────

def run_batch(resolver: SymbolResolver, input_file: str | None = None):
    """Run pipeline on cases and save formatted input/output to demo_assets/.

    With no input_file, uses the built-in DEMO_CASES.
    With a JSONL input_file, reads records with 'query' and optional 'months'/'story'.
    """
    DEMO_ASSETS_DIR.mkdir(exist_ok=True)

    if input_file:
        with open(input_file) as f:
            cases = [json.loads(line) for line in f if line.strip()]
    else:
        cases = DEMO_CASES

    print(f"\n  Batch mode: saving {len(cases)} cases to {DEMO_ASSETS_DIR}/")

    for i, case in enumerate(cases, 1):
        query = case["query"] if isinstance(case, dict) else case
        months = case.get("months", 6) if isinstance(case, dict) else 6
        story = case.get("story", f"Case {i}") if isinstance(case, dict) else f"Case {i}"

        result = run_pipeline(resolver, query, months=months)
        symbol = result.get("symbol", "UNKNOWN")
        raw = result.get("raw_output", "")
        normalized = result.get("normalized", False)
        data = result.get("data", {})
        memo = result.get("memo", "")

        # Input file
        input_path = DEMO_ASSETS_DIR / f"demo_case_{i}_input.txt"
        input_path.write_text(query + "\n")

        # Output markdown
        md = [
            f"## Demo Case {i}: {story}",
            "",
            f"**Input query:** `{query}`  ",
            f"**Raw resolver output:** `{raw}`  ",
            f"**Resolved symbol:** `{symbol}`  ",
            f"**Normalized:** {'Yes — alias/format fix applied' if normalized else 'No — direct model match'}  ",
            "",
        ]

        if data.get("data_source") == "yfinance":
            ret = data.get("period_return", "N/A")
            direction = "▲" if isinstance(ret, (int, float)) and ret >= 0 else "▼"
            md += [
                "### Market Data",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| Company | {data.get('company', symbol)} |",
                f"| Period | {data.get('period', 'N/A')} |",
                f"| Current Price | ${data.get('current_price', 'N/A')} |",
                f"| Period Return | {direction} {abs(ret) if isinstance(ret, (int, float)) else ret}% |",
                f"| Period High | ${data.get('period_high', 'N/A')} |",
                f"| Period Low | ${data.get('period_low', 'N/A')} |",
                f"| Sector | {data.get('sector', 'N/A')} |",
                "",
            ]
        elif data:
            md += ["*Data source: stub (install yfinance for live data)*", ""]

        if memo:
            md += ["### Research Memo", "", "```", memo, "```", ""]

        output_path = DEMO_ASSETS_DIR / f"demo_case_{i}_output.md"
        output_path.write_text("\n".join(md) + "\n")
        print(f"  ✓ Saved: {input_path}  +  {output_path}")

    print(f"\n  ✓ Batch complete — {len(cases)} cases written to {DEMO_ASSETS_DIR}/\n")


# ── Entry Point ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Investment Research Copilot — Ticker Disambiguation"
    )
    parser.add_argument("--query", type=str, help="Single query to resolve")
    parser.add_argument("--months", type=int, default=6, help="Lookback months")
    parser.add_argument("--demo", action="store_true", help="Run 3 annotated demo cases")
    parser.add_argument(
        "--batch", nargs="?", const="", metavar="FILE",
        help="Save demo outputs to demo_assets/; pass a JSONL file to use custom cases",
    )
    args = parser.parse_args()

    resolver = SymbolResolver()

    if args.demo:
        run_demo(resolver)
    elif args.batch is not None:
        run_batch(resolver, input_file=args.batch if args.batch else None)
    elif args.query:
        run_pipeline(resolver, args.query, months=args.months)
    else:
        print("\n  Investment Research Copilot")
        print("  Type a query, or 'quit' to exit.\n")
        resolver.load()
        while True:
            try:
                query = input("  Query > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!")
                break
            if not query or query.lower() in ("quit", "exit", "q"):
                print("  Goodbye!")
                break
            run_pipeline(resolver, query)


if __name__ == "__main__":
    main()
