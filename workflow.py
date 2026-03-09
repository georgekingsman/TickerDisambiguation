"""
workflow.py  –  Minimal Research Copilot Workflow
               最小研究 Copilot 工作流

End-to-end pipeline: user query → symbol resolution → market data → memo.
端到端流水线：用户请求 → 代码解析 → 市场数据 → 研究备忘录。

Usage / 用法:
    # Interactive mode | 交互模式
    python workflow.py

    # Single query | 单次查询
    python workflow.py --query "Research Alphabet class A for me"

    # Run 3 demo cases | 运行 3 个 demo 案例
    python workflow.py --demo

Requirements / 依赖:
    pip install transformers torch peft yfinance
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Configuration | 配置 ──────────────────────────────────────────

FINAL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FINAL_ADAPTER = "checkpoints/lora_v1p2_seed42"
PROMPT_TEMPLATE = "prompts/zero_shot_plain.txt"
FLYWHEEL_LOG = "data/flywheel_log.jsonl"

TICKER_SET = {
    "AAPL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "NFLX",
    "GOOGL", "GOOG", "BRK-A", "BRK-B", "JPM", "BAC",
    "AMD", "INTC", "PLTR", "DIS", "TSM", "BABA", "PDD",
}

TICKER_ALIASES = {"FB": "META"}

_TICKER_RE = re.compile(r"\b([A-Z]{1,5}(?:[.\-/][A-Z0-9]{1,3})?)\b")

COMPANY_NAMES = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.", "TSLA": "Tesla Inc.", "NVDA": "NVIDIA Corp.",
    "NFLX": "Netflix Inc.", "GOOGL": "Alphabet Inc. (Class A)",
    "GOOG": "Alphabet Inc. (Class C)", "BRK-A": "Berkshire Hathaway (Class A)",
    "BRK-B": "Berkshire Hathaway (Class B)", "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corp.", "AMD": "Advanced Micro Devices Inc.",
    "INTC": "Intel Corp.", "PLTR": "Palantir Technologies Inc.",
    "DIS": "The Walt Disney Co.", "TSM": "Taiwan Semiconductor Mfg.",
    "BABA": "Alibaba Group", "PDD": "PDD Holdings Inc.",
}


# ── Layer 1: Symbol Resolver | 第一层：代码解析器 ─────────────────

class SymbolResolver:
    """LoRA v1.2 + normalize ticker resolver."""

    def __init__(self, model_name=FINAL_MODEL, adapter_path=FINAL_ADAPTER):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.prompt_template = Path(PROMPT_TEMPLATE).read_text().strip()
        self._model: Any = None
        self._tokenizer: Any = None

    def _load_model(self):
        if self._model is not None:
            return
        print("  Loading model...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, trust_remote_code=True
        )
        if device.type != "cuda":
            model = model.to(device)  # type: ignore[arg-type]
        model = PeftModel.from_pretrained(model, self.adapter_path)
        model = model.merge_and_unload()  # type: ignore[assignment]
        model.eval()
        self._model = model
        self._device = device
        print(f"  Model loaded on {device}")

    def resolve(self, user_query: str) -> dict:
        """Resolve a user query to a canonical ticker symbol."""
        self._load_model()

        prompt = self.prompt_template.replace("{input}", user_query)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        raw_output = self._tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        ).strip()

        symbol, was_normalized = normalize_ticker(raw_output)

        return {
            "symbol": symbol,
            "raw_output": raw_output,
            "normalized": was_normalized,
            "query": user_query,
        }


def normalize_ticker(raw: str) -> tuple[str, bool]:
    """Normalize raw model output to a canonical ticker symbol."""
    text = raw.strip().upper()
    clean = text.replace(".", "-").replace("/", "-")

    if clean in TICKER_ALIASES:
        return TICKER_ALIASES[clean], True
    if clean in TICKER_SET:
        return clean, clean != text

    candidates = _TICKER_RE.findall(text)
    for candidate in candidates:
        normalized = candidate.replace(".", "-").replace("/", "-")
        if normalized in TICKER_ALIASES:
            return TICKER_ALIASES[normalized], True
        if normalized in TICKER_SET:
            return normalized, True

    for ticker in TICKER_SET:
        if ticker in text:
            return ticker, True

    return "UNKNOWN", True


# ── Layer 2: Market Data Fetch | 第二层：市场数据获取 ─────────────

def fetch_market_data(symbol: str, months: int = 6) -> dict:
    """Fetch price history + basic info for a symbol.

    Uses yfinance if available; falls back to a stub with synthetic data.
    """
    yf_symbol = symbol.replace("-", ".")  # BRK-B → BRK.B for yfinance

    try:
        import yfinance as yf

        ticker = yf.Ticker(yf_symbol)
        end = datetime.now()
        start = end - timedelta(days=months * 30)
        hist = ticker.history(start=start.strftime("%Y-%m-%d"),
                              end=end.strftime("%Y-%m-%d"))

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
        # Fallback stub — always works, for demo purposes
        return {
            "symbol": symbol,
            "company": COMPANY_NAMES.get(symbol, symbol),
            "period": f"{months} months",
            "current_price": "N/A (yfinance not installed)",
            "data_source": "stub",
            "note": "Install yfinance for live data: pip install yfinance",
        }


# ── Layer 3: Memo Generation | 第三层：备忘录生成 ─────────────────

def generate_memo(symbol: str, data: dict, query: str) -> str:
    """Generate a short research memo from market data."""
    company = data.get("company", symbol)
    period = data.get("period", "N/A")

    lines = [
        f"{'='*60}",
        f"  RESEARCH MEMO: {company} ({symbol})",
        f"  研究备忘录：{company} ({symbol})",
        f"{'='*60}",
        f"  Query:   {query}",
        f"  Period:  {period}",
        f"",
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
            f"  [数据来源：存根 — 安装 yfinance 获取实时数据]",
        ]

    lines += [
        f"",
        f"{'='*60}",
    ]
    return "\n".join(lines)


# ── Layer 4: Flywheel Logger | 第四层：飞轮日志 ──────────────────

def log_flywheel(record: dict):
    """Append a structured JSONL record for training flywheel."""
    record["timestamp"] = datetime.now().isoformat()
    with open(FLYWHEEL_LOG, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Main Pipeline | 主流水线 ─────────────────────────────────────

def run_pipeline(resolver: SymbolResolver, query: str, months: int = 6) -> dict:
    """Full pipeline: query → resolve → fetch → memo → log."""
    print(f"\n{'─'*60}")
    print(f"  INPUT: {query}")
    print(f"{'─'*60}")

    # Step 1: Resolve symbol
    result = resolver.resolve(query)
    symbol = result["symbol"]
    raw = result["raw_output"]
    norm_flag = " (normalized)" if result["normalized"] else ""
    print(f"  ✓ Resolver:  {symbol}  (raw: {raw!r}){norm_flag}")

    if symbol == "UNKNOWN":
        print(f"  ✗ Could not resolve symbol. Skipping data fetch.")
        return result

    # Step 2: Fetch market data
    data = fetch_market_data(symbol, months=months)
    print(f"  ✓ Data:      {data.get('data_source', 'unknown')} source")

    # Step 3: Generate memo
    memo = generate_memo(symbol, data, query)
    print(f"\n{memo}")

    # Step 4: Log to flywheel
    log_record = {
        "query": query,
        "raw_output": raw,
        "resolved_symbol": symbol,
        "normalized": result["normalized"],
        "data_source": data.get("data_source"),
        "model": f"{FINAL_MODEL}+LoRA({FINAL_ADAPTER})",
    }
    log_flywheel(log_record)
    print(f"  ✓ Logged to {FLYWHEEL_LOG}")

    return {**result, "data": data, "memo": memo}


# ── Demo Cases | Demo 案例 ───────────────────────────────────────

DEMO_CASES = [
    {
        "query": "Research Alphabet class A for me",
        "months": 6,
        "story": "Google share class — previously confused GOOGL↔GOOG, now correctly resolves to GOOGL",
    },
    {
        "query": "Research FB for the past 3 months",
        "months": 3,
        "story": "Meta alias — model outputs 'FB', normalize layer canonicalizes to META",
    },
    {
        "query": "Analyze Apple over the last 1 year",
        "months": 12,
        "story": "Standard case — stable resolution, proves system handles routine queries reliably",
    },
]


def run_demo(resolver: SymbolResolver):
    """Run all 3 demo cases."""
    print("\n" + "═" * 60)
    print("  TICKER DISAMBIGUATION RESEARCH COPILOT — DEMO")
    print("  股票代码消歧研究 Copilot — 演示")
    print("═" * 60)
    print(f"\n  Final resolver: LoRA v1.2 + normalize(FB→META, BRK.B→BRK-B)")
    print(f"  Base model:     {FINAL_MODEL}")
    print(f"  Adapter:        {FINAL_ADAPTER}")

    for i, case in enumerate(DEMO_CASES, 1):
        print(f"\n\n{'▓'*60}")
        print(f"  DEMO {i}/3: {case['story']}")
        print(f"{'▓'*60}")
        run_pipeline(resolver, case["query"], months=case["months"])

    print(f"\n\n{'═'*60}")
    print(f"  ✓ Demo complete — {len(DEMO_CASES)} cases processed")
    print(f"  Flywheel log: {FLYWHEEL_LOG}")
    print(f"{'═'*60}\n")


# ── Entry Point | 入口 ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ticker Disambiguation Research Copilot Workflow"
    )
    parser.add_argument("--query", type=str, help="Single query to resolve")
    parser.add_argument("--months", type=int, default=6, help="Lookback months")
    parser.add_argument("--demo", action="store_true", help="Run 3 demo cases")
    args = parser.parse_args()

    resolver = SymbolResolver()

    if args.demo:
        run_demo(resolver)
    elif args.query:
        run_pipeline(resolver, args.query, months=args.months)
    else:
        # Interactive mode
        print("\n  Ticker Disambiguation Research Copilot")
        print("  Type a query, or 'quit' to exit.\n")
        resolver._load_model()
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
