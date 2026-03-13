"""src/config.py — Shared constants for the copilot pipeline."""

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
