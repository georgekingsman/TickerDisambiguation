"""src/resolver.py — LoRA v1.2 symbol resolver + normalize layer."""

import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.config import (
    FINAL_MODEL, FINAL_ADAPTER, PROMPT_TEMPLATE,
    TICKER_SET, TICKER_ALIASES,
)

_TICKER_RE = re.compile(r"\b([A-Z]{1,5}(?:[.\-/][A-Z0-9]{1,3})?)\b")


def normalize_ticker(raw: str) -> tuple[str, bool]:
    """Normalize raw model output to a canonical ticker symbol.

    Returns (symbol, was_normalized).
    """
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


class SymbolResolver:
    """LoRA v1.2 + normalize ticker resolver."""

    def __init__(self, model_name: str = FINAL_MODEL, adapter_path: str = FINAL_ADAPTER):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.prompt_template = Path(PROMPT_TEMPLATE).read_text().strip()
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None

    def load(self):
        """Explicitly load model into memory (called lazily on first resolve)."""
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
            model = model.to(device)
        model = PeftModel.from_pretrained(model, self.adapter_path)
        model = model.merge_and_unload()
        model.eval()
        self._model = model
        self._device = device
        print(f"  Model loaded on {device}")

    def resolve(self, user_query: str) -> dict:
        """Resolve a user query to a canonical ticker symbol.

        Returns a dict with keys: symbol, raw_output, normalized, query.
        """
        self.load()

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
