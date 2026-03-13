"""src/mcp_server.py — Minimal Yahoo Finance MCP server (stdlib only).

Wraps data_fetch.py as HTTP endpoints, making the copilot pipeline compatible
with n8n's HTTP Request nodes.  Import or port to a proper ASGI framework
(FastAPI / Starlette) if production throughput is needed.

Start:
    python -m src.mcp_server          # default port 8000
    python -m src.mcp_server --port 9000

Endpoints:
    GET  /health
    GET  /mcp/get_price_history?symbol=AAPL&months=6
    GET  /mcp/get_ticker_info?symbol=AAPL
    POST /resolve                     (body: {"query": "Research Apple"})

n8n integration:
    Set base_url = http://localhost:8000 in all HTTP Request nodes.
    The resolver service (POST /resolve) can be run separately on port 8001
    via:  python app.py --server  (not yet wired; see n8n_workflow.json for
    the intended topology).
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from src.data_fetch import fetch_market_data
from src.resolver import normalize_ticker

DEFAULT_PORT = 8000


class MCPHandler(BaseHTTPRequestHandler):
    """Route GET + POST requests to the appropriate data_fetch functions."""

    def log_message(self, fmt, *args):  # suppress default access log
        pass

    # ── helpers ──────────────────────────────────────────────────

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── GET routing ───────────────────────────────────────────────

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        def p(key: str, default: str = "") -> str:
            return qs.get(key, [default])[0]

        if parsed.path == "/health":
            self._send_json({"status": "ok", "service": "Yahoo Finance MCP"})

        elif parsed.path == "/mcp/get_price_history":
            symbol = p("symbol", "AAPL")
            months = int(p("months", "6"))
            data = fetch_market_data(symbol, months=months)
            # Return price-focused fields only (matches MCP tool contract)
            self._send_json({
                "symbol": data.get("symbol"),
                "company": data.get("company"),
                "period": data.get("period"),
                "current_price": data.get("current_price"),
                "period_start_price": data.get("period_start_price"),
                "period_return": data.get("period_return"),
                "period_high": data.get("period_high"),
                "period_low": data.get("period_low"),
                "avg_volume": data.get("avg_volume"),
                "data_source": data.get("data_source"),
                "cache_hit": data.get("cache_hit", False),
            })

        elif parsed.path == "/mcp/get_ticker_info":
            symbol = p("symbol", "AAPL")
            data = fetch_market_data(symbol, months=1)
            # Return fundamentals only (matches MCP tool contract)
            self._send_json({
                "symbol": data.get("symbol"),
                "company": data.get("company"),
                "market_cap": data.get("market_cap", "N/A"),
                "pe_ratio": data.get("pe_ratio", "N/A"),
                "sector": data.get("sector", "N/A"),
                "data_source": data.get("data_source"),
                "cache_hit": data.get("cache_hit", False),
            })

        else:
            self._send_json({"error": f"Unknown endpoint: {parsed.path}"}, 404)

    # ── POST routing ──────────────────────────────────────────────

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if parsed.path == "/resolve":
            query: str = body.get("query", "")
            # Extract last uppercase token as candidate ticker for normalize
            candidate = query.strip().split()[-1].upper() if query.strip() else ""
            symbol, normalized = normalize_ticker(candidate)
            self._send_json({
                "query": query,
                "symbol": symbol,
                "normalized": normalized,
            })

        else:
            self._send_json({"error": f"Unknown endpoint: {parsed.path}"}, 404)


def main() -> None:
    port = DEFAULT_PORT
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])

    server = HTTPServer(("localhost", port), MCPHandler)
    print(f"  Yahoo Finance MCP server  →  http://localhost:{port}")
    print("  Endpoints:")
    print(f"    GET  http://localhost:{port}/health")
    print(f"    GET  http://localhost:{port}/mcp/get_price_history?symbol=AAPL&months=6")
    print(f"    GET  http://localhost:{port}/mcp/get_ticker_info?symbol=AAPL")
    print(f"    POST http://localhost:{port}/resolve")
    print("  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
