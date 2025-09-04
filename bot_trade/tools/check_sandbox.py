from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
import yaml

from bot_trade.tools.force_utf8 import force_utf8
from bot_trade.gateways.sandbox_gateway import SandboxGateway


def _has_keys(exchange: str) -> bool:
    envs = {
        "binance": ("BINANCE_API_KEY", "BINANCE_API_SECRET"),
        "bybit": ("BYBIT_API_KEY", "BYBIT_API_SECRET"),
    }
    k1, k2 = envs[exchange]
    return bool(os.getenv(k1) and os.getenv(k2))


def _price_only(exchange: str, symbol: str) -> int:
    cfg = yaml.safe_load(Path("config/live_dry_run.yaml").read_text(encoding="utf-8"))
    ecfg = cfg[exchange]
    url = ecfg["rest"] + ecfg["price_path"]
    try:
        r = requests.get(url, params={"symbol": symbol}, timeout=10)
        r.raise_for_status()
        data = r.json()
        price = float(
            data.get("price")
            or data.get("result", {}).get("list", [{}])[0].get("lastPrice", 0)
        )
        print("last_price", price)
        print(
            f"[HINT] set {exchange.upper()}_API_KEY/{exchange.upper()}_API_SECRET for order test"
        )
        return 0
    except Exception as exc:  # pragma: no cover - network
        print(f"[ERROR] price fetch failed: {exc}")
        print(f"[HINT] check internet or endpoint {url}")
        return 1


def main() -> None:
    force_utf8()
    ap = argparse.ArgumentParser(description="Sandbox gateway smoke test")
    ap.add_argument("--exchange", required=True, choices=["binance", "bybit"])
    ap.add_argument("--symbol", required=True)
    args = ap.parse_args()

    if not _has_keys(args.exchange):
        sys.exit(_price_only(args.exchange, args.symbol))

    try:
        gw = SandboxGateway(args.exchange, args.symbol)
        price = gw.get_price()
        print("last_price", price)
        bal = gw.get_balance("USDT")
        print("USDT_balance", bal)
        res = gw.place_order(None, "BUY", 0.001, "MARKET")
        print("order_id", res.id)
        gw.cancel_order(res.id)
        print("cancelled")
        sys.exit(0)
    except Exception as exc:  # pragma: no cover - network
        print(f"[ERROR] {exc}")
        envs = {
            "binance": ("BINANCE_API_KEY", "BINANCE_API_SECRET"),
            "bybit": ("BYBIT_API_KEY", "BYBIT_API_SECRET"),
        }
        k1, k2 = envs[args.exchange]
        print(f"[HINT] ensure {k1}/{k2} and testnet connectivity")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
