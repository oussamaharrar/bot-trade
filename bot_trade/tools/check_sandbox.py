from __future__ import annotations

import argparse

from bot_trade.tools.force_utf8 import force_utf8
from bot_trade.gateways.sandbox_gateway import SandboxGateway


def main() -> None:
    force_utf8()
    ap = argparse.ArgumentParser(description="Sandbox gateway smoke test")
    ap.add_argument("--exchange", required=True, choices=["binance", "bybit"])
    ap.add_argument("--symbol", required=True)
    args = ap.parse_args()

    gw = SandboxGateway(args.exchange, args.symbol)
    price = gw.get_price()
    print("last_price", price)
    bal = gw.get_balance("USDT")
    print("USDT_balance", bal)
    res = gw.place_order(None, "BUY", 0.001, "MARKET")
    print("order_id", res.id)
    gw.cancel_order(res.id)
    print("cancelled")


if __name__ == "__main__":
    main()
