"""Live dry-run runner consuming live prices and executing paper trades."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import yaml

from bot_trade.data.live_feed import LiveFeed
from bot_trade.gateways import SandboxGateway
from bot_trade.tools.atomic_io import append_jsonl, write_json, write_text
from bot_trade.tools.force_utf8 import force_utf8


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    force_utf8()
    ap = argparse.ArgumentParser(
        description="Live dry-run",
        epilog=(
            "Examples:\n"
            "  paper:   --exchange binance --symbol BTCUSDT --frame 1m\n"
            "  sandbox: --gateway sandbox --exchange binance --symbol BTCUSDT --frame 1m --i-understand-testnet\n"
            "Unknown args are ignored with a warning"
        ),
    )
    ap.add_argument("--exchange", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--frame", required=True)
    ap.add_argument("--gateway", default="paper")
    ap.add_argument("--i-understand-testnet", action="store_true")
    ap.add_argument("--max-orders", type=int, default=50)
    ap.add_argument("--max-notional", type=float)
    ap.add_argument("--model", help="Path to model zip")
    ap.add_argument(
        "--model-optional",
        action="store_true",
        help="Allow missing model and fall back to random actions",
    )
    ap.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Run duration in seconds",
    )
    ap.add_argument("--config", default="config/live_dry_run.yaml")
    ap.add_argument(
        "--bootstrap-price",
        help="VALUE or FILE containing last close to seed the feed",
    )
    ns, extras = ap.parse_known_args()
    if extras:
        print(f"[WARN] ignoring extras: {' '.join(extras)}")
    args = ns
    args.duration = max(int(args.duration), 1)

    bootstrap_price = None
    if args.bootstrap_price:
        bp = Path(args.bootstrap_price)
        if bp.exists():
            with bp.open("r", encoding="utf-8") as fh:
                bootstrap_price = float(fh.read().strip())
        else:
            bootstrap_price = float(args.bootstrap_price)

    cfg = _load_config(args.config)
    ecfg = cfg[args.exchange]
    if args.gateway == "sandbox":
        if not args.i_understand_testnet:
            raise SystemExit("--i-understand-testnet required")
        envs = {
            "binance": ("BINANCE_API_KEY", "BINANCE_API_SECRET"),
            "bybit": ("BYBIT_API_KEY", "BYBIT_API_SECRET"),
        }
        k1, k2 = envs[args.exchange]
        if not os.getenv(k1) or not os.getenv(k2):
            raise SystemExit(f"Missing {k1}/{k2}")
        if "test" not in ecfg.get("rest", "").lower():
            raise SystemExit("sandbox endpoints must be testnet")
        gw = SandboxGateway(args.exchange, args.symbol)
    else:
        gw = None
    feed = LiveFeed(
        ecfg.get("ws"),
        ecfg["rest"] + ecfg["price_path"],
        interval=1.0,
        bootstrap_price=bootstrap_price,
    )

    class HoldPolicy:
        def action(self, _: float) -> str:
            return "hold"

    class RandomPolicy:
        def __init__(self) -> None:
            self.rng = random.Random()

        def action(self, _: float) -> str:
            return self.rng.choice(["buy", "sell", "hold"])

    policy: object
    if args.model and Path(args.model).exists():
        policy = HoldPolicy()  # placeholder for real model policy
    elif not args.model_optional:
        raise SystemExit("model required")
    else:
        print("[WARN] model missing, falling back to RandomPolicy")
        policy = RandomPolicy()
    base_policy = policy
    policy_ref = {"policy": policy}

    run_dir = Path("results") / args.symbol / args.frame / str(int(time.time()))
    logs_dir = run_dir / "logs"
    metrics = []
    order_count = 0
    notional_total = 0.0
    orders_path = run_dir / "orders.jsonl"
    exec_path = run_dir / "executions.jsonl"

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data: str) -> None:  # pragma: no cover - simple tee
            for s in self.streams:
                s.write(data)

        def flush(self) -> None:  # pragma: no cover - simple tee
            for s in self.streams:
                s.flush()

    logs_dir.mkdir(parents=True, exist_ok=True)
    log_fh = (logs_dir / "run.log").open("w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_fh)

    def on_tick(price: float) -> None:
        nonlocal order_count, notional_total
        action = policy_ref["policy"].action(price)
        metrics.append({"ts": time.time(), "price": price, "action": action})
        print(f"[LIVE] tick={price} action={action}")
        if gw and action in {"buy", "sell"}:
            if order_count >= args.max_orders:
                print("[LIVE] max-orders reached; skipping order")
                return
            notional = price * 0.001
            if args.max_notional and notional_total + notional > args.max_notional:
                print("[LIVE] max-notional reached; skipping order")
                return
            side = "BUY" if action == "buy" else "SELL"
            qty = 0.001
            start_ts = time.time()
            res = gw.place_order(None, side, qty, "LIMIT", price)
            ack_ms = int((time.time() - start_ts) * 1000)
            print(
                f"ORDER id={res.id} type=LIMIT side={side} qty={qty} px={price} ack_ms={ack_ms}"
            )
            append_jsonl(
                orders_path,
                {"id": res.id, "side": side, "qty": qty, "px": price, "ack_ms": ack_ms, "ts": time.time()},
            )
            order_count += 1
            notional_total += notional
            if res.filled_qty:
                latency_ms = ack_ms
                print(
                    f"FILL side={side} qty={res.filled_qty} px={res.avg_price} fee={res.fees} "
                    f"maker={res.status == 'FILLED'} min_fee=0 slip_bps=0 latency_ms={latency_ms} ts={time.time()}"
                )
                append_jsonl(
                    exec_path,
                    {
                        "side": side,
                        "qty": res.filled_qty,
                        "px": res.avg_price,
                        "fee": res.fees,
                        "maker": res.status == 'FILLED',
                        "min_fee": 0,
                        "slip_bps": 0,
                        "latency_ms": latency_ms,
                        "ts": time.time(),
                    },
                )

    feed.stream(args.symbol, on_tick)
    start = time.time()
    warned = False
    try:
        while time.time() - start < args.duration:
            if feed.last_price is None:
                if not warned and time.time() - start > 10 * feed.interval:
                    print("[WARN] no valid price after 10 polls; using HoldPolicy")
                    policy_ref["policy"] = HoldPolicy()
                    warned = True
            elif warned and isinstance(policy_ref["policy"], HoldPolicy):
                policy_ref["policy"] = base_policy
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout = orig_stdout
        log_fh.close()
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "ticks": len(metrics),
            "orders": order_count,
            "notional": notional_total,
            "gateway": args.gateway,
        }
        write_json(run_dir / "summary.json", summary)
        risk_path = run_dir / "risk_flags.jsonl"
        if metrics:
            for m in metrics:
                append_jsonl(risk_path, m)
        else:
            write_text(risk_path, "")
        csv_path = run_dir / "metrics.csv"
        if metrics:
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=metrics[0].keys())
                w.writeheader()
                w.writerows(metrics)
        else:
            write_text(csv_path, "")
        if gw:
            if not orders_path.exists():
                write_text(orders_path, "")
            if not exec_path.exists():
                write_text(exec_path, "")


if __name__ == "__main__":
    main()
