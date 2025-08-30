"""Interactive monitor manager for training session."""
from __future__ import annotations

from bot_trade.tools import bootstrap  # noqa: F401  # Import path fixup when run directly

import argparse
from typing import List

from bot_trade.tools.analytics_common import wait_for_first_write
from bot_trade.tools.monitor_launch import launch_new_console
from bot_trade.tools.paths import results_dir, report_dir

MENU = """
=== MONITOR MANAGER (symbol={symbol} frame={frame}) ===
[t] Live Ticker ({refresh}s)
[c] CLI Console
[a] Anomaly Watch ({refresh}s)
[r] Resource Monitor (1s)
[e] Batch Exporter (one-shot)
[q] Quit Manager
Choose: """


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='BTCUSDT')
    ap.add_argument('--frame', default='1m')
    ap.add_argument('--refresh', type=int, default=10)
    ap.add_argument('--images-out')
    ap.add_argument('--base')
    ap.add_argument('--run-id', required=True)  # NOTE: interface changed here - run_id required for path namespacing
    ap.add_argument('--no-wait', action='store_true')
    args = ap.parse_args()

    img_dir = args.images_out.format(symbol=args.symbol, frame=args.frame) if args.images_out else None

    base_dir = args.base or str(results_dir(args.symbol, args.frame))

    if not args.no_wait:
        print('Waiting for first log write...', flush=True)
        wait_for_first_write(base_dir, args.symbol, args.frame)

    while True:
        try:
            choice = input(MENU.format(symbol=args.symbol, frame=args.frame, refresh=args.refresh)).strip().lower()[:1]
        except EOFError:
            break
        if choice == 'q':
            break
        elif choice == 't':
            args_list = [
                '--symbol', args.symbol,
                '--frame', args.frame,
                '--refresh', str(args.refresh),
                '--base', base_dir,
                '--run-id', args.run_id,
            ]
            if img_dir:
                args_list += ['--images-out', img_dir]
            launch_new_console('LIVE-TICKER', 'tools.live_ticker', args_list)
        elif choice == 'c':
            launch_new_console('CLI-CONSOLE', 'tools.cli_console', [
                '--symbol', args.symbol,
                '--frame', args.frame,
                '--base', base_dir,
            ])
        elif choice == 'a':
            launch_new_console('ANOMALY-WATCH', 'tools.anomaly_watch', [
                '--symbol', args.symbol,
                '--frame', args.frame,
                '--refresh', str(args.refresh),
                '--base', base_dir,
                '--run-id', args.run_id,
            ])
        elif choice == 'r':
            launch_new_console('RESOURCE-MON', 'tools.resource_monitor', [
                '--symbol', args.symbol,
                '--frame', args.frame,
                '--base', base_dir,
            ])
        elif choice == 'e':
            out = img_dir or str(report_dir(args.symbol, args.frame, args.run_id))
            launch_new_console('BATCH-EXPORT', 'tools.export_charts', [
                '--symbol', args.symbol,
                '--frame', args.frame,
                '--base', base_dir,
                '--run-id', args.run_id,
                '--out', out,
            ])
            print(f'Exporting to {out}', flush=True)
        else:
            print('unknown choice', flush=True)


if __name__ == '__main__':
    main()
