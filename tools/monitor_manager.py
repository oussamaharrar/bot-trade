"""Interactive monitor manager for training session."""
from __future__ import annotations

import argparse
import os
from typing import List

from tools.analytics_common import wait_for_first_write
from tools.monitor_launch import launch_new_console

TOOLS_DIR = os.path.dirname(__file__)

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
    ap.add_argument('--base', default='results')
    ap.add_argument('--no-wait', action='store_true')
    args = ap.parse_args()

    img_dir = args.images_out.format(symbol=args.symbol, frame=args.frame) if args.images_out else None

    if not args.no_wait:
        print('Waiting for first log write...', flush=True)
        wait_for_first_write(args.base, args.symbol, args.frame)

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
                '--base', args.base,
            ]
            if img_dir:
                args_list += ['--images-out', img_dir]
            launch_new_console('LIVE-TICKER', os.path.join(TOOLS_DIR, 'live_ticker.py'), args_list)
        elif choice == 'c':
            launch_new_console('CLI-CONSOLE', os.path.join(TOOLS_DIR, 'cli_console.py'), [
                '--symbol', args.symbol,
                '--frame', args.frame,
                '--base', args.base,
            ])
        elif choice == 'a':
            launch_new_console('ANOMALY-WATCH', os.path.join(TOOLS_DIR, 'anomaly_watch.py'), [
                '--symbol', args.symbol,
                '--frame', args.frame,
                '--refresh', str(args.refresh),
                '--base', args.base,
            ])
        elif choice == 'r':
            launch_new_console('RESOURCE-MON', os.path.join(TOOLS_DIR, 'resource_monitor.py'), [
                '--symbol', args.symbol,
                '--frame', args.frame,
                '--base', args.base,
            ])
        elif choice == 'e':
            out = img_dir or os.path.join('exports', args.symbol, args.frame, 'batch')
            launch_new_console('BATCH-EXPORT', os.path.join(TOOLS_DIR, 'export_charts.py'), [
                '--symbol', args.symbol,
                '--frame', args.frame,
                '--base', args.base,
                '--out', out,
            ])
            print(f'Exporting to {out}', flush=True)
        else:
            print('unknown choice', flush=True)


if __name__ == '__main__':
    main()
