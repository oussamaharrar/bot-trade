"""Simple CPU/RAM/GPU usage monitor."""
from __future__ import annotations

import argparse
import time
import os

import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    NVML = True
except Exception:
    NVML = False

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def gpu_stats() -> str:
    if NVML:
        try:
            parts = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                parts.append(f"GPU{i}:{util.gpu}% {mem.used/1e9:.1f}/{mem.total/1e9:.1f}GB")
            return " | ".join(parts)
        except Exception:
            pass
    if torch is not None and torch.cuda.is_available():
        try:
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"GPU0:{used:.1f}/{total:.1f}GB"
        except Exception:
            pass
    return "GPU:n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', type=float, default=1.0)
    ap.add_argument('--no-wait', action='store_true')
    ap.add_argument('--base', default='results')
    ap.add_argument('--symbol', default='BTCUSDT')
    ap.add_argument('--frame', default='1m')
    args = ap.parse_args()

    from bot_trade.tools.analytics_common import wait_for_first_write
    if not args.no_wait:
        wait_for_first_write(args.base, args.symbol, args.frame)

    while True:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        line = f"CPU {cpu:5.1f}% | RAM {mem.used/1e9:.1f}/{mem.total/1e9:.1f}GB ({mem.percent:.1f}%)"
        line += " | " + gpu_stats()
        try:
            io = psutil.disk_io_counters()
            line += f" | IO read {io.read_bytes/1e6:.0f}MB write {io.write_bytes/1e6:.0f}MB"
        except Exception:
            pass
        print(line, flush=True)
        time.sleep(max(0.1, args.refresh))


if __name__ == '__main__':
    main()
