# bot-trade-codex with Binance Loader

## Overview
This version integrates `binance_loader.py` to automatically load and convert Binance trade CSVs to OHLCV.

## binance_loader.py
- Place your Binance trade CSV files in a directory, e.g., `data/`.
- Use the loader in your scripts:

```python
from binance_loader import load_binance_data

df = load_binance_data(
    data_dir="C:/path/to/data",
    symbol="BTCUSDT",
    data_type="trades",
    freq="1min",
    year_range=(2021, 2025),
    force_rebuild=False
)
```

## Integration
Replace data loading in:
- `autolearn.py`
- `train_rl.py`
- `evaluate_model.py`

with:

```python
df = load_binance_data(data_dir, symbol, data_type="trades", freq="1min")
```

## Files
- `binance_loader.py`
- Updated project scripts...

## Monitor windows

Helper scripts can be launched in separate consoles using
`tools/monitor_launch.launch_new_console`. Configure behaviour with environment
variables:

- `MONITOR_SHELL=cmd|powershell` – force a specific Windows shell.
- `MONITOR_USE_CONDA_RUN=1` – wrap commands with `conda run` to ensure the
  active environment.
- `MONITOR_DEBUG=1` – print the final command line used for spawning.

## Memory and resume

Training runs persist state in `memory/` and can resume automatically.
Invoke training with `--auto-resume` to continue from the latest snapshot.
Inspect saved state with:

```
python -m tools.memory_cli --print-latest
```
