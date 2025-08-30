# bot-trade-codex

## System Overview
`bot_trade` is a research environment for reinforcement-learning based trading.
The package is divided into a few key parts:

- `bot_trade.config`: environment setup, risk management, writer utilities and
  RL helper modules.
- `bot_trade.signals`: entry/exit and reward signal helpers.
- `bot_trade.ai_core`: knowledge base, verification and self-improvement tools.
- `bot_trade.tools`: ETL helpers and miscellaneous utilities.

Training orchestration lives in `bot_trade.train_rl` and can be executed via
`python -m bot_trade.train_rl`. Artifacts such as logs, CSV summaries and
`memory/memory.json` are written under `results/` and `logs/` every 100k steps
(configurable) and at the start/end of a run. Training resumes exactly from the
latest snapshot when `--auto-resume` is supplied. A knowledge base is stored in
`memory/knowledge_base_full.json` and updated at each artifact tick.

## Installation

```bash
pip install -e .
```

## Running

```bash
python -m bot_trade.train_rl --help
python -m bot_trade.run_multi_gpu --help
python -m bot_trade.run_bot --help
python -m bot_trade.run_rl_agent --help

# CLI

The monitor manager can be invoked either as a module or via the installed
console script:

```
python -m bot_trade.tools.monitor_manager --symbol BTCUSDT --frame 1m
# or simply
bot-monitor --symbol BTCUSDT --frame 1m
```

## Collaboration

Anyone who reads or edits code **must** record a change note in
[CHANGE_NOTES.md](CHANGE_NOTES.md) describing what was changed and why.

## Legacy Binance Loader

This version integrates `binance_loader.py` to automatically load and convert
Binance trade CSVs to OHLCV. Place your Binance trade CSV files in a directory,
e.g., `data/`, then use the loader in your scripts:

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

Replace existing data loading in your scripts (`autolearn.py`, `train_rl.py`,
`evaluate_model.py`) with the call above.

## Monitor windows

Helper scripts can be launched in separate consoles using
`tools/monitor_launch.launch_new_console`. Configure behaviour with environment
variables:

- `MONITOR_SHELL=cmd|powershell` – force a specific Windows shell.
- `MONITOR_USE_CONDA_RUN=1` – wrap commands with `conda run` to ensure the
  active environment.
- `MONITOR_DEBUG=1` – print the final command line used for spawning.

## Memory and resume

Training runs persist state in `memory/memory.json` using schema version 2 and
can resume automatically. Inspect saved state via:

```bash
python -m tools.memory_cli --print-latest
```

