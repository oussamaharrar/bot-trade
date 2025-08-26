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
