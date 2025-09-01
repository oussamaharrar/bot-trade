# CLI Quick Reference

## Training
```
python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --device cpu --allow-synth
```

## Monitor
```
python -m bot_trade.tools.monitor_manager --symbol BTCUSDT --frame 1m --run-id latest --debug-export
```

## Charts Export
```
python -m bot_trade.tools.export_run_charts --symbol BTCUSDT --frame 1m --run-id latest
```

## Evaluation
```
python -m bot_trade.tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest
```

## Synthetic Data
```
python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready
```
