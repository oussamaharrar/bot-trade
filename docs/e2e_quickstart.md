# E2E Quickstart

1. Train a small model:
   ```bash
   python bot_trade/train_rl.py --config config/train_btcusdt_1m.yaml --max-steps 4000 --seed 7
   ```
2. Evaluate the latest run and produce a tearsheet:
   ```bash
   python -m bot_trade.eval.eval_run --run-dir results/BTCUSDT/1m/latest --tearsheet-html
   ```
3. Apply walk-forward gate:
   ```bash
   python -m bot_trade.eval.wfa_gate --config config/wfa.yaml --symbol BTCUSDT --frame 1m --windows 3 --embargo 0.05
   ```
4. Optional live dry run with model fallback:
   ```bash
   python -m bot_trade.runners.live_dry_run --exchange binance --symbol BTCUSDT --frame 1m --gateway paper --model results/BTCUSDT/1m/latest/artifacts/best_model.zip --duration 120 --model-optional
   ```
5. Verify paths:
   ```bash
   python -m bot_trade.tools.paths_doctor --strict
   ```
