# E2E Quickstart

1. Generate tiny synthetic data:
   ```bash
   python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --days 3 --out data_ready
   ```
2. Train a small model:
   ```bash
   python bot_trade/train_rl.py --config config/train_btcusdt_1m.yaml --max-steps 4000 --seed 7 --data-dir data_ready
   ```
3. Evaluate the latest run and produce a tearsheet:
   ```bash
   python -m bot_trade.eval.eval_run --run-dir results/BTCUSDT/1m/latest --tearsheet-html
   ```
4. Apply walk-forward gate (smoke profile):
   ```bash
   python -m bot_trade.eval.wfa_gate --config config/wfa.yaml --symbol BTCUSDT --frame 1m --windows 3 --embargo 0.05 --profile smoke
   ```
5. Optional live dry run with model fallback:
   ```bash
   python -m bot_trade.runners.live_dry_run --exchange binance --symbol BTCUSDT --frame 1m \
     --gateway paper --model results/BTCUSDT/1m/latest/artifacts/best_model.zip --duration 90 --model-optional
   ```
6. Verify paths:
   ```bash
   python -m bot_trade.tools.paths_doctor --symbol BTCUSDT --frame 1m --strict
   ```
