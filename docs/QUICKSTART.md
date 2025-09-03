# Quickstart

1. Synthetic data + PPO eval
```bash
python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready
python -m bot_trade.train_rl --algorithm PPO --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --total-steps 128 --headless --allow-synth --data-dir data_ready --no-monitor
python -m bot_trade.tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest --tearsheet
```

2. Continuous SAC
```bash
python -m bot_trade.train_rl --algorithm SAC --continuous-env --gateway paper \
  --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --total-steps 128 \
  --headless --allow-synth --data-dir data_ready --no-monitor
python -m bot_trade.tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest
```

3. ai_core dummy signals
```bash
python -m bot_trade.train_rl --algorithm SAC --continuous-env --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --total-steps 128 --headless --allow-synth --data-dir data_ready --ai-core --emit-dummy-signals --no-monitor
```
