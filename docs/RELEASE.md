# Release Checklist

## Pre-release
- `python -m py_compile $(git ls-files 'bot_trade/**/*.py' 'bot_trade/*.py')`
- Run CPU smokes for PPO and SAC; logs contain `[EVAL]` and `[POSTRUN]`.
- `python -m bot_trade.tools.dev_checks --symbol BTCUSDT --frame 1m --run-id latest` → `[CHECKS] ok=.. warnings=..`
- `python -m bot_trade.tools.sweep --mode random --n-trials 4 --symbol BTCUSDT --frame 1m --algorithm SAC --continuous-env --headless --allow-synth --data-dir data_ready` → `[SWEEP]` with CSV/MD/JSONL.
- Ensure tearsheet generated for latest run.

## Artifacts
- `reports/.../charts/*.png`
- `reports/.../tearsheet.html`
- `reports/.../sweep/summary.csv|summary.md|runs.jsonl`
- optional `checkpoints/` index

## Guards
- `[ALGO_GUARD]`
- `[SWEEP_WARN]`
- `[RISK_CLAMP]`
- `[IO] fixed_trailing_newline`

## Reproduce
```bash
python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready
python -m bot_trade.train_rl --algorithm PPO --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --total-steps 128 --headless --allow-synth --data-dir data_ready --no-monitor
python -m bot_trade.tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest --tearsheet
python -m bot_trade.tools.sweep --mode random --n-trials 4 --symbol BTCUSDT --frame 1m --algorithm SAC --continuous-env --headless --allow-synth --data-dir data_ready
python -m bot_trade.tools.dev_checks --symbol BTCUSDT --frame 1m --run-id latest
```

## Versioning
- Tag as `v<major>.<minor>.<patch>`.
- Update `bot_trade/__version__.py` and `CHANGE_NOTES.md`.
