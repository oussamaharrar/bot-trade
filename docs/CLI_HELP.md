# CLI Help

## Core flags
- `--continuous-env` enable Box actions for SAC/TD3/TQC
- `--algorithm` choose RL algo
- `--ai-core` enable external signal pipeline
- `--emit-dummy-signals` produce demo signals
- `--dry-run` run ai_core without writes

## Presets & Sweeper
- `--preset training=name --preset net=name`
- `tools.sweep --mode grid|random --n-trials N --learning-rate a,b --batch-size x,y`

## Eval / tearsheet
`tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest --tearsheet`

## Example
`python -m bot_trade.train_rl --algorithm PPO --symbol BTCUSDT --frame 1m --preset training=ppo_cpu_smoke --preset net=tiny --headless`

Expected logs: `[HEADLESS] backend=Agg`, `[PRESET] training=...`, `[EVAL] ...`, `[POSTRUN] ...`

### Troubleshooting
- `[ALGO_GUARD]` invalid algo/policy combo
- `[SWEEP_WARN]` sweeper thresholds
- `[LATEST] none` missing run
