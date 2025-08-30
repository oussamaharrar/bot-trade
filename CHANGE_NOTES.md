# Change Notes

Record every code change here.

Template:

```
## <date>
- **Files**: <files touched>
- **Rationale**: <why>
- **Risks**: <potential issues>
- **Test Steps**: <how to verify>
```

## 2024-05-08
- **Files**: moved core modules under `bot_trade/`, updated imports, new CLI wrappers, `pyproject.toml`, `train_rl.py`, `run_multi_gpu.py`, docs
- **Rationale**: normalize package layout and avoid heavy imports on CLI help
- **Risks**: potential import regressions or missing dependencies during runtime
- **Test Steps**: `pip install -e .`, `python -m bot_trade.train_rl --help`, `python -m bot_trade.run_multi_gpu --help`, `python - <<'PY' ... py_compile`

## 2025-08-30
- **Files**: `bot_trade/config/rl_writers.py`, `bot_trade/tools/monitor_manager.py`
- **Rationale**: tail sanitization for writers and clearer run-id diagnostics
- **Risks**: low, affects logging only
- **Test Steps**: `python -m py_compile bot_trade/config/rl_writers.py bot_trade/config/rl_callbacks.py bot_trade/config/update_manager.py bot_trade/train_rl.py bot_trade/tools/memory_manager.py bot_trade/tools/monitor_manager.py bot_trade/tools/analytics_common.py bot_trade/config/log_setup.py bot_trade/config/rl_args.py`, `pytest`, `python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --total-steps 2000 --n-envs 2 --device cpu --resume-auto; test $? -eq 3 && echo EXIT_CODE_3`, `python -m bot_trade.tools.monitor_manager --symbol BTCUSDT --frame 1m --no-wait --headless; test $? -eq 2 && echo EXIT_CODE_2`

