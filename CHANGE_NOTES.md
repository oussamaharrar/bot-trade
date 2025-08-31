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

## 2025-08-31
- **Files**: multiple across training pipeline (paths, writers, monitor, args, model management)
- **Rationale**: VecNormalize resume application, headless monitor exports, run-aware directories, UTF-8 writers with run_id columns, signals/callback logging, model lifecycle management, batch-size clamping, and graceful interrupt handling
- **Risks**: moderate; wide-reaching changes may affect logging and resume behaviour
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py`, `pytest`, headless smoke run `python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --total-steps 2000 --n-envs 2 --device cpu --headless`, then `python -m bot_trade.tools.monitor_manager --symbol BTCUSDT --frame 1m --run-id <run_id> --headless`


## 2025-09-01
- **Files**: `bot_trade/config/rl_paths.py`, `bot_trade/tools/runctx.py`, `bot_trade/config/rl_writers.py`, `bot_trade/train_rl.py`, `bot_trade/tools/monitor_manager.py`, `bot_trade/tools/migrate_artifacts.py`, `CHANGE_NOTES.md`
- **Rationale**: introduce `RunPaths` with short run IDs and run.json metadata, apply VecNormalize on resume, headless monitor exports PNG charts, atomic UTF-8 writers, migration helper, and best-model tracking.
- **Risks**: path refactor may break legacy scripts; monitor charts are basic placeholders.
- **Test Steps**: `python -m py_compile bot_trade/config/.py bot_trade/tools/.py`, `pytest -q || echo "no tests found"`, `python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --total-steps 2000 --n-envs 2 --device cpu --headless`, `python -m bot_trade.tools.monitor_manager --symbol BTCUSDT --frame 1m --headless`

## 2025-09-02
- **Files**: `bot_trade/tools/monitor_manager.py`, `CHANGE_NOTES.md`
- **Rationale**: generate real headless charts without placeholders and ensure risk flag exports via `risk_log.csv`.
- **Risks**: chart generation still depends on log availability.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py`, `pytest -q || echo "no tests found"`, `python -m bot_trade.tools.monitor_manager --symbol BTCUSDT --frame 1m --headless`
