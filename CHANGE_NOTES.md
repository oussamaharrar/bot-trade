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
2025-09-03 — Dual-archive model lifecycle: previous BEST models now moved to agents/<S>/<F>/archive_best/ (stamped filenames + optional vecnorm_best snapshot + index.csv). Non-best artifacts remain under archive/. Atomic promotions; best_meta.json updated.
## 2025-09-16
- chart export made robust (column aliases, non-empty guarantee, Agg backend), synchronous post-run export, and atomic KB updates with JSONL entries.

## 2025-09-03
- **Files**: `bot_trade/config/rl_paths.py`, `bot_trade/tools/export_charts.py`, `bot_trade/tools/monitor_manager.py`, `bot_trade/tools/evaluate_model.py`, `bot_trade/tools/gen_synth_data.py`, `bot_trade/train_rl.py`, `bot_trade/config/rl_args.py`, `CHANGE_NOTES.md`
- **Rationale**: reports now ingest reward/step/train/risk/callbacks/signals with ≥5 charts, automatic evaluation and summary export, knowledge base JSONL updates, synthetic data generator for dev, and refreshed CLI defaults.
- **Risks**: synthetic evaluation is simplistic; chart placeholders may hide data issues.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/train_rl.py`; run smoke `python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready`, `python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --policy MlpPolicy --device cpu --n-envs 2 --n-steps 512 --batch-size 1024 --total-steps 2048 --net-arch "1024,512,256" --activation silu --vecnorm --headless --allow-synth --kb-file Knowlogy/kb.jsonl --data-dir data_ready`, then `python -m bot_trade.tools.monitor_manager --symbol BTCUSDT --frame 1m --run-id latest --debug-export`.

## 2025-09-20
- **Files**: `bot_trade/tools/kb_writer.py`, `bot_trade/train_rl.py`, `CHANGE_NOTES.md`
- **Rationale**: centralize knowledge base writes with canonical path and richer run metadata.
- **Risks**: KB file may grow large; portfolio metrics depend on state availability.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/train_rl.py`, `python - <<'PY'
from bot_trade.config.rl_paths import RunPaths
from bot_trade.tools.kb_writer import kb_append
rp = RunPaths('BTCUSDT','1m','testrun')
kb_append(rp, {"run_id":"testrun","symbol":"BTCUSDT","frame":"1m","ts":"0","images":0,"rows_reward":0,"rows_step":0,"rows_train":0,"rows_risk":0,"rows_signals":0,"vecnorm_applied":False,"vecnorm_snapshot_saved":False,"best":False,"last":False,"best_model_path":"/tmp","eval":{},"portfolio":{},"notes":""})
print('KB', rp.kb_file)
PY`
