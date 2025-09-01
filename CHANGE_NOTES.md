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

## 2025-09-21
- **Files**: `bot_trade/train_rl.py`, `CHANGE_NOTES.md`
- **Rationale**: persist run and latest state snapshots under performance reports with atomic writes.
- **Risks**: directories without write access may miss state files.
- **Test Steps**: `python -m py_compile bot_trade/train_rl.py`, `pytest -q || echo "no tests found"`

## 2025-09-22
- **Files**: `bot_trade/train_rl.py`, `bot_trade/config/rl_callbacks.py`, `continue_train_agent.py`, `CHANGE_NOTES.md`
- **Rationale**: ensure dual-archive promotion moves previous best atomically and guard resume against corrupt checkpoints.
- **Risks**: move operations may fail across filesystems; resume fallback may omit certain errors.
- **Test Steps**: `python -m py_compile bot_trade/train_rl.py bot_trade/config/rl_callbacks.py continue_train_agent.py`

## Developer Notes — 2025-09-23
- A) KB: canonical JSONL appender; atomic writes; schema stabilized; integrated after charts+eval+state.

## 2025-09-01
- **Files**: `bot_trade/config/rl_args.py`, `bot_trade/config/device.py`, `bot_trade/config/log_setup.py`, `bot_trade/tools/monitor_launch.py`, `bot_trade/tools/run_state.py`, `bot_trade/train_rl.py`, `DEV_NOTES.md`, `CHANGE_NOTES.md`
- **Rationale**: split train orchestrator utilities into dedicated modules and merge duplicate run-state helpers.
- **Risks**: consumers importing old helper paths may break; global run_state files deprecated.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/train_rl.py`, `python -m bot_trade.train_rl --help`, `python -m bot_trade.tools.export_charts --help`, `python -m bot_trade.tools.eval_run --help`, smoke training run with `--allow-synth`.

## Developer Notes — 2025-09-01 (Structural Split & De-dup)

| source                                   | canonical module                 | merged funcs                         | removed |
|------------------------------------------|----------------------------------|--------------------------------------|---------|
| `train_rl.validate_args`                 | `config.rl_args.validate_args`   | yes                                   | -       |
| `train_rl.clamp_batch`                   | `config.rl_args.clamp_batch`     | yes                                   | -       |
| `train_rl.auto_shape_resources`          | `config.rl_args.auto_shape_resources` | yes                             | -       |
| `train_rl._maybe_print_device_report`    | `config.device.maybe_print_device_report` | yes                       | -       |
| `train_rl._logging_monitor_loop`         | `config.log_setup.drain_log_queue` | yes                               | -       |
| `train_rl._spawn_monitor`                | `tools.monitor_launch.spawn_monitor_manager` | yes                   | -       |
| `train_rl._update_portfolio_state`       | `tools.run_state.update_portfolio_state` | yes                      | -       |
| `train_rl._write_run_states`             | `tools.run_state.write_run_state_files` | yes                      | -       |
| `train_rl._ensure_aux_files`             | removed (perf-dir state only)    | -                                    | yes     |

Shims/Deprecations: internal helpers moved; existing public CLI flags unaffected.

Risks/Migration: callers importing from old locations must switch to canonical modules; global run_state files are no longer updated.
- B) Charts: Agg backend; ≥5 PNG with placeholders; row counts printed; atomic PNGs.
- C) Eval: synthetic-first metrics; summary.json + portfolio_state.json; equity/drawdown charts (Agg).
- D) State: run_state.json + state_latest.json (atomic) after portfolio; events.lock optional.
- E) Lifecycle: deep_rl_last.zip updated each run; atomic best promotion; archive_best timestamped; corrupt checkpoint guards.
- F) CLI: allow-synth examples; help for tools; heavy deps moved inside main; standardized POSTRUN fields.
- Risks/Migration: do not json.load KB (JSONL). Do not follow symlinks for latest. Atomic I/O everywhere.
- Next: optional archive index CSV; WFA eval pack; CI smoke; SAC/TD3/TQC builders.

## 2025-09-24
- **Files**: `bot_trade/tools/monitor_manager.py`, `bot_trade/tools/export_run_charts.py`, `bot_trade/tools/eval_run.py`, `bot_trade/tools/kb_writer.py`, `bot_trade/train_rl.py`, `CHANGE_NOTES.md`
- **Rationale**: standardize debug/charts prints, strict KB schema with de-dup append, non-silent eval output, retrying post-run exports and latest guards.
- **Risks**: downstream parsers must handle new debug lines and KB fields; eval/output formats may require script updates.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/train_rl.py`; `python -m bot_trade.tools.monitor_manager --help`; `python -m bot_trade.tools.export_run_charts --help`; `python -m bot_trade.tools.eval_run --help`; generate synth data then run `python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --policy MlpPolicy --device cpu --n-envs 2 --n-steps 512 --batch-size 1024 --total-steps 2048 --net-arch "1024,512,256" --activation silu --vecnorm --headless --allow-synth --kb-file Knowlogy/kb.jsonl --data-dir data_ready`; `python -m bot_trade.tools.monitor_manager --symbol BTCUSDT --frame 1m --run-id latest`; `python -m bot_trade.tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest`.

## Developer Notes — 2025-09-24 (Delta Pack)
- Standardized [DEBUG_EXPORT] and [CHARTS] prints and added concise [EVAL] line.
- KB appends now enforce full schema, defaulting missing fields and skipping duplicates.
- Post-run chart export retries once; risk flags saved as `risk_flags.png`.
- `latest` resolution prints `[LATEST] none` with exit code 2 when no runs exist.
- Legacy flags like `--no-wait` and `--debug-export` remain accepted but are no-ops.
- Risks/Migration: tools now emit extra lines; parsers expecting silence must adapt. KB records include `rows_callbacks` and may change ordering.
- Next: track KB size growth, richer eval metrics, and automation around smoke tests.

## 2025-09-24
- **Files**: `bot_trade/tools/export_run_charts.py`, `bot_trade/tools/monitor_manager.py`, `bot_trade/tools/eval_run.py`, `bot_trade/train_rl.py`, `DEV_NOTES.md`, `CHANGE_NOTES.md`
- **Rationale**: finalize Delta Pack with standardized debug/charts lines, headless notices, atomic PNG params, and non-silent eval.
- **Risks**: scripts parsing old outputs may need updates; additional prints could affect log consumers.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/train_rl.py`; `python -m bot_trade.tools.monitor_manager --help`; `python -m bot_trade.tools.export_run_charts --help`; `python -m bot_trade.tools.eval_run --help`; run synthetic training and verify [DEBUG_EXPORT]/[CHARTS]/[EVAL] lines.

## 2025-09-24
- **Files**: `bot_trade/tools/atomic_io.py`, `bot_trade/tools/latest.py`, `bot_trade/tools/export_charts.py`, `bot_trade/tools/eval_run.py`, `bot_trade/tools/monitor_manager.py`, `bot_trade/tools/export_run_charts.py`, `bot_trade/tools/evaluate_model.py`, `bot_trade/tools/kb_writer.py`, `bot_trade/config/update_manager.py`, `bot_trade/train_rl.py`, `DEV_NOTES.md`, `CHANGE_NOTES.md`
- **Rationale**: foundation cleanup—single KB appender with atomic helpers, canonical exporter/evaluator with shims, standardized prints and latest guards.
- **Risks**: consumers parsing old outputs must handle new lines and deprecation warnings; KB writes now strictly JSONL.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/train_rl.py`; CLI `--help` for monitor/export/eval; synthetic run verifying [DEBUG_EXPORT] → [CHARTS] → [POSTRUN] and `[EVAL]` lines; `--run-id latest` diagnostics.
## 2025-09-25
- **Files**: `bot_trade/tools/atomic_io.py`, `bot_trade/tools/runctx.py`, `DEV_NOTES.md`, `CHANGE_NOTES.md`
- **Rationale**: unify atomic writers under `atomic_io`; PNG writer defaults to `dpi=120`; `runctx.atomic_write_json` becomes a shim.
- **Risks**: downstream imports should migrate to canonical helpers; increased PNG dpi may slightly enlarge files.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/train_rl.py`; synthetic training run with `--allow-synth` verifying `[DEBUG_EXPORT]`, `[CHARTS]`, `[POSTRUN]`, `[EVAL]` lines.

## Developer Notes — 2025-09-25 (Foundation + Split)
- Canonical: `bot_trade.tools.atomic_io.write_json/append_jsonl/write_png`.
- Shimmed: `bot_trade.tools.runctx.atomic_write_json` → `atomic_io.write_json`.
- Logic moved: runctx.atomic_write_json → atomic_io.write_json.
- De-dup Map: atomic JSON writer consolidated; no duplicate text writer.
- Standardized outputs & latest guards remain.
- Risks/Migration: update imports from `runctx.atomic_write_json`; PNGs now saved at 120 DPI.
- Next: prune deprecated helpers and monitor KB size.
