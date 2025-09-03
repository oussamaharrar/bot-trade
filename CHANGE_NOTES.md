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

## 2025-09-03
- **Files**: `bot_trade/env/trading_env_continuous.py`, `bot_trade/config/rl_builders.py`, `bot_trade/config/rl_args.py`, `bot_trade/train_rl.py`, `bot_trade/tools/kb_writer.py`, `CHANGE_NOTES.md`
- **Rationale**: introduce optional continuous trading environment and lazy registry supporting SAC/TD3/TQC with robust policy kwargs.
- **Risks**: off-policy algorithms depend on correct Box spaces and sb3-contrib; new env may expose untested edge cases.
- **Test Steps**: `python -m py_compile bot_trade/**/*.py bot_trade/*.py`; `python -m bot_trade.train_rl --help | grep -E "algorithm|PPO|SAC|TD3|TQC|continuous-env"`; discrete guard `python -m bot_trade.train_rl --algorithm SAC --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 32 --batch-size 64 --total-steps 64 --headless --allow-synth --data-dir data_ready --no-monitor`; continuous smoke `python -m bot_trade.train_rl --algorithm SAC --continuous-env --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 64 --batch-size 64 --total-steps 256 --headless --allow-synth --data-dir data_ready --no-monitor`.

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

## 2025-09-26
- **Files**: `bot_trade/tools/_headless.py`, `bot_trade/tools/monitor_manager.py`, `bot_trade/tools/export_charts.py`, `bot_trade/tools/eval_run.py`, `bot_trade/train_rl.py`, `CHANGE_NOTES.md`
- **Rationale**: single headless notice helper, strict latest guards, and standardized debug/chart outputs.
- **Risks**: missing helper calls may skip Agg backend; log parsers must handle new `[HEADLESS]` lines.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/train_rl.py`; synthetic run verifying `[HEADLESS]`, `[DEBUG_EXPORT]`, `[CHARTS]`, `[POSTRUN]`, `[EVAL]` lines; `--run-id latest` diagnostics.

## Developer Notes — 2025-09-26 10:00 (Phase 1: Finish Foundations)
- What: Introduced `ensure_headless_once` and enforced `[LATEST] none` exit guards with zero-defaulted row counts.
- Why: avoid duplicate headless prints and guard missing runs across CLIs.
- Risks: callers that skip the helper may display GUI backends; consumers must handle the new `[HEADLESS]` line.
- Migration: call `ensure_headless_once` in new CLIs before plotting; update log parsers for the headless notice.
- Next: expand smoke coverage and track chart file sizes.

## 2025-09-26
- **Files**: `bot_trade/config/rl_builders.py`, `bot_trade/train_rl.py`, `bot_trade/config/rl_args.py`, `bot_trade/config/rl_paths.py`, `bot_trade/tools/paths.py`, `bot_trade/config/config.yaml`, `DEV_NOTES.md`, `CHANGE_NOTES.md`
- **Rationale**: Phase 1: introduce SAC alongside PPO with algorithm switch and algorithm-scoped artefact directories.
- **Risks**: new directory layout may break existing scripts; SAC requires continuous action spaces.
- **Test Steps**: `python -m py_compile bot_trade/config/rl_builders.py bot_trade/train_rl.py bot_trade/config/rl_args.py bot_trade/config/rl_paths.py bot_trade/tools/paths.py`, `python -m bot_trade.train_rl --help`, `python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --total-steps 1 --allow-synth --no-monitor --headless`, `python -m bot_trade.train_rl --algorithm SAC --symbol BTCUSDT --frame 1m --total-steps 1 --allow-synth --no-monitor --headless --buffer-size 1000 --learning-starts 1 --train-freq 1 --gradient-steps 1 --batch-size 2 --ent-coef auto`

## 2025-09-01
- **Files**: `bot_trade/config/rl_builders.py`, `bot_trade/config/rl_args.py`, `bot_trade/config/rl_paths.py`, `bot_trade/tools/kb_writer.py`, `bot_trade/train_rl.py`, `DEV_NOTES.md`, `CHANGE_NOTES.md`
- **Rationale**: registry-based algorithm selection with SAC warm-start option, legacy path fallbacks, and algorithm metadata in KB and post-run lines.
- **Risks**: fallback detection may miss atypical layouts; warm-start assumes PPO checkpoints are compatible; TD3/TQC not yet implemented.
- **Test Steps**: `python -m py_compile bot_trade/config/rl_builders.py bot_trade/config/rl_args.py bot_trade/config/rl_paths.py bot_trade/tools/kb_writer.py bot_trade/train_rl.py`, `python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --total-steps 1 --n-envs 1 --n-steps 1 --batch-size 1 --headless --allow-synth`, `python -m bot_trade.train_rl --algorithm SAC --symbol BTCUSDT --frame 1m --total-steps 1 --n-envs 1 --n-steps 1 --batch-size 1 --headless --allow-synth; test $? -eq 1`, run a small script calling `build_algorithm('SAC', Pendulum-v1)` to verify warm-start skip message.

## Developer Notes — 2025-09-02T00:02:38Z (Phase 3)
- What: registry builder with SAC/TD3/TQC entries, SAC action-space guard & override prints, optional warm-start, algorithm-scoped paths with legacy read-only fallback, POSTRUN/KB include algorithm, explicit exit codes.
- Why: enable multi-algorithm training and clear user feedback.
- Risks: path resolution may miss atypical layouts; warm-start assumes PPO checkpoints compatible.
- Migration: prefer new algo-scoped helpers (`last_agent`, `best_agent`); legacy paths remain read-only.
- Next: implement TD3/TQC algorithms.
## Developer Notes — 2025-09-02T00:28:13Z (Phase 3)
- What: finalized algorithm registry with PPO/SAC and TD3/TQC stubs; SAC now guards Box action spaces, prints true overrides, and can warm-start from PPO checkpoints.
- Why: enable multi-algorithm flexibility with clear user feedback and safe defaults.
- Risks: warm-start assumes compatible PPO feature extractors; path scoping may miss custom layouts.
- Migration: use new --algorithm flag and algo-scoped helpers; legacy paths remain read-only fallbacks.
- Next: implement full TD3/TQC support and expand SAC validation.

## 2025-09-27
- **Files**: `bot_trade/eval/*`, `bot_trade/tools/eval_run.py`, `bot_trade/tools/monitor_manager.py`, `bot_trade/tools/atomic_io.py`, `CHANGE_NOTES.md`
- **Rationale**: add evaluation suite with metrics, walk-forward analysis, and tearsheet generation.
- **Risks**: optional PDF generation may fail if dependencies missing; WFA requires sufficient data for splits.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/eval/*.py bot_trade/train_rl.py`; run synthetic training then `python -m bot_trade.tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest --wfa-splits 4 --wfa-embargo 0.02 --tearsheet`.

## Developer Notes — 2025-09-27T00:00:00Z (Phase 2 — Evaluation Suite)
- Added eval package: metrics.py, walk_forward.py (purged/embargoed), tearsheet.py (HTML/PDF).
- Extended eval_run summary with sortino/calmar/max_drawdown/turnover/slippage_proxy.
- Added WFA CLI & Tearsheet CLI with headless Agg and atomic writes; placeholders on missing data.
- Latest guards: [LATEST] none + exit=2 for walk_forward/tearsheet when no runs.
- No breaking changes to existing flags/prints; heavy deps imported inside main().

## Developer Notes — 2025-09-27T12:34:56Z (Phase 2 merge)
- Numeric-stability guards return None on invalid ratios.
- BOT_REPORTS_DIR can override reports root; paths resolved to absolutes.
- Latest guards for walk_forward/tearsheet print `[LATEST] none` with exit code 2.
- Tearsheet emits NO DATA placeholders and writes HTML/PDF atomically.
- Risks: consumers must handle None metrics and env overrides.
- Test Steps: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/eval/*.py bot_trade/train_rl.py`; synthetic run then `python -m bot_trade.tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest --wfa-splits 4 --wfa-embargo 0.02 --tearsheet`.

## 2025-09-28
- **Files**: `bot_trade/config/rl_builders.py`, `bot_trade/train_rl.py`, `DEV_NOTES.md`, `CHANGE_NOTES.md`
- **Rationale**: move SAC warm-start into builder with prioritized PPO checkpoint search and clearer Box-space guard logging.
- **Risks**: warm-start resolver may miss unconventional layouts; assumes compatible feature extractors.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/train_rl.py`; `python -m bot_trade.train_rl --help`; PPO and SAC smoke runs verifying warm-start skip message.

## Developer Notes — 2025-09-02T02:17:32Z (Phase 4)
- What: added sweep CLI (grid parsing, summary outputs, failure handling), CI smoke workflow (standardized log and artifact checks, single headless notice, latest guards, dev checks), and atomic HTML/PDF writers used in tearsheet with no path/flag regressions.
- Why: enable quick CPU experiments and catch regressions early while ensuring robust artifact writes.
- Risks: sweeps may generate large artifacts; CI runtime may grow; padded HTML/PDF could hide empty content.
- Migration: use `python -m bot_trade.tools.sweep` for experiments and run `python -m bot_trade.tools.dev_checks`; existing paths and flags unchanged.
- Next: broaden algorithm coverage in sweeps and expand smoke tests.

## 2025-09-29
- **Files**: `bot_trade/env/execution_sim.py`, `bot_trade/config/env_trading.py`, `bot_trade/config/risk_manager.py`, `bot_trade/config/rl_callbacks.py`, `bot_trade/config/rl_args.py`, `bot_trade/tools/gen_synth_data.py`, `bot_trade/tools/kb_writer.py`, `bot_trade/train_rl.py`, `bot_trade/config/config.yaml`
- **Rationale**: introduce pluggable execution simulator with slippage/latency models, add circuit-breaker risk guards and exposure caps, surface execution metrics via logs and CLI overrides, and extend knowledge base schema.
- **Risks**: simplified models may not capture real market behavior; new risk flags require downstream parsers to handle additional columns.
- **Test Steps**: `python -m py_compile bot_trade/env/execution_sim.py bot_trade/config/env_trading.py bot_trade/config/risk_manager.py bot_trade/config/rl_callbacks.py bot_trade/config/rl_args.py bot_trade/tools/gen_synth_data.py bot_trade/tools/kb_writer.py bot_trade/train_rl.py`; `python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready`; `python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 256 --batch-size 256 --total-steps 512 --headless --allow-synth --data-dir data_ready --slippage-model vol_aware --slippage-params '{"k":0.5,"vol_window":50}' --latency-ms 40 --max-spread-bp 20 --allow-partial-exec`; `python -m bot_trade.tools.monitor_manager --symbol BTCUSDT --frame 1m --run-id latest --tearsheet`

## 2025-09-30
- **Files**: `bot_trade/config/strategy_failure.py`, `bot_trade/config/stratigy_failure.py`, `bot_trade/config/env_trading.py`, `bot_trade/config/rl_callbacks.py`, `bot_trade/config/rl_args.py`, `bot_trade/config/config.yaml`, `bot_trade/train_rl.py`, `bot_trade/tools/export_charts.py`, `CHANGE_NOTES.md`, `DEV_NOTES.md`
- **Rationale**: consolidate strategy failure modules and add safety policy with clamps, logs, chart export, and CLI flags.
- **Risks**: aggressive clamps may reduce trading activity; additional callbacks add slight overhead.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/strat/*.py bot_trade/env/*.py bot_trade/train_rl.py`; `python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready`; `python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 64 --batch-size 64 --total-steps 128 --headless --allow-synth --data-dir data_ready`

## Developer Notes — 2025-09-02 03:41:34Z
- **What**: execution simulator models with unified params; structured risk circuit breakers and logs; CLI overrides for slippage/latency/partial/max-spread; evaluation turnover & slippage proxies; headless prints and latest guards unified; POSTRUN/KB enriched with algorithm and eval_max_drawdown.
- **Why**: prepare pluggable execution and risk safety foundation with consistent reporting.
- **Risks**: simplified execution may misprice fills; downstream tools must handle new risk_log order and diagnostic line.
- **Migration Steps**: consume new CLI flags, update parsers for risk_log columns, accept optional `[RISK_DIAG]` output.
- **Next Actions**: add regime detection interfaces and wire multi-algo registry.

## 2025-09-02
- **Files**: `bot_trade/strat/regime.py`, `bot_trade/strat/adaptive_controller.py`, `bot_trade/config/config.yaml`, `bot_trade/config/rl_args.py`, `bot_trade/config/rl_callbacks.py`, `bot_trade/train_rl.py`, `DEV_NOTES.md`
- **Rationale**: add regime detection and adaptive reward/risk controller, integrate execution hardening logs and charts.
- **Risks**: misconfigured thresholds may return unknown regimes; extra logging and plotting add minor overhead.
- **Test Steps**: `python -m py_compile bot_trade/strat/regime.py bot_trade/strat/adaptive_controller.py bot_trade/config/rl_args.py bot_trade/config/rl_callbacks.py bot_trade/train_rl.py`; `python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 128 --batch-size 128 --total-steps 256 --headless --allow-synth --data-dir data_ready --regime-aware --regime-log`; `python -m bot_trade.tools.monitor_manager --symbol BTCUSDT --frame 1m --run-id latest --tearsheet`; `python -m bot_trade.tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest --tearsheet`; `python -m bot_trade.tools.monitor_manager --symbol FAKE --frame 1m --run-id latest`.

## 2025-09-02
- **Files**: bot_trade/strat/adaptive_controller.py, bot_trade/config/env_trading.py, bot_trade/strat/strategy_features.py, bot_trade/strat/__init__.py, DEV_NOTES.md, CHANGE_NOTES.md
- **Rationale**: clamp adaptive controller parameters with baseline restoration, expose regime in env info, and introduce strategy feature registry stub.
- **Risks**: mis-specified bounds may be silently clipped; registry currently unused.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/strat/*.py bot_trade/train_rl.py`; `python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready`; `python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 128 --batch-size 128 --total-steps 256 --headless --allow-synth --data-dir data_ready --regime-aware --regime-log --regime-window 5`; `python -m bot_trade.tools.monitor_manager --symbol BTCUSDT --frame 1m --run-id latest --tearsheet`; `python -m bot_trade.tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest --tearsheet`; `python -m bot_trade.tools.monitor_manager --symbol FAKE --frame 1m --run-id latest`.

## 2025-10-01
- **Files**: `bot_trade/config/strategy_features.py`, `bot_trade/config/rl_callbacks.py`, `bot_trade/config/strategy_failure.py`, `bot_trade/config/stratigy_failure.py`, `DEV_NOTES.md`, `CHANGE_NOTES.md`
- **Rationale**: merge legacy strategy failure modules into `strategy_features.py` and update callbacks to use the unified implementation.
- **Risks**: external imports from removed modules may fail until switched to `strategy_features`.
- **Test Steps**: `python -m py_compile bot_trade/config/strategy_features.py bot_trade/config/rl_callbacks.py bot_trade/train_rl.py`; `python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready`; `python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 32 --batch-size 32 --total-steps 64 --headless --allow-synth --data-dir data_ready`

## 2025-10-02
- **Files**: `bot_trade/config/config.yaml`, `bot_trade/strat/adaptive_controller.py`, `bot_trade/config/strategy_features.py`, `bot_trade/tools/export_charts.py`, `bot_trade/tools/kb_writer.py`, `bot_trade/DEV_NOTES.md`, `CHANGE_NOTES.md`
- **Rationale**: add regime weight limits with clamped controller, log strategy_failure as risk flag, export regimes chart, and extend KB schema with regime summary.
- **Risks**: misconfigured limits may over-clamp adjustments; charts rely on adaptive_log presence.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/strat/*.py bot_trade/train_rl.py`; `python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready`; `python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 32 --batch-size 32 --total-steps 64 --headless --allow-synth --data-dir data_ready --regime-aware --regime-log --regime-window 5`; `python -m bot_trade.tools.monitor_manager --symbol BTCUSDT --frame 1m --run-id latest --tearsheet`; `python -m bot_trade.tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest --tearsheet`; `python -m bot_trade.tools.monitor_manager --symbol FAKE --frame 1m --run-id latest`

## 2025-09-02T06:03:52Z
- **Files**: `bot_trade/eval/metrics.py`, `bot_trade/eval/walk_forward.py`, `bot_trade/eval/tearsheet.py`, `bot_trade/tools/eval_run.py`, `bot_trade/tools/monitor_manager.py`, `DEV_NOTES.md`, `CHANGE_NOTES.md`
- **Rationale**: Phase 2 — Evaluation Suite: added robust metrics (Sharpe, Sortino, Calmar, MaxDD, turnover, slippage proxy, win rate, avg trade PnL), purged/embargoed walk-forward analysis, atomic HTML/PDF tearsheet, numeric-stability guards, BOT_REPORTS_DIR override, latest-guard exit=2, and single headless notice. Migration: parsers should handle null ratios when windows are unstable.
- **Risks**: WFA may fail on small datasets; PDF generation relies on optional dependencies.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/eval/*.py bot_trade/train_rl.py`; `python -m bot_trade.tools.eval_run --help`; `python -m bot_trade.eval.walk_forward --help`; `python -m bot_trade.eval.tearsheet --help`; `python -m bot_trade.tools.monitor_manager --help`; `python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready`; `python -m bot_trade.train_rl --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 64 --batch-size 64 --total-steps 128 --headless --allow-synth --data-dir data_ready`; `python -m bot_trade.tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest --wfa-splits 4 --wfa-embargo 0.02 --tearsheet`

## 2025-10-03
- **Files**: `bot_trade/config/rl_args.py`, `bot_trade/config/rl_builders.py`, `bot_trade/train_rl.py`, `DEV_NOTES.md`, `CHANGE_NOTES.md`
- **Rationale**: Phase 3 — Multi-Algo Builders with registry dispatch, SAC Box-space guard, warm-start search order, and explicit CLI override tracking.
- **Risks**: SAC requires continuous action spaces; TD3/TQC remain stubs and exit early.
- **Test Steps**: `python -m py_compile bot_trade/config/*.py bot_trade/tools/*.py bot_trade/train_rl.py`; `python -m bot_trade.train_rl --help`; `python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready`; `python -m bot_trade.train_rl --algorithm PPO --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 64 --batch-size 64 --total-steps 128 --headless --allow-synth --data-dir data_ready`; `python -m bot_trade.train_rl --algorithm SAC --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 2 --batch-size 2 --total-steps 2 --headless --allow-synth --data-dir data_ready ; test $? -eq 1`; `python - <<'PY'\nfrom types import SimpleNamespace\nimport gymnasium as gym\nfrom bot_trade.config.rl_builders import build_algorithm\nenv = gym.make('Pendulum-v1')\nargs = SimpleNamespace(symbol='TEST', frame='1m', seed=0, device_str='cpu',\n                       sac_warmstart_from_ppo=True, warmstart_from_ppo=None,\n                       sb3_verbose=0, policy_kwargs={}, learning_rate=3e-4,\n                       buffer_size=None, learning_starts=None, train_freq=None,\n                       gradient_steps=None, batch_size=None, tau=None,\n                       ent_coef='auto', sac_gamma=0.99, _specified=set())\nmodel = build_algorithm('SAC', env, args, {})\nprint(type(model).__name__)\nPY`; `python -m bot_trade.train_rl --algorithm TD3 --symbol BTCUSDT --frame 1m --total-steps 1 --headless --allow-synth --data-dir data_ready ; test $? -eq 2`; `python -m bot_trade.train_rl --algorithm TQC --symbol BTCUSDT --frame 1m --total-steps 1 --headless --allow-synth --data-dir data_ready ; test $? -eq 2`


## 2025-10-04
- **Files**: `bot_trade/tools/sweep.py`, `bot_trade/tools/dev_checks.py`, `.github/workflows/smoke.yml`, `DEV_NOTES.md`, `CHANGE_NOTES.md`
- **Rationale**: add CPU sweeper with ranked summaries and runs log, extend dev checks for log order, chart sizes and KB metadata, and introduce new CI smoke workflow covering synth data generation, training, eval, sweep and checks.
- **Risks**: sweeps may still take time; workflow assumes synthetic data availability.
- **Test Steps**: `python -m py_compile $(git ls-files 'bot_trade/config/*.py' 'bot_trade/tools/*.py' 'bot_trade/eval/*.py' 'bot_trade/strat/*.py' 'bot_trade/env/*.py' 'bot_trade/train_rl.py')`; `python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready`; `python -m bot_trade.train_rl --algorithm PPO --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 64 --batch-size 64 --total-steps 128 --headless --allow-synth --data-dir data_ready`; `python -m bot_trade.tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest`; `python -m bot_trade.tools.sweep --symbol BTCUSDT --frame 1m --out-dir sweeps/out_test --algo-grid "PPO,SAC" --param-grid "n_steps=[64];batch_size=[64]" --trials 1 --allow-synth --headless --data-dir data_ready`; `python -m bot_trade.tools.dev_checks`

## 2025-10-05
- **Files**: `CHANGE_NOTES.md`
- **Rationale**: append consolidated developer notes outlining final contract and invariants.
- **Risks**: none; documentation only.
- **Test Steps**: read appended Developer Notes section.

## Developer Notes — 2025-10-05T00:00:00Z
- Ordered prints: `[DEBUG_EXPORT]` → `[CHARTS]` → `[POSTRUN]` with zero defaults for missing rows.
- Single headless notice per CLI via `ensure_headless_once`.
- Latest guard: `--run-id latest` prints `[LATEST] none` and exits 2 without traceback.
- Risk artifacts: CSV schema `risk_flag,flag_reason,value,threshold,ts` and mandatory `risk_flags.png` (≥1 KB); smoke runs emit `[RISK_DIAG] no_flags_fired` when clean.
- Eval numeric guards return `None` for empty/degenerate windows; BOT_REPORTS_DIR overrides output destinations.
- Registry-based algorithm builders with CLI override tracking; SAC enforces Box spaces and optional PPO warm-start search order (best → last → legacy).
- Regime controller logs `adaptive_log.jsonl`, exports `regimes.png`, and tracks active regime distribution in KB.
- Sweeper and dev checks ensure chart sizes, KB fields, and provide CI coverage.
- Knowledge base appends enforce full schema, skip duplicates, and record algorithm metadata.
- Next: implement full TD3/TQC support and expand regime metrics.

## 2025-10-06
- **Files**: `bot_trade/tools/_headless.py`, `bot_trade/tools/export_charts.py`, `bot_trade/tools/eval_run.py`, `bot_trade/tools/monitor_manager.py`, `DEV_NOTES.md`, `CHANGE_NOTES.md`
- **Rationale**: ensure single `[HEADLESS] backend=Agg` notice per CLI, expose `eval_max_drawdown` on `[POSTRUN]`, and harden `--run-id latest` guards across export/eval/monitor tools.
- **Risks**: base path overrides may diverge from `BOT_REPORTS_DIR`.
- **Test Steps**: `python -m py_compile $(git ls-files '*.py')`; synthetic run `python -m bot_trade.tools.gen_synth_data --symbol BTCUSDT --frame 1m --out data_ready`; `python -m bot_trade.train_rl --algorithm PPO --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 32 --batch-size 32 --total-steps 64 --headless --allow-synth --data-dir data_ready`; missing-run guards `python -m bot_trade.tools.monitor_manager --symbol FAKE --frame 1m --run-id latest; test $? -eq 2 && echo EXIT_CODE_2`; same for `export_charts` and `eval_run`; evaluation `python -m bot_trade.tools.eval_run --symbol BTCUSDT --frame 1m --run-id latest`.

## 2025-10-07
- **Files**: `bot_trade/env/trading_env_continuous.py`, `bot_trade/config/rl_args.py`, `bot_trade/config/rl_builders.py`, `bot_trade/tools/kb_writer.py`, `DEV_NOTES.md`, `CHANGE_NOTES.md`
- **Rationale**: single `[ENV]` notice for continuous env, pass-through `policy_kwargs` with JSON parsing and unused-key warnings, standardized TQC guard, KB newline assurance and condensed `policy_kwargs`, and updated `--continuous-env` help.
- **Risks**: unused key detection is shallow; KB newline fix may touch large files.
- **Test Steps**: `python -m py_compile bot_trade/env/trading_env_continuous.py bot_trade/config/rl_args.py bot_trade/config/rl_builders.py bot_trade/tools/kb_writer.py`; discrete guard `python -m bot_trade.train_rl --algorithm SAC --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 32 --batch-size 64 --total-steps 64 --headless --allow-synth --data-dir data_ready --no-monitor`; continuous smoke `python -m bot_trade.train_rl --algorithm SAC --continuous-env --symbol BTCUSDT --frame 1m --device cpu --n-envs 1 --n-steps 64 --batch-size 64 --total-steps 256 --headless --allow-synth --data-dir data_ready --no-monitor`; policy_kwargs big nets snippet; TQC dependency guard snippet.

## Developer Notes — 2025-09-02 22:30:40 UTC
- headless notice confined to module `main` ensuring a single `[HEADLESS] backend=Agg` line per CLI.
- strict `[LATEST] none` guard (exit code 2) retained across export/monitor/eval tools.
- `[POSTRUN]` lines now include `algorithm`, `eval_win_rate`, `eval_sharpe`, and `eval_max_drawdown`.
- evaluation suite refined: metrics, walk-forward, and tearsheet utilities integrated through `tools.eval_run`.
- RL algorithm registry uses lazy imports; off-policy stubs validate Box spaces and emit clear skip notices.
- migration: callers continue using `--algorithm` for `train_rl`; TD3/TQC remain stubs exiting early.

## 2025-09-02
- **Files**: `bot_trade/eval/metrics.py`, `bot_trade/eval/equity.py`, `bot_trade/eval/walk_forward.py`, `bot_trade/tools/eval_run.py`, `bot_trade/tools/kb_writer.py`, `bot_trade/train_rl.py`, `CHANGE_NOTES.md`, `DEV_NOTES.md`
- **Rationale**: expand evaluation suite with equity helpers, aggregated metrics, walk-forward summaries, KB schema updates and canonical chart/eval prints.
- **Risks**: walk-forward may yield empty placeholders on tiny datasets; downstream consumers must handle new KB fields.
- **Test Steps**: `python -m py_compile $(git ls-files 'bot_trade/**/*.py' 'bot_trade/*.py')`; synthetic run + eval sequence as in verification snippet.

## Developer Notes — 2025-09-02 22:47:38 UTC
- headless single-notice upheld across all CLIs.
- latest guards emit `[LATEST] none` with exit 2 when runs missing.
- `[POSTRUN]` lines report algorithm, win rate, sharpe and max drawdown.
- evaluation modules: `equity.build_equity_drawdown`, `metrics.compute_all`, walk-forward summaries.
- migration: KB entries now include `sortino`, `calmar` and `images_list`; update parsers accordingly.

## 2025-09-02
- **Files**: bot_trade/tools/export_charts.py, bot_trade/tools/monitor_manager.py, bot_trade/tools/eval_run.py, bot_trade/eval/metrics.py, bot_trade/eval/tearsheet.py, DEV_NOTES.md, CHANGE_NOTES.md
- **Rationale**: unify headless prints, enforce latest guards, consolidate evaluation suite and enrich knowledge base entries.
- **Risks**: downstream parsers may rely on prior metric ordering or KB fields.
- **Test Steps**: `python -m py_compile bot_trade/**/*.py bot_trade/*.py`; run synth training and evaluation commands from quick verification.

## Developer Notes — 2025-09-02 23:16:24 UTC
- Unified headless single-notice in CLIs
- Strict latest guards (exit=2)
- Canonical [EVAL] then [POSTRUN] (includes algorithm & eval_max_drawdown)
- Evaluation suite: metrics/equity/walk_forward/tearsheet
- KB enrichment: metrics+images, duplicate run_id guard
- Placeholders for sparse data; all outputs atomic
- Migration notes for older scripts parsing POSTRUN

## Developer Notes — 2025-09-03 00:22:39 UTC
- Added continuous trading environment behind --continuous-env (Box action space)
- Implemented lazy registry for PPO/SAC/TD3/TQC with strict Box guard and TQC dependency guard
- Enabled PPO→SAC warm-start (encoder-only, shape-checked) with single-line status
- Fixed policy_kwargs passthrough; support dict or JSON string; recorded condensed policy_kwargs in algo_meta
- Kept logging contracts, latest guard, and Evaluation Suite behavior unchanged; all outputs atomic

## Developer Notes — 2025-09-03 01:15:11 UTC
- What: Added YAML-driven rewards registry with clamps, regime detector with JSONL logging, adaptive controller applying reward/risk deltas, regimes.png exports, dev_checks updates, KB log counts, and CLI flags for reward/adaptive specs.
- Why: enable configurable reward shaping and regime-aware risk adjustments while preserving existing interfaces.
- Risks: misconfigured specs may yield silent clamping or no adaptive triggers; reward registry not yet fully integrated with environment.
- Migration: provide --reward-spec and --adaptive-spec paths (defaults available), parse regime/adaptive log counts in KB consumers.
- Next Actions: expand reward terms, tighten regime thresholds, and link registry outputs directly to env rewards.

## Developer Notes — 2025-09-03 01:42:15 UTC
- What: Debounced `[ADAPT]` logging per `{regime, window_id}`, seeded `RegimeDetector`, sanitized reward term handling, nested KB log counts, and enforced `regimes.png` DPI/size checks.
- Why: reduce duplicate adaptive prints, ensure deterministic regime analysis and numeric stability, and satisfy artifact requirements.
- Risks: adaptive deltas skipped if controller invoked repeatedly within same window; dev checks now fail on low-DPI regime charts.
- Migration: consumers must read `logs.regime_log_count`/`logs.adaptive_log_count` instead of old flat fields.
- Next Actions: monitor regime transitions in extended runs and refine detector thresholds.
## Developer Notes — 2025-09-03 02:02:13 UTC
- What: Removed redundant `[ADAPT_CHARTS]` print from `train_rl.py` to keep regimes chart notice centralized in exporters.
- Why: avoid duplicate adaptive chart logs and maintain single-source diagnostics.
- Risks: if exporters are skipped, regimes chart generation might lack explicit notice.
- Migration: none; workflows already call `export_charts` for chart diagnostics.
- Next Actions: watch for future consolidation of adaptive outputs across CLIs.

## Developer Notes — 2025-09-03 02:14:59 UTC
- What: Added ai_core skeleton (collectors, normalizers, enrichers, signal engine, bridge), exogenous signal reader, CLI flags, KB updates, and dev checks.
- Why: Enable optional ai_core pipeline generating external signals and safe ingestion without altering policy.
- Risks: Minimal signal set; future collectors may need concurrency handling; dry-run skips feature injection.
- Migration: Use --ai-core to enable pipeline; consume new ai_core section in KB; optional dummy signals via --emit-dummy-signals.
- Next Actions: Expand collectors, refine feature mapping, and connect signals to policy components.

## Developer Notes — 2025-09-03 02:32:43 UTC
- Implemented ai_core pipeline (collectors/normalizers/enrichers/signal_engine/bridge)
- Added dummy-signal recipe (MACD/RSI/Realized Volatility) with z-score normalization and guards
- Wrote atomic memory/Knowlogy/signals.jsonl (de-dupe, newline discipline)
- Added read_exogenous_signals and one-shot [AI_CORE] injection log
- Flags: --ai-core, --dry-run, --emit-dummy-signals (backward compatible)
- Extended KB with ai_core.signals_count and sources
- Kept logging contracts and Evaluation Suite behavior unchanged

## 2025-10-08
- **Files**: bot_trade/tools/atomic_io.py, bot_trade/tools/eval_run.py, bot_trade/tools/dev_checks.py, bot_trade/tools/sweep.py, .github/workflows/smoke.yml, DEV_NOTES.md, CHANGE_NOTES.md
- **Rationale**: add CPU-only sweep with grid/random modes and eval metrics, strengthen dev checks with chart DPI/size and ai_core signal assertions, log trailing newline fixes, and expand smoke CI to train/eval PPO+SAC and run random sweeps.
- **Risks**: sweep may prolong CI; dev checks depend on knowledge base availability.
- **Test Steps**: `python -m py_compile $(git ls-files 'bot_trade/**/*.py' 'bot_trade/*.py')`; `python -m bot_trade.tools.sweep --mode random --n-trials 4 --symbol BTCUSDT --frame 1m --algorithm SAC --continuous-env --headless --allow-synth --data-dir data_ready`; `python -m bot_trade.tools.dev_checks --symbol BTCUSDT --frame 1m --run-id latest`

## Developer Notes — 2025-09-03 03:20:01 UTC
- Added CPU sweeper (grid/random), atomic summaries (csv/md/jsonl), and [SWEEP] line
- Implemented warning thresholds with env overrides; added [SWEEP_WARN] markers
- Strengthened dev_checks (fields, charts DPI/size, ai_core signals) with summary [CHECKS]
- Improved atomic writers with [IO] fixed_trailing_newline
- Added smoke CI pipeline and artifacts
