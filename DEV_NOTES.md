# Developer Notes

## Module map

- `bot_trade.config` – environment logic, risk management, RL builders and
  callbacks, writer utilities and logging setup.
- `bot_trade.signals` – entry, reward, danger and recovery signal helpers.
- `bot_trade.ai_core` – knowledge base, verification, simulation and
  self-improvement engines.
- `bot_trade.tools` – data conversion, ETL utilities and assorted helpers.
- CLI entrypoints like `train_rl` and `run_multi_gpu` reside directly under
  `bot_trade` and are invoked via `python -m bot_trade.<name>`.

## Logging & shutdown

On Windows a single `QueueListener` must run in the main process while workers
use `QueueHandler` only. Shutdown order:
1. close environments and workers
2. stop the listener
3. `logging.shutdown()`
4. close writers

## Artifact schedule

Artifacts (CSV, JSON and visuals) are written every `artifact_every_steps` steps
and at start and end of training. Files:
- `logs/benchmarks.log`, `logs/errors.log`, `logs/risk.log`, `logs/reward.log`
- `results/deep_rl_evaluation.csv`, `results/deep_rl_trades.csv`,
  `results/step_log.csv`, `results/train_log.csv`
- `results/portfolio_state.json`

## Resume snapshot

`memory/memory.json` stores the full training state including RNG, env pointer,
open position and equity. Use `--auto-resume` to continue from the latest
snapshot.

## Knowledge base

The central file `memory/knowledge_base_full.json` is read before training and
updated at each artifact tick with EMA statistics and session information.

## Contribution workflow

- Keep commits small and focused.
- Open pull requests with a concise description of changes and test steps.
- After any code change add an entry to `CHANGE_NOTES.md` with date, files
  touched, rationale, risks and test steps.

## Running tools directly vs module mode

Project helper scripts live under the `tools/` package. When executing these
scripts it is recommended to run them as modules from the project root, e.g.

```bash
python -m tools.export_charts --help
```

This ensures that the project root is on `sys.path` and avoids `ImportError`
issues. For convenience the scripts also include a small `sys.path` fix-up so
that direct execution (`python tools/export_charts.py`) still works if needed.

## Developer Notes — 2025-09-24 (Delta Pack — merged)
- Standardized `[DEBUG_EXPORT]` and `[CHARTS]` lines with zero-defaulted counts.
- Headless mode announced on import; PNGs saved atomically with explicit format and DPI and exports retry once.
- Knowledge base writes enforce full schema and skip duplicate run IDs.
- Eval CLI emits a single `[EVAL]` line and guards `latest` lookups with exit code 2.
- Legacy flags remain accepted as no-ops; CLIs are the supported interface for tests.
- Migration: consumers should handle new debug lines and extended KB fields when parsing outputs.

## Developer Notes — 2025-09-24 (Foundation Cleanup)
- Deprecated `export_run_charts` and `evaluate_model` now shim to canonical tools.
- `export_charts` and `eval_run` own exports/eval; strict KB schema enforced via single appender.
- Added atomic I/O helpers and unified `[DEBUG_EXPORT]`/`[CHARTS]` prints with `latest` guards.
- Risks/Migration: downstream parsers should tolerate new lines and deprecated shim notices.
- Next: monitor KB growth and expand eval metrics.

## Developer Notes — 2025-09-01 (train_rl split)
- Extracted arg validation and auto resource shaping to `config.rl_args`.
- Device report printer lives in `config.device`.
- Logging queue drain and monitor spawning moved to `config.log_setup` and `tools.monitor_launch`.
- Run-state and portfolio writers consolidated in `tools.run_state`; train orchestrator now slimmer.
- Risks: callers using removed internals must import from new modules. Legacy global run_state files deprecated.
- Next: finish lifecycle callback consolidation and monitor remaining shims.
