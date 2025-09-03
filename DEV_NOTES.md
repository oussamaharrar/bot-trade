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

## Developer Notes — 2025-09-25 (Foundation + Split)
- Centralized atomic I/O helpers: `atomic_io.write_json`, `append_jsonl` and `write_png(dpi=120)` are the canon.
- `runctx.atomic_write_json` is now a deprecated shim delegating to `atomic_io.write_json`.
- De-dup Map: `runctx.atomic_write_json` → `atomic_io.write_json`.
- Standardized outputs and latest guards unchanged.
- Risks/Migration: modules relying on `runctx.atomic_write_json` should migrate; PNGs slightly larger due to higher DPI.
- Next: remove old text writer and audit remaining shims.

## Developer Notes — 2025-09-26 (SAC phase 1)
- What: Added SAC algorithm option with CLI flag/config, algorithm-scoped paths, warm-start utility, and TD3/TQC stubs.
- Why: prepare off-policy methods alongside PPO.
- Risks: new path layout may break scripts expecting previous structure; SAC requires continuous action spaces.
- Migration: specify `--algorithm` or set `rl.algorithm`; update path lookups for new layout.
- Next: implement TD3/TQC builders and evaluation upgrades.

## Developer Notes — 2025-09-01T23:27:54Z (SAC registry & warm-start)
- Added algorithm registry with TD3/TQC stubs raising friendly not-implemented exits.
- SAC builder enforces Box action space and logs CLI overrides; warm-start from PPO best with optional flag.
- Paths now fall back to legacy locations for reads; KB and [POSTRUN] lines include `algorithm`.
- Risks: legacy detection may miss exotic layouts; warm-start assumes compatible extractor.
- Migration: update consumers to read `algorithm` field; ensure PPO checkpoints exist before warm-start.
- Next: implement real TD3/TQC builders and expand SAC tests.

## Developer Notes — 2025-09-27T12:34:56Z (Phase 2 merge)
- What: added numeric-stability guards returning None for invalid Sharpe/Sortino/Calmar, honored BOT_REPORTS_DIR for absolute report paths, extended latest-run guards to walk-forward and tearsheet CLIs, and enforced NO DATA placeholders with atomic HTML/PDF writes.
- Why: ensure evaluation outputs remain sane, allow optional report relocation, and keep CLI behaviour predictable even without data.
- Risks: consumers must handle None metrics and misconfigured report paths; PDF generation still depends on optional dependencies.
- Migration: set BOT_REPORTS_DIR to redirect reports, handle None in metrics, and install `weasyprint` for PDF output.
- Next: broaden stability checks to additional metrics and integrate more comprehensive tests.

## Developer Notes — 2025-09-28T00:00:00Z (Phase 3 warm-start relocation)
- What: moved SAC warm-start into `build_sac` with prioritized PPO checkpoint resolver; `train_rl` no longer duplicates logic.
- Why: centralize algorithm-specific setup and keep CLI flags authoritative.
- Risks: resolver may miss unconventional layouts; warm-start assumes compatible feature extractors.
- Migration: none, existing flags unchanged.
- Next: implement full TD3/TQC support.

## Developer Notes — 2025-09-29 (Execution realism & safety)
- What: Added pluggable `ExecutionSim` with fixed/vol/depth-aware slippage, latency and partial fills; integrated circuit-breaker risk guards (spread jump, gap, liquidity, loss streak) plus exposure caps and execution diagnostics in step/risk logs. CLI overrides surface slippage model and latency; knowledge base and evaluation now track turnover and slippage_proxy.
- Why: decouple order execution from environment and provide configurable safety rails for more realistic simulations.
- Risks: simplified slippage models and circuit breakers may misrepresent live conditions; downstream tooling must handle new log columns.
- Migration: update parsing scripts to read `risk_flag` columns and optional execution metrics; adjust configs/CLIs if relying on previous fixed fills.
- Next: refine depth-aware modelling and expand evaluation around circuit-breaker effectiveness.

## Developer Notes — 2025-09-02T04:20:01Z (Regime-aware & hardening merge)
- What: added regime detection & adaptive reward/risk controller; enforced log order [DEBUG_EXPORT]→[CHARTS]→[POSTRUN], risk flag CSV schema, placeholder risk_flags.png & regimes.png, [RISK_DIAG] no_flags_fired, headless notice, [LATEST] none exit 2.
- Why: enable regime-aware training with robust execution safeguards.
- Risks: misconfigured thresholds may mute controller; extra plotting adds overhead.
- Migration: supply regime mappings in config, enable with --regime-aware; consumers should parse new adaptive_log.jsonl and KB.regime.
- Next Actions: refine regime classifier and expose more metrics.

## Developer Notes — 2025-09-02T04:51:04Z (Regime clamps & features registry)
- What: clamped adaptive controller with baseline restoration and env-exposed regime; added pluggable strategy feature registry stub.
- Why: prevent runaway risk settings and prep strategy plug-ins.
- Risks: mis-specified bounds may silently clamp; registry unused until plugins land.
- Migration: none; import from bot_trade.strat.strategy_features for future extensions.
- Next Actions: add concrete feature providers and expand regime metrics.


## Developer Notes — 2025-09-30T00:00:00Z
- What: consolidated strategy_failure modules, added safety policy with clamps, logs, and safety chart.
- Why: unify failure handling and escalate actions for risky regimes.
- Risks: policy may over-trigger and limit trading.
- Migration: import from `config.strategy_failure`; configure `strategy_failure` block and CLI overrides.
- Next Actions: integrate strategy selector and regime-based comparisons.

## Developer Notes — 2025-10-01 (Failure policy folded into features)
- What: merged `stratigy_failure.py` and `strategy_failure.py` into `strategy_features.py` and updated callbacks to consume the unified module.
- Why: remove duplicate files and centralize strategy utilities.
- Risks: code still importing `config.strategy_failure` must switch to `config.strategy_features`.
- Migration: replace imports with `from bot_trade.config import strategy_features`.
  - Next Actions: watch for edge cases in merged failure logic.

## Developer Notes — 2025-10-02T00:00:00Z (Regime controller & safety)
- What: added regime detection with adaptive reward/risk controller, single headless notice helper, latest-run guards, strategy_failure as risk flag, and ensured regimes.png/adaptive_log.jsonl artifacts.
- Why: tune behaviour per market regime, avoid duplicate CLI prints, surface missing runs, keep risk logging invariant, and track regime adjustments.
- Risks: misconfigured mappings may over-clamp values or miss logging; regime chart relies on adaptive_log presence.
- Migration: supply `regime` config with thresholds, mappings, and weight/bound limits; parse `adaptive_log.jsonl` and KB `regime` field; expect `[LATEST] none` and single `[HEADLESS]` lines.
- Next Actions: expand regime taxonomy and integrate distribution into analysis tools.

## Developer Notes — 2025-09-02T06:03:52Z (Phase 2 — Evaluation Suite)
- What: implemented equity-from-rewards curve, purged/embargoed walk-forward with aggregate keys, optional PDF tearsheet with trades/risk/regime panels, and PDF passthrough flags in eval_run/monitor_manager with embargo clamping.
- Why: finalize evaluation suite with robust metrics and artifacts.
- Risks: WFA needs sufficient data; PDF generation depends on optional libraries; placeholders may hide data gaps.
- Migration: handle None ratios and new `splits` field in `wfa.json`; install PDF tools if needed.
- Next Actions: enrich visualisation and add algorithm metadata to reports.

## Developer Notes — 2025-10-03T00:00:00Z (Phase 3 — Multi-Algo Builders)
- What: introduced registry-based algorithm builders (PPO, SAC, TD3/TQC stubs), enforced SAC Box-space guard, searched warm-start path order (user path → algo-scoped PPO best/last → legacy), added algo-scoped paths with legacy read-only fallbacks, logged algorithm source, and extended POSTRUN/KB with algorithm & eval_max_drawdown.
- Why: enable multi-algorithm training with safe defaults and consistent artefact layout.
- Risks: SAC fails on discrete actions; TD3/TQC currently unimplemented (exit 2).
- Migration: specify `--algorithm` and ensure environments expose Box spaces for SAC; legacy PPO checkpoints remain readable.
- Next Actions: implement full TD3/TQC builders and expand warm-start compatibility checks.

## Developer Notes — 2025-10-04T00:00:00Z (Phase 4 — Sweep & Smoke)
- What: added CPU-only `tools.sweep` CLI writing `runs.jsonl` and ranked `summary.csv`/`summary.md` with failure rows and atomic writers; extended `dev_checks` to assert log order, chart sizes and KB fields; introduced `smoke.yml` workflow running synth data, PPO train, eval, sweep and checks.
- Why: provide quick experiment sweeps and automated end-to-end validation with algorithm metadata.
- Risks: large sweeps may still be slow; smoke workflow covers only minimal steps.
- Migration: run `python -m bot_trade.tools.sweep` for CPU grids; inspect `smoke.yml` for CI expectations.
- Contracts: single headless notice preserved; `[LATEST] none` returns exit 2; CSV/MD/JSONL written atomically.

## Developer Notes — 2025-10-06T00:00:00Z (Headless notice & latest guards)
- What: consolidated single `[HEADLESS] backend=Agg` notice per CLI, moved backend selection into entrypoints, appended `[POSTRUN]` in `eval_run` with `eval_max_drawdown`, and hardened `--run-id latest` guards across export/eval/monitor.
- Why: avoid duplicate headless prints, surface max drawdown in eval summaries, and ensure predictable exits when no runs exist.
- Risks: base-directory overrides may still misalign with `BOT_REPORTS_DIR`.
- Migration Steps: none; CLIs remain unchanged but consumers should expect updated headless line and POSTRUN metrics.
- Next Actions: unify path resolution for custom report roots.

## Developer Notes — 2025-10-07
- What: single `[ENV]` notice on continuous env instantiation, policy_kwargs parser with JSON support, unused-key and invalid-JSON warnings, standardized TQC guard, KB newline enforcement and condensed policy_kwargs metadata.
- Why: tighten logging contracts and ensure KB entries capture algorithm config succinctly.
- Risks: unused-key detection is shallow; newline fix touches entire KB file on very large runs.
- Migration: none; pass policy_kwargs as dict/JSON; install `sb3-contrib` for TQC.
- Next Actions: expand policy_kwargs validation depth and monitor KB file growth.

## Developer Notes — 2025-09-02 22:47:38 UTC (Eval suite refresh)
- What: added equity/drawdown helpers, metric aggregator, walk-forward summary exports, KB schema (sortino/calmar/images_list), and canonical chart/eval print flow.
- Why: consolidate evaluation logic and enrich knowledge base context.
- Risks: walk-forward outputs may be empty on small logs; downstream parsers must handle new KB fields.
- Migration Steps: regenerate eval artifacts; update KB consumers for added fields.
- Next Actions: broaden regime metrics and refine WFA robustness.

## Developer Notes — 2025-09-02 23:16:24 UTC (Eval suite consolidation)
- What: localized `[HEADLESS]` prints in export/eval/monitor CLIs, enriched KB entries with metrics and image names, moved `[TEARSHEET]` prints into generator, and switched metrics.compute_all to equity-based inputs.
- Why: ensure single headless notice per CLI, avoid duplicate KB records, and finalize evaluation artifact workflow.
- Risks: downstream parsers may expect previous metric order or tearsheet prints from callers.
- Migration Steps: adjust scripts to new `[EVAL]` metric order and rely on `eval.tearsheet.generate_tearsheet` for its notice.
- Next Actions: extend metrics and regime analytics across evaluation utilities.

## Developer Notes — 2025-09-03 02:32:43 UTC (ai_core dummy signals)
- What changed: Implemented ai_core pipeline generating MACD/RSI/RV dummy signals and atomic bridge writes.
- Why: provide deterministic exogenous signals for strategy features and knowledge base records.
- Risks: signal file may grow quickly; normalization assumes finite inputs.
- Migration steps: enable with --ai-core and optional --emit-dummy-signals; consume via strategy_features.read_exogenous_signals.
- Next actions: hook real collectors and broaden feature set.

## Developer Notes — 2025-10-08
- What: CPU-only sweeper with grid/random modes and eval integration, stronger dev checks with chart DPI/size and ai_core signal assertions, algorithm-aware eval_run, newline fix logger, and expanded smoke workflow.
- Why: automate quick experiments and enforce artifact integrity in CI.
- Risks: sweep runs still spawn training loops; dev_checks rely on KB entries.
- Migration Steps: use `python -m bot_trade.tools.sweep` and `python -m bot_trade.tools.dev_checks --symbol <S> --frame <F> --run-id latest`.
- Next Actions: broaden hyper-parameter coverage and extend checks to additional artifacts.

## Developer Notes — 2025-09-03 03:20:01 UTC
- What: finalized CPU sweeper with warning thresholds and artifact checks, hardened dev_checks for fields/charts/signals, added smoke CI pipeline, and ensured newline fixes log via `[IO]`.
- Why: provide reproducible hyperparameter sweeps and stronger validation in development and CI.
- Risks: sweeps may be slow; dev_checks require populated knowledge base entries.
- Migration Steps: invoke `tools.sweep` for CPU grids or random trials; run `tools.dev_checks --run-id latest` after sweeps; review smoke workflow for expected commands.
- Next Actions: expand sweep hyper-parameters and extend dev checks to additional artifacts and metrics.

## Developer Notes — 2025-09-03T05:32:21Z (Platform Hardening)
- What changed: algorithm/run-scoped VecNormalize paths with legacy shim, UTF-8 enforcement across CLIs, configurable config generator, centralized action-space detection, data store skeleton with diagnostics, and warning-only dev checks.
- Why: harden path/encoding contracts, ease configuration and future data-platform work.
- Risks: deprecated vecnorm_path may hide legacy layouts; minimal data store offers limited validation.
- Migration steps: use `RunPaths.features.vecnorm` for new paths, regenerate configs via `tools.make_config`, run `paths_doctor`/`data_doctor` for diagnostics.
- Next actions: flesh out data store backends and expand config schema.
