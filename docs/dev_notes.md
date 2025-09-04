[2024-05-31 00:00] initial notes file created.
[2024-05-31 12:00] execution_layer / time_splits / gates — added deterministic execution simulator with fees/slippage/latency/partials, new chronological split helpers, and threshold gate with pass ratios.
[2024-05-31 12:30] style pass — fixed type hints and import ordering per ruff.
[2025-06-01 00:00] E2E Integration after ExecutionLayer/TimeSplits/Gates —
- Added maker/min fee handling and fill logging in backtest ExecutionLayer.
- Introduced eval_run wrapper, HTML tearsheet, gate fail-hard promotion, live dry-run fallbacks, paths doctor and CI workflow.
- Risks: limited real exchange testing; paths doctor only checks basic structure.
- Next: expand lint coverage gradually, enhance live feed resilience.
[2025-10-12 00:00] E2E close-out — CLI unification, data smoke, WFA thresholds (smoke), paths_doctor expectations, fill logging.
- Added unified args for wfa_gate, live_dry_run, and paths_doctor with smoke profiles and hints.
- Ensured eval_run emits summary.json and metrics.csv; HTML tearsheet reads them and skips PDF gracefully.
- Expanded Fill model and execution logs with maker/min-fee fields; seed persisted in meta.
- Risks: strict path checks may flag custom layouts; random fills still simplistic.
- Next: broaden real-exchange coverage and refine walk-forward metrics.
[2025-10-13 00:00] [V3 Base] Merge V1/V2 deltas + live feed fallback + paths_doctor tightening + fill log standardization.
- WHAT: merged variant features, added HTTP polling fallback with rate limiting, tightened paths_doctor checks, standardized fill logs.
- WHY: consolidate improvements for stable base and ensure predictable tooling.
- RISKS: HTTP polling may hit API limits; paths_doctor hints assume default configs.
- NEXT: extend policies for live trading and enhance fallback resiliency.
[2025-10-14 00:00] [Live Feed Patch] Robust polling: bad-price guards, EWMA smoothing, last-known fallback, limiter-aware backoff.
- WHAT: added price validation with last-price fallback, EWMA smoothing and rate limiter-aware exponential backoff.
- WHY: prevent invalid ticks from breaking runs and stabilize metrics under network limits.
- RISKS: smoothing may lag fast markets; aggressive backoff delays recovery.
- MIGRATION: optional --bootstrap-price seeds initial price; no other steps.
- NEXT: tune smoothing factor and expand real exchange coverage.
[2025-10-15 00:00] [Final Merge] Import live tests, unify logs, tighten CI order, quickstart polish.
- WHAT: merged live feed/runner tests, standardized LIVE/FILL logs, documented E2E commands and CLI flags, tightened CI to lint, compile, test, and validate paths.
- WHY: ensure reproducible end-to-end workflow and grep-friendly logging for final release.
- RISKS: stricter CI may fail on legacy branches; log parsing tools must handle new format.
- MIGRATION: rerun quickstart to regenerate results; update any custom parsers for new logs.
- NEXT: broaden exchange coverage and integrate real order execution.
[2025-10-16 00:00] [Final Cutover] live_sandbox execution + safety guards + RC checklist.
- WHAT: enabled sandbox live orders with strict flags, key checks, order/fill persistence, paths doctor sandbox checks, docs update, VERSION 0.1-rc.
- WHY: finalize testnet execution path and prepare release candidate.
- RISKS: limited testnet coverage; misconfigured endpoints may hit mainnet.
- MIGRATION: export testnet API keys, run live_dry_run with --gateway sandbox --i-understand-testnet, regenerate results.
- NEXT: expand exchange adapters and refine PnL tracking.
[2025-09-04 00:00] UI Panel: v0 MVP —
- WHAT: added PySimpleGUI control panel with runner, results watcher, whitelist, defaults and docs.
- WHY: enable desktop orchestration of training, tools, and live runs.
- RISKS: limited real-trading coverage; minimal log/error handling.
- NEXT: enrich controls, metrics charts, and sandbox safety tests.
[2025-10-17 00:00] UI Panel v0 MVP → v1 Controls —
- WHAT: expanded panel with dev-notes preview, tee_path runner API and safer stop_process_tree.
- WHY: improve traceability and reliability for multi-run orchestration.
- RISKS: stop handling may miss zombie children on exotic platforms.
- NEXT: surface metrics charts and sandbox key validation.
[2025-09-04 05:16:28] Developer Notes — Panel v1 Controls
- WHAT: added PySimpleGUI panel with runner, results watcher, whitelist and defaults.
- WHY: enable desktop orchestration with safe process handling.
- RISKS: limited log handling; watcher polls filesystem.
- NEXT: surface charts and queue management.
[2025-10-18 00:00] UI Panel v2 — Controls
- WHAT: added commands registry, device selector, job-aware runner and results watcher events; shipped new panel_gui with config persistence.
- WHY: enable safe orchestration of multiple runs with full observability.
- RISKS: UI remains polling based; incomplete flag coverage may block advanced users.
- MIGRATION: existing `bot_trade.ui.*` modules now shim to `bot_trade.tools.*`.
- NEXT: queue management and richer log colouring.

Panel v3 — Controls, multi-algo, WFA/tearsheet, sweeps, CI smoke (2025-09-04T06:53:28.334438)
