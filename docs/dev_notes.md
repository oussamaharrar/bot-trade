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
