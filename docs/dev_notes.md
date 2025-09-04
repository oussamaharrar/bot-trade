[2024-05-31 00:00] initial notes file created.
[2024-05-31 12:00] execution_layer / time_splits / gates — added deterministic execution simulator with fees/slippage/latency/partials, new chronological split helpers, and threshold gate with pass ratios.
[2024-05-31 12:30] style pass — fixed type hints and import ordering per ruff.
[2025-06-01 00:00] E2E Integration after ExecutionLayer/TimeSplits/Gates —
- Added maker/min fee handling and fill logging in backtest ExecutionLayer.
- Introduced eval_run wrapper, HTML tearsheet, gate fail-hard promotion, live dry-run fallbacks, paths doctor and CI workflow.
- Risks: limited real exchange testing; paths doctor only checks basic structure.
- Next: expand lint coverage gradually, enhance live feed resilience.
