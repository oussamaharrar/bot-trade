# Registries

## Algorithms
PPO, SAC, TD3, and TQC load lazily. Off-policy algos require the `--continuous-env` Box space and the TQC guard warns if `sb3-contrib` is missing.

*Extend*: add `myalgo` in `rl_builders.py` registry and guard imports.

## Rewards
Terms and clamps come from YAML (`config/rewards/*.yml`). Each term has a weight and min/max clamp applied before a global clamp.

*Extend*: create `config/rewards/custom.yml` and pass `--reward-spec path`.

## Risk rules
Circuit breakers and exposure caps live in `RiskManager`. Values beyond safe bounds are clamped with `[RISK_CLAMP]` lines.

*Extend*: add new checks in `risk_manager.py` and log markers on trigger.

## Slippage models
`ExecutionSim` supports `fixed_bp`, `vol_aware`, and `depth_aware`. CLI flags choose model and parameters.

*Extend*: implement `_slippage_bp` branch and register the name.

## Feature providers
Market features come from `strategy_features.py` and optional `ai_core` signals.

*Extend*: add functions producing new columns; ensure KB log counts update.

## Regime/adaptive wiring
`regime.yml` maps detected regimes to reward/risk deltas. Adaptive controller clamps changes and logs `[ADAPT]` once per window.

*Extend*: edit `config/adaptive/regime.yml` and adjust controller thresholds.

### Troubleshooting
- `[ALGO_GUARD]` unsupported or missing algo
- `[RISK_CLAMP]` unsafe risk override
- `[SWEEP_WARN]` out-of-range sweep metric
