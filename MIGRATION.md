# Migration Guide

This project evolves over time; key changes impacting usage are documented here.

## VecNormalize paths

VecNormalize files are now scoped by algorithm and run ID via `RunPaths.features.vecnorm`. Legacy locations are read once and redirect to the new path, printing a single `[DEPRECATION] vecnorm legacy path used → redirected` line.

### Action

- Tools and scripts should stop using `vecnorm_path(...)` and rely on `RunPaths.features.vecnorm`.
- After migration, remove legacy vecnorm files.

## Configuration source order

Configuration settings are loaded with the following precedence:

1. CLI arguments
2. `config.yaml` (if present)
3. `config.default.yaml`

### Action

- Place project defaults in `config.default.yaml`.
- Create `config.yaml` for local overrides or generate one via `python -m bot_trade.tools.make_config`.

