# Developer Notes

## VecNormalize scoping
VecNormalize statistics now live under `reports/<ALGO>/<SYMBOL>/<FRAME>/<RUN_ID>/features/vecnorm.pkl`. The helper `vecnorm_path` warns once when called without explicit algorithm/run and falls back to legacy locations.

## UTF-8 policy
Entry points call `force_utf8()` to enforce `PYTHONIOENCODING=UTF-8` and to reconfigure `stdout` with replacement error handling. One `[ENCODING]` line is printed on start.

## Config precedence
Configuration loads in the following order: CLI `--config` > `config/config.yaml` if present > `config/config.default.yaml`.

## Tooling
- `python -m bot_trade.tools.make_config` – generate configs from defaults and presets.
- `python -m bot_trade.tools.paths_doctor` – verify run path layout and vecnorm split.
- `python -m bot_trade.tools.data_doctor` – report data store initialization status.
