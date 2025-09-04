# Panel v2 — Quick Tour

Cross-platform PySimpleGUI control panel to orchestrate training, evaluation and
auxiliary tools. Commands are constructed from a strict whitelist registry and
executed via the safe runner with tee logging and process-tree termination.

## Usage

```bash
python -m bot_trade.tools.panel_gui
```

Tabs:
- **Train** – launch RL training with device selection and optional flags.
- **Eval / WFA** – export charts or run walk-forward analysis for a given run.
- **Live** – spawn a paper live-dry-run against supported exchanges.
- **Tools** – helpers like synthetic data generation or paths doctor.
- **Jobs** – view active jobs, stop them and tail their logs.
- **Logs** – live tail with colour markers for `[DEBUG_EXPORT]`, `[CHARTS]`, `[POSTRUN]` and `[EVAL]`.
- **Config** – persist last selections and preview latest developer note.

## Panel v3

```
python -m bot_trade.tools.ui_panel
```

Tabs now cover Run, Jobs, Tools, Live and Config. Commands resolve via
`commands_registry.yaml` with strict flag validation. Jobs can be listed and
stopped, and the Live tab tails logs with structured `[CHARTS]` and `[POSTRUN]`
events. Settings persist across sessions.

## Safety Notes

- All commands are expanded from `commands_registry` templates; unknown flags are rejected.
- Runner writes UTF‑8 tee logs and stops the whole process tree on request.
- Results watcher parses structured events only inside the run directory.

### Panel v3

Panel v3 introduces a lightweight job queue with device pinning and a live log tail. Presets from `config/presets/training.yml` allow one-click command filling while the whitelist keeps unsafe flags out.
