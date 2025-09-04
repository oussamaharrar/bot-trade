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

## Safety Notes

- All commands are expanded from `commands_registry` templates; unknown flags are rejected.
- Runner writes UTF‑8 tee logs and stops the whole process tree on request.
- Results watcher parses structured events only inside the run directory.
