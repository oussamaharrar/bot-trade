# UI Control Panel

Minimal PySimpleGUI interface to manage training runs and tools.

## Usage

```
python -m bot_trade.ui.panel_gui
```

The panel validates commands against a strict whitelist and spawns them
via `ui.runner`. Results are tailed by `ui.results_watcher`.

## Safety Notes

- Only predefined commands may run.
- Process trees are terminated on Stop.
- Logs are written UTF-8 to the `tee` files.
