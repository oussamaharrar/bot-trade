# UI Panel

Minimal PySimpleGUI control panel for orchestrating training and tooling.

## Install

```bash
pip install -r requirements.txt  # includes PySimpleGUI, psutil, watchdog, pyyaml
```

## Launch

```bash
python -m bot_trade.ui.panel
```

## Tabs

- **Run Train** – set symbol/frame/algorithm and launch trainings. Each run streams logs and can be stopped independently.
- **Eval & WFA** – run evaluation helpers on completed results.
- **Live (Paper/Sandbox)** – guarded entry points for paper or sandbox live modes.
- **Data & Tools** – whitelisted maintenance scripts such as synthetic data generation.
- **Logs & Errors** – switch between run logs and copy the last 200 lines.
- **Config** – environment key status and latest developer-note entry.

State is saved to `bot_trade/ui/panel_config.yaml` on exit so selections persist.

## Sandbox notes

Sandbox live trading is disabled unless testnet API keys are present and the consent box is checked.

## Troubleshooting

- Missing results artefacts → run a training first.
- "⚠️" on API keys → export the required environment variables.
- GPU not detected → panel defaults to CPU; select `cpu` explicitly if needed.
