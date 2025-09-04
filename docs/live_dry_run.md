# Live Dry-Run

The live dry-run mode consumes live ticker data but routes orders to the
in-memory paper gateway.  It is intended for quick end-to-end checks without
risking real funds.

```bash
python -m bot_trade.runners.live_dry_run --exchange binance --symbol BTCUSDT \
       --frame 1m --gateway paper --duration 120 --model-optional
```

Outputs are written under `results/<symbol>/<frame>/<run_id>/` including
`summary.json` and `risk_flags.jsonl`. If `--model-optional` is used and the
model is missing, the runner falls back to random actions and prints a warning.
