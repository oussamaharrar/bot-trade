# Walk-Forward Analysis Gate

The WFA gate splits a date range into rolling windows and evaluates performance
metrics per window.  After running, `wfa_report.json`, `wfa_report.csv` and
`charts/wfa_overview.png` are produced.

```
python -m bot_trade.eval.wfa_gate --config config/wfa.yaml --symbol BTCUSDT \
       --frame 1m --windows 6 --embargo 0.05 --profile default
```
