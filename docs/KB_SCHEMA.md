# KB Schema

```json
{"run_id":"abc123","symbol":"BTCUSDT","frame":"1m","algorithm":"SAC","algo_meta":{"policy_kwargs":{"net_arch":{"pi":[64,64],"vf":[64,64]},"activation_fn":"ReLU"}},"eval":{"sharpe":0.42,"max_drawdown":0.10},"images":7,"regime":{"active":"bull"},"ai_core":{"signals_count":3,"sources":["dummy"]}}
```

Lines are appended atomically and deduplicated on `run_id`.

### Troubleshooting
- `[KB] duplicate run_id` skipped
- `[IO] fixed_trailing_newline file=<path>` repaired newline
