# Sandbox Gateway Quickstart

The sandbox gateway connects the trading bot to exchange testnets for safe
experimentation.  Only Binance and Bybit testnets are supported.

## Setup

1. Obtain testnet API keys and export them:
   ```bash
   export BINANCE_API_KEY=... BINANCE_API_SECRET=...
   export BYBIT_API_KEY=...   BYBIT_API_SECRET=...
   ```
2. Use the gateway via the CLI:
   ```bash
   python -m bot_trade.tools.check_sandbox --exchange binance --symbol BTCUSDT
   ```

## Features

- Separate adapters for Binance and Bybit.
- Shared token bucket rate limiter with automatic retries.
- Never touches real-money endpoints; all requests use testnet hosts.
