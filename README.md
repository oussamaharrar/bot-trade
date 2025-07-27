# 🤖 AI-Driven Crypto Trading Bot

An intelligent, modular trading bot that uses rule-based and machine learning strategies, automated retraining, live evaluation, and full reporting — all orchestrated via a unified command system.


## 📁 Project Structure

bot-trade/
├── run_bot.py # Main orchestrator: runs strategy, logs, trains, evaluates
├── dashboard.py # CLI dashboard for maintenance & control
├── autolearn.py # Trains the model (DecisionTreeClassifier)
├── evaluate_model.py # Evaluation + trend charts + PDF report generation
├── config.yaml # Configurations for coin, thresholds, etc.
├── market_data.py # Binance data fetcher (via ccxt)
├── ml_strategy.py # ML-based strategy with auto-fallback
├── env_trading.py # Gym environment for RL
├── train_rl.py # Train PPO/DQN agent
├── run_rl_agent.py # Run trained RL agent
├── results_logger.py # Logs trades and feeds training dataset
├── training_dataset.csv # Cumulative dataset for ML training
├── requirements.txt # Dependencies
├── models/ # All saved ML models (.pkl)
├── results/ # Logged trades per run (.csv)
├── reports/ # Plots & PDF reports
├── logs/ # Runtime & error logs

yaml
Copy
Edit

---

## 🧠 Strategy

- **ML strategy** using `DecisionTreeClassifier`
- **RL strategy** using `Stable-Baselines3` (PPO)
- Strategy input features: `price_change`, `coin_delta`, `usdt_delta`, `value_delta`, etc.
- Auto retrain after configurable trade threshold
- Feature mismatch recovery included

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
Ensure you are using Python 3.9+

⚙️ Configuration
Edit config.yaml:

yaml
Copy
Edit
coins:
  - BTC/USDT
  - ETH/USDT
  - SOL/USDT
max_trades_before_retrain: 100
strategy: "ml"  # or "rule"
# set to "rl" to use the reinforcement learning agent
debug_mode: true
report_format: "pdf"
output_dirs:
  models: "models/"
  results: "results/"
  reports: "reports/"
  logs: "logs/"
🚀 Running the Bot
Run full pipeline:

bash
Copy
Edit
python run_bot.py
Or interact with CLI Dashboard:

bash
Copy
Edit
python dashboard.py
📊 Outputs
✅ Logs: results/run_*.csv

✅ Models: models/trained_model_<timestamp>.pkl

✅ Evaluation Chart: reports/f1_accuracy_trend.png

✅ PDF Report: reports/report_<timestamp>.pdf

✅ Errors: logs/errors.log

🧩 Auto-Repair Logic
If model fails or is outdated, autolearn.py runs automatically

If features mismatch during prediction, system retrains on the fly

🛡️ Security Note
No real API keys are required for simulation

The bot now loads variables from a `.env` file. Live trading is disabled by
default (`LIVE_TRADING=false`). Copy `.env.example` to `.env` and adjust values
if you want to trade on a real exchange.

### Docker

Build and run with Streamlit dashboard exposed on port 8501:

```bash
docker build -t bot-trade .
docker run -p 8501:8501 -v $(pwd)/models:/app/models -v $(pwd)/results:/app/results bot-trade
```

### Makefile Helpers

Common tasks can be run with `make`:

```bash
make train         # train model
make evaluate      # evaluate model
make rl-train      # train RL agent
```

### Scheduled Retraining

Use `python scripts/schedule.py` to trigger periodic retraining based on the
`RETRAIN_CRON` environment variable (defaults to daily).

👨‍💻 Maintainer
Built and improved by: @oussamaharrar


