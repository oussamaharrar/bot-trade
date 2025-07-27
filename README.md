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


---

## 🧠 Strategy

- **ML strategy** using RandomForest, XGBoost or LightGBM models
- **RL strategy** via Stable-Baselines3 (PPO or DQN)
- Strategy input features: `price_change`, `coin_delta`, `usdt_delta`, `value_delta`, etc.
- Auto retrain after configurable trade threshold
- Feature mismatch recovery included

## 🚀 Training Options
- `python autolearn.py --model [rf|xgb|lgbm]`
- `python train_rl.py --agent [ppo|dqn]`


---


## 🛠️ Installation

```bash
pip install -r requirements.txt
Ensure you are using Python 3.9+

⚙️ Configuration

coins:
  - BTC/USDT
  - ETH/USDT
  - SOL/USDT
max_trades_before_retrain: 100
strategy: "ml"  # or "rule"
model_type: rf  # rf|xgb|lgbm
rl_agent: ppo  # ppo|dqn
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

python run_bot.py
Or interact with CLI Dashboard:

python dashboard.py
### Streamlit Dashboard

```bash
streamlit run dashboard.py
```
### Docker Usage

```bash
docker build -t bot-trade .
docker run --rm -it bot-trade
```
### Makefile Shortcuts

```bash
make train      # train ML model
make train-rl   # train RL agent
make run        # run bot
make dashboard  # launch Streamlit
make docker-build
make docker-run
```



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

Add .env support if integrating with real exchange (Binance/Bybit/etc.)

👨‍💻 Maintainer
Built and improved by: @oussamaharrar


