import subprocess
import os
import yaml
from datetime import datetime

# Create required folders if missing
for d in ["models", "results", "reports", "logs"]:
    os.makedirs(d, exist_ok=True)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

max_trades = config.get("max_trades_before_retrain", 100)

def print_banner():
    print("\n" + "="*50)
    print("🚀 Starting AI-Driven Trading Bot Runner")
    print("="*50 + "\n")

def run_step(name, cmd):
    print(f"▶️ {name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error in {name}: {result.stderr}")
        with open("logs/errors.log", "a") as logf:
            logf.write(f"[{datetime.now()}] {name} failed:\n{result.stderr}\n\n")
    else:
        print(f"✅ {name} completed.\n")

def get_trade_count():
    try:
        trades = os.listdir("results")
        return sum(1 for f in trades if f.endswith(".csv"))
    except:
        return 0

# ==== EXECUTION SEQUENCE ====

print_banner()

run_step("Running bot logic", ["python", "bot_loop_ml.py"])

# Check if it's time to retrain
if get_trade_count() >= max_trades:
    run_step("Training model (autolearn.py)", ["python", "autolearn.py"])
    run_step("Evaluating model", ["python", "evaluate_model.py"])
else:
    print(f"ℹ️ Model retraining not triggered — only {get_trade_count()} trades logged.\n")

print("✅ All done. You may review reports/ and models/ now.")
