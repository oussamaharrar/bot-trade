import pandas as pd
import os
import subprocess
import joblib

DATASET_PATH = "training_dataset.csv"
MODEL_PATH = "trained_model.pkl"
RETRAIN_THRESHOLD = 100

def check_and_train():
    if not os.path.exists(DATASET_PATH):
        print("[❌] No dataset found.")
        return

    df = pd.read_csv(DATASET_PATH)
    df = df[df['signal'].isin(['BUY', 'SELL'])]
    df = df[df['pnl_class'] >= 0]

    if len(df) < RETRAIN_THRESHOLD:
        print(f"[ℹ️] Only {len(df)} trades in dataset. Need {RETRAIN_THRESHOLD} to train.")
        return

    print(f"[⚙️] Retraining model on {len(df)} trades...")
    subprocess.run(["python", "train_model.py"])

    if os.path.exists(MODEL_PATH):
        print("[✅] Model updated and saved.")
    else:
        print("[❌] Training failed.")

if __name__ == "__main__":
    check_and_train()