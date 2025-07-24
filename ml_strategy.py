import pandas as pd
import joblib
import subprocess
import os

# Ensure required directories exist
for d in ["models", "results", "reports", "logs"]:
    os.makedirs(d, exist_ok=True)

def apply_ml_strategy(df):
    # Feature engineering
    df['price_change'] = df['close'].pct_change().fillna(0)
    df['coin_delta'] = df['close'].diff().fillna(0)
    df['usdt_delta'] = df['volume'].diff().fillna(0)
    df['value_delta'] = df['price_change'] + df['coin_delta'] + df['usdt_delta']

    # Try to load model and feature columns
    try:
        model, expected_columns = joblib.load("models/trained_model.pkl")
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
    except Exception as e:
        print(f"⚠️ Model loading failed or incompatible: {e}")
        print("🔄 Re-training model with autolearn.py...")
        subprocess.run(["python", "autolearn.py"])
        model, expected_columns = joblib.load("models/trained_model.pkl")

    # Generate signals
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        features = pd.DataFrame([{
            col: row.get(col, 0) for col in expected_columns
        }])
        prediction = model.predict(features)[0]
        signal = "BUY" if prediction == 1 else "SELL"
        signals.append((df['timestamp'].iloc[i], df['close'].iloc[i], signal))

    return signals
