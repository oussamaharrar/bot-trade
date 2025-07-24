import pandas as pd
import joblib

model = joblib.load("trained_model.pkl")

def apply_ml_strategy(df):
    df['price_change'] = df['close'].pct_change().fillna(0)
    df['coin_delta'] = df['close'].diff().fillna(0)
    df['usdt_delta'] = df['volume'].diff().fillna(0)
    df['value_delta'] = df['price_change'] + df['coin_delta'] + df['usdt_delta']

    signals = []
    for i in range(1, len(df)):
        row = df.iloc[i]
        features = pd.DataFrame([{
            "price": row['close'],
            "price_change": row['price_change'],
            "coin_value": row['close'],  # proxy
            "usdt": 1000,                # proxy
            "coin_delta": row['coin_delta'],
            "usdt_delta": row['usdt_delta'],
            "value_delta": row['value_delta']
        }])
        prediction = model.predict(features)[0]
        signal = "BUY" if prediction == 1 else "SELL"
        signals.append((df['timestamp'].iloc[i], df['close'].iloc[i], signal))
    return signals