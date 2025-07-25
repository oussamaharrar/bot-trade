import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Ensure required directories exist
for d in ["models", "results", "reports", "logs"]:
    os.makedirs(d, exist_ok=True)

def load_dataset(file="training_dataset.csv"):
    df = pd.read_csv(file)
    df = df[df['pnl_class'] >= 0]  # Only keep WIN/LOSS
    df = df[df['signal'].isin(["BUY", "SELL"])]  # Ignore HOLD for training
    df['signal_code'] = df['signal'].map({'BUY': 1, 'SELL': 0})
    return df

def train_model(df):
    X = df[["price", "price_change", "coin_value", "usdt", "coin_delta", "usdt_delta", "value_delta"]]
    y = df["signal_code"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[🎓] Model Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["SELL", "BUY"]))

    joblib.dump(model, "trained_model.pkl")
    print("[💾] Model saved to trained_model.pkl")

if __name__ == "__main__":
    df = load_dataset()
    train_model(df)
