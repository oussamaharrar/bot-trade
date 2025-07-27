import argparse
import json
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from data_utils import load_dataset
from strategy_features import add_strategy_features

# Ensure required directories exist
for d in ["models", "results", "reports", "logs", "memory"]:
    os.makedirs(d, exist_ok=True)

MODEL_DIR = "models"
REPORTS_DIR = "reports"
EVAL_LOG = "model_evaluation_log.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML models for trading")
    parser.add_argument("--data", default="training_dataset.csv", help="Dataset path (csv, zip, parquet)")
    parser.add_argument("--balance", choices=["none", "smote", "undersample"], default="none", help="Balance technique")
    return parser.parse_args()


def balance(X, y, method: str):
    if method == "smote":
        sampler = SMOTE(random_state=42)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=42)
    else:
        return X, y
    X_bal, y_bal = sampler.fit_resample(X, y)
    return X_bal, y_bal


def train_models(X_train, y_train, X_test, y_test):
    models = [
        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("XGBoost", XGBClassifier(eval_metric="logloss", use_label_encoder=False)),
        ("LightGBM", LGBMClassifier())
    ]
    best = None
    best_f1 = -1.0
    best_pred = None
    best_name = ""
    for name, clf in models:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        f1 = f1_score(y_test, preds, average="weighted")
        if f1 > best_f1:
            best = clf
            best_f1 = f1
            best_pred = preds
            best_name = name
    return best, best_name, best_pred, best_f1


def main():
    args = parse_args()
    df = load_dataset(args.data)
    if len(df) < 100:
        print("Not enough data to retrain the model. Minimum required: 100 rows.")
        return

    df = df.dropna(subset=["pnl_class"])
    df = add_strategy_features(df)
    X = df.drop(columns=["pnl_class"]).select_dtypes(include=["number"])
    y = pd.Categorical(df["pnl_class"]).codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train, y_train = balance(X_train, y_train, args.balance)

    model, model_name, y_pred, best_f1 = train_models(X_train, y_train, X_test, y_test)

    accuracy = accuracy_score(y_test, y_pred)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    latest_model_path = os.path.join(MODEL_DIR, "trained_model.pkl")
    versioned_model_path = os.path.join(MODEL_DIR, f"trained_model_{timestamp}.pkl")
    joblib.dump({"model": model, "columns": list(df.drop(columns=["pnl_class"]).select_dtypes(include=["number"]).columns), "scaler": scaler}, latest_model_path)
    joblib.dump({"model": model, "columns": list(df.drop(columns=["pnl_class"]).select_dtypes(include=["number"]).columns), "scaler": scaler}, versioned_model_path)
    print(f"✅ Latest model saved to {latest_model_path}")

    prev_best = None
    if os.path.exists(EVAL_LOG):
        eval_df = pd.read_csv(EVAL_LOG)
        if not eval_df.empty:
            prev_best = eval_df.sort_values(by="f1_score", ascending=False).iloc[0]

    f1_delta = 0.0
    if prev_best is not None:
        f1_delta = best_f1 - prev_best["f1_score"]
        print(f"Previous best F1: {prev_best['f1_score']:.4f}")

    with open(EVAL_LOG, "a") as f:
        if os.stat(EVAL_LOG).st_size == 0:
            f.write("timestamp,accuracy,f1_score,f1_delta,num_samples,model_path\n")
        f.write(f"{datetime.now()},{accuracy:.4f},{best_f1:.4f},{f1_delta:.4f},{len(df)},{versioned_model_path}\n")
    print(f"📈 Evaluation logged to {EVAL_LOG}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"confusion_matrix_{timestamp}.png"))
    plt.close()

    memory_path = os.path.join("memory", "memory.json")
    memory = {}
    if os.path.exists(memory_path):
        try:
            with open(memory_path, "r") as f:
                memory = json.load(f)
        except Exception:
            memory = {}
    memory["last_training_accuracy"] = best_f1
    memory["last_training_model"] = model_name
    with open(memory_path, "w") as f:
        json.dump(memory, f, indent=2)

    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {best_f1:.4f} using {model_name}")


if __name__ == "__main__":
    main()
