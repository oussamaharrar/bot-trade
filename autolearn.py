import argparse
import json
import os
import zipfile
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

# Ensure required directories exist

for d in ["models", "results", "reports", "logs"]:
    os.makedirs(d, exist_ok=True)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from data_utils import load_dataset
from strategy_features import add_strategy_features

MODEL_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_dataset(path: str = "training_dataset.csv") -> pd.DataFrame:
    """Load CSV, zipped CSV or Parquet file."""
    if path.endswith(".zip"):
        dfs = []
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith(".csv"):
                    with z.open(name) as f:
                        dfs.append(pd.read_csv(f))
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    elif path.endswith(".parquet") or path.endswith(".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


parser = argparse.ArgumentParser()
parser.add_argument("--data", default="training_dataset.csv", help="Dataset path")
parser.add_argument("--smote", action="store_true", help="Apply SMOTE oversampling")
parser.add_argument("--undersample", action="store_true", help="Apply random undersampling")
args = parser.parse_args()


df = load_dataset(args.data)

if len(df) < 100:
    print("Not enough data to retrain the model. Minimum required: 100 rows.")
    exit()

df = df.dropna(subset=["pnl_class"])  # remove rows with missing labels
df = df[df["pnl_class"] >= 0]

X = df.drop(columns=["pnl_class"])
X = X.select_dtypes(include=["number"])
y = df["pnl_class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Optional: Load best previous model based on highest F1 score

best_model_path = None
best_row = None
if os.path.exists("model_evaluation_log.csv"):
    eval_df = pd.read_csv("model_evaluation_log.csv", on_bad_lines="skip")
    eval_df = eval_df.dropna(subset=["f1_score", "model_path"])
    if not eval_df.empty:
        best_row = eval_df.sort_values(by="f1_score", ascending=False).iloc[0]
        best_model_path = best_row["model_path"]
        print(
            f"📦 Best previous model: {best_model_path} (F1: {best_row['f1_score']:.4f})"
        )

if best_model_path:
    best_model = joblib.load(best_model_path)
else:
    print("⚠️ No valid previous model found. Skipping best model load.")


# Train RandomForest, XGBoost and LightGBM; pick best by F1
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="logloss", use_label_encoder=False, random_state=42
    ),
    "LightGBM": LGBMClassifier(random_state=42),
}

best_model = None
best_f1 = -1.0
best_algo = None
best_pred = None

for algo, clf in models.items():
    steps = [("scaler", StandardScaler())]
    if args.smote:
        steps.append(("sampler", SMOTE(random_state=42)))
    elif args.undersample:
        steps.append(("sampler", RandomUnderSampler(random_state=42)))
    steps.append(("clf", clf))
    pipe = ImbPipeline(steps)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    f1_val = f1_score(y_test, preds, average="weighted")
    if f1_val > best_f1:
        best_f1 = f1_val
        best_model = pipe
        best_algo = algo
        best_pred = preds

model = best_model
y_pred = best_pred

# Save the model (latest)
latest_model_path = os.path.join(MODEL_DIR, "trained_model.pkl")
joblib.dump((model, list(X.columns)), latest_model_path)

print(f"✅ Latest model saved to {latest_model_path}")

# Save versioned model
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
versioned_model_path = os.path.join(MODEL_DIR, f"trained_model_{timestamp}.pkl")
joblib.dump((model, list(X.columns)), versioned_model_path)
print(f"🗂️ Versioned model saved as {versioned_model_path}")

# reload dataset for logging sample count
from strategy_features import add_strategy_features

df_log = load_dataset(args.data)
df_log = df_log.dropna(subset=["pnl_class"])
df_log = add_strategy_features(df_log)


# Evaluate best model
accuracy = accuracy_score(y_test, y_pred)
f1 = best_f1
print(f"Model Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f} ({best_algo})")

# Compare with previous best F1 score
if best_model_path:
    prev_f1 = best_row["f1_score"]
    diff = f1 - prev_f1
    if diff > 0:
        print(f"🔼 Model improved! Previous F1: {prev_f1:.4f} → Current: {f1:.4f} (+{diff:.4f})")
    elif diff < 0:
        print(f"🔽 Model dropped! Previous F1: {prev_f1:.4f} → Current: {f1:.4f} ({diff:.4f})")
    else:
        print(f"➖ No change in F1 score. ({f1:.4f})")
else:
    print("ℹ️ No previous model to compare with.")

# Log performance
log_file = "model_evaluation_log.csv"
log_exists = os.path.exists(log_file)
f1_delta = f1 - best_row["f1_score"] if best_model_path else 0.0

with open(log_file, "a") as f:
    if not log_exists:
        f.write(
            "timestamp,accuracy,f1_score,f1_delta,num_samples,algorithm,model_path\n"
        )
    f.write(
        f"{datetime.now()},{accuracy:.4f},{f1:.4f},{f1_delta:.4f},{len(df_log)},{best_algo},{versioned_model_path}\n"
    )

print(f"📈 Evaluation logged to {log_file}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, f"confusion_matrix_{timestamp}.png"))
plt.close()

# Feature Importance Plot (if available)
if hasattr(model.named_steps["clf"], "feature_importances_"):
    importances = model.named_steps["clf"].feature_importances_
    feature_names = X.columns
    feat_imp_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=feat_imp_df, x="Importance", y="Feature", color="blue")
    plt.title(f"Feature Importance ({best_algo})")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"feature_importance_{timestamp}.png"))
    plt.close()

# Plot F1 Score over time
try:
    df_eval = pd.read_csv("model_evaluation_log.csv", on_bad_lines="skip")
    df_eval["timestamp"] = pd.to_datetime(df_eval["timestamp"])
    df_eval = df_eval.sort_values("timestamp")

    plt.figure(figsize=(8, 4))
    plt.plot(df_eval["timestamp"], df_eval["f1_score"], marker="o", linestyle="-")
    plt.title("F1 Score Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"f1_trend_{timestamp}.png"))
    plt.close()
except Exception as e:
    print(f"⚠️ Could not plot F1 Score over time: {e}")
    
# Plot F1 Score, Accuracy, and F1 Delta over time
try:
    df_eval = pd.read_csv("model_evaluation_log.csv", on_bad_lines="skip")
    df_eval["timestamp"] = pd.to_datetime(df_eval["timestamp"])
    df_eval = df_eval.sort_values("timestamp")

    plt.figure(figsize=(10, 5))

    plt.plot(df_eval["timestamp"], df_eval["f1_score"], label="F1 Score", marker="o")
    plt.plot(df_eval["timestamp"], df_eval["accuracy"], label="Accuracy", marker="s")
    plt.plot(df_eval["timestamp"], df_eval["f1_delta"], label="F1 Delta", marker="^")

    plt.title("Model Evaluation Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Score")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"evaluation_over_time_{timestamp}.png"))
    plt.close()
except Exception as e:
    print(f"⚠️ Could not plot evaluation chart: {e}")


# Update memory
=======
EVAL_LOG = "model_evaluation_log.csv"

for d in [MODEL_DIR, "results", REPORTS_DIR, "logs", "memory"]:
    os.makedirs(d, exist_ok=True)


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
