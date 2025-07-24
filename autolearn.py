import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure required directories exist
for d in ["models", "results", "reports", "logs"]:
    os.makedirs(d, exist_ok=True)

# === Setup model directory ===
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv("training_dataset.csv")

# Check data size
if len(df) < 100:
    print("Not enough data to retrain the model. Minimum required: 100 rows.")
    exit()

# Prepare features and labels
X = df.drop(columns=["pnl_class"])
X = X.select_dtypes(include=["number"])
y = df["pnl_class"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Optional: Load best previous model based on highest F1 score
best_model_path = None
if os.path.exists("model_evaluation_log.csv"):
    eval_df = pd.read_csv("model_evaluation_log.csv")
    eval_df = eval_df.dropna(subset=["f1_score", "model_path"])
    if not eval_df.empty:
        best_row = eval_df.sort_values(by="f1_score", ascending=False).iloc[0]
        best_model_path = best_row["model_path"]
        print(f"📦 Best previous model: {best_model_path} (F1: {best_row['f1_score']:.4f})")
        # Optional: load it for comparison or warm-start
if best_model_path:
    best_model = joblib.load(best_model_path)
    # Optional: evaluate or compare with current model
else:
    print("⚠️ No valid previous model found. Skipping best model load.")


# Train the model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save the model (latest)
latest_model_path = os.path.join(MODEL_DIR, "trained_model.pkl")
joblib.dump((model, list(X.columns)), latest_model_path)

print(f"✅ Latest model saved to {latest_model_path}")

# Save versioned model
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
versioned_model_path = os.path.join(MODEL_DIR, f"trained_model_{timestamp}.pkl")
joblib.dump(model, versioned_model_path)
print(f"🗂️ Versioned model saved as {versioned_model_path}")

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Model Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

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
        f.write("timestamp,accuracy,f1_score,f1_delta,num_samples,model_path\n")
    f.write(f"{datetime.now()},{accuracy:.4f},{f1:.4f},{f1_delta:.4f},{len(df)},{versioned_model_path}\n")

print(f"📈 Evaluation logged to {log_file}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Feature Importance Plot
importances = model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=feat_imp_df, x="Importance", y="Feature", color="blue")
plt.title("Feature Importance (Decision Tree)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# Plot F1 Score over time
try:
    df_eval = pd.read_csv("model_evaluation_log.csv")
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
    plt.show()
except Exception as e:
    print(f"⚠️ Could not plot F1 Score over time: {e}")
    
# Plot F1 Score, Accuracy, and F1 Delta over time
try:
    df_eval = pd.read_csv("model_evaluation_log.csv")
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
    plt.show()
except Exception as e:
    print(f"⚠️ Could not plot evaluation chart: {e}")
