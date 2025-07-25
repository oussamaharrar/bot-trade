import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os


def main():
    df = pd.read_csv('training_dataset.csv')
    X = df.drop(columns=['pnl_class']).select_dtypes(include=['number'])
    y = df['pnl_class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = joblib.load(os.path.join('models', 'trained_model.pkl'))
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(
        f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}"
    )
    print(
        f"F1 Score: {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}"
    )


if __name__ == '__main__':
    main()
