import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/fraud_model.joblib"


def train_model():

    # Check that dataset exists
    if not os.path.exists(DATA_PATH):
        print("ERROR: creditcard.csv was not found!")
        print("Make sure it is inside the data folder.")
        return

    print("Loading real credit card fraud dataset...")

    data = pd.read_csv(DATA_PATH)

    print(f"Dataset loaded successfully!")
    print(f"Number of transactions: {len(data)}")
    print(f"Number of columns: {len(data.columns)}")

    # Separate features and target
    X = data.drop("Class", axis=1)
    y = data["Class"]

    print("\nFraud distribution:")
    print(y.value_counts())

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Random Forest model
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    print("\nTraining Random Forest model...")
    print("Please wait...")

    model.fit(X_train, y_train)

    print("Model training completed! ✅")

    # Predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print("\n" + "=" * 50)
    print(f"Model saved to: {MODEL_PATH}")
    print("Real dataset model is ready! ✅")
    print("=" * 50)


if __name__ == "__main__":
    train_model()