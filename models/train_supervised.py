# models/train_supervised.py
"""
Train a supervised RandomForestClassifier for fraud detection.

- Loads engineered features via build_features(...)
- Loads labels from the original CSV ('label' column)
- Trains a RandomForestClassifier with class_weight="balanced"
- Prints classification report and ROC-AUC
- Saves trained model to models/fraud_model.pkl
"""
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from models.features import build_features


def load_features_and_labels(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    # Build features (assumed to return a DataFrame of numeric features)
    features = build_features(csv_path)
    # Load original CSV to get labels
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise KeyError(f"'label' column not found in {csv_path}")

    labels = df["label"]

    # Align indexes deterministically
    features = features.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    if len(features) != len(labels):
        raise ValueError(
            "Number of rows in engineered features does not match number of labels. "
            "Ensure build_features preserves row alignment with the original CSV."
        )

    return features, labels


def main():
    csv_path = "data/upi_transactions.csv"
    print(f"Loading features and labels from '{csv_path}'...")
    X_df, y = load_features_and_labels(csv_path)

    X = X_df.values
    y = y.values

    # Train/test split with stratification to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Initialize RandomForest with balanced class weights
    clf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)

    print("Fitting RandomForestClassifier...")
    clf.fit(X_train, y_train)

    # Predictions and probabilities for evaluation
    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
    else:
        # Fallback to decision_function if predict_proba is unavailable
        try:
            y_proba = clf.decision_function(X_test)
        except Exception:
            y_proba = None

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Print ROC-AUC if possible
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_proba)
            print(f"ROC-AUC: {auc:.4f}")
        except Exception as e:
            print(f"Could not compute ROC-AUC: {e}")
    else:
        print("Probability scores unavailable; skipping ROC-AUC.")

    # Persist the trained model
    model_path = "models/fraud_model.pkl"
    joblib.dump(clf, model_path)
    print(f"Trained model saved to '{model_path}'.")


if __name__ == "__main__":
    main()