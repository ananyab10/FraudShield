# models/train_anomaly.py
"""
Train an IsolationForest anomaly detector on legitimate transactions.

- Loads engineered features via build_features(...)
- Loads original CSV and selects only label==0 (legit) for training
- Fits IsolationForest on legit transactions
- Computes anomaly scores for all data and prints summary statistics
- Saves trained model to models/anomaly_model.pkl
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from models.features import build_features


def load_features_and_labels(csv_path: str):
    features = build_features(csv_path).reset_index(drop=True)
    df = pd.read_csv(csv_path).reset_index(drop=True)
    if "label" not in df.columns:
        raise KeyError(f"'label' column not found in {csv_path}")
    labels = df["label"]
    if len(features) != len(labels):
        raise ValueError(
            "Number of rows in engineered features does not match number of labels. "
            "Ensure build_features preserves row alignment with the original CSV."
        )
    return features, labels


def summary_stats(arr: np.ndarray) -> str:
    pct = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    stats = {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "1%": float(pct[0]),
        "5%": float(pct[1]),
        "25%": float(pct[2]),
        "50% (median)": float(pct[3]),
        "75%": float(pct[4]),
        "95%": float(pct[5]),
        "99%": float(pct[6]),
    }
    lines = [f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()]
    return "\n".join(lines)


def main():
    csv_path = "data/upi_transactions.csv"
    print(f"Loading features and labels from '{csv_path}'...")
    features, labels = load_features_and_labels(csv_path)

    # Select only legitimate transactions (label == 0) for training
    legit_mask = labels == 0
    X_legit = features[legit_mask]
    if X_legit.shape[0] == 0:
        raise ValueError("No legitimate (label==0) transactions found for training.")

    # Fit IsolationForest on legitimate transactions only
    iso = IsolationForest(random_state=42)
    print(f"Fitting IsolationForest on {X_legit.shape[0]} legitimate transactions...")
    iso.fit(X_legit)

    # Compute anomaly scores for all transactions
    # decision_function: higher means more normal; lower means more anomalous
    scores = iso.decision_function(features.values)

    # Print summary statistics of the anomaly scores
    print("\nAnomaly score summary (higher == more normal):")
    print(summary_stats(scores))

    # Save trained anomaly model
    model_path = "models/anomaly_model.pkl"
    joblib.dump(iso, model_path)
    print(f"\nAnomaly model saved to '{model_path}'.")


if __name__ == "__main__":
    main()