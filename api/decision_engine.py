"""
Fraud decision engine for real-time transaction evaluation.

Loads pre-trained RandomForest and IsolationForest models and applies
explicit decision rules to classify transactions as:
ALLOW, SOFT_BLOCK, or HARD_BLOCK.

This module is:
- Deterministic
- Stateless
- Safe to call in real-time
"""

from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import pandas as pd


# ---------------------------------------------------------------------
# Model loading (done once at module import)
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

_FRAUD_MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
_ANOMALY_MODEL_PATH = BASE_DIR / "models" / "anomaly_model.pkl"

_fraud_model = joblib.load(_FRAUD_MODEL_PATH)
_anomaly_model = joblib.load(_ANOMALY_MODEL_PATH)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def make_decision(feature_row: pd.DataFrame) -> Dict[str, Any]:
    """
    Make a fraud decision for a single transaction.

    Args:
        feature_row:
            Single-row pandas DataFrame with numeric features.
            Must include:
            - is_qr
            - beneficiary_is_new

    Returns:
        dict with:
        - decision: "ALLOW" | "SOFT_BLOCK" | "HARD_BLOCK"
        - risk_score: float (0.0 – 1.0)
        - anomaly_score: float (lower = more anomalous)
        - reason_code: str
    """

    # ----------------------------
    # Defensive checks
    # ----------------------------
    if not isinstance(feature_row, pd.DataFrame):
        raise ValueError("feature_row must be a pandas DataFrame")

    if feature_row.shape[0] != 1:
        raise ValueError("feature_row must contain exactly one row")

    # IMPORTANT:
    # Always pass a DataFrame (not .values) to avoid sklearn warnings
    X = feature_row.copy()

    # ----------------------------
    # Fraud probability
    # ----------------------------
    if hasattr(_fraud_model, "predict_proba"):
        fraud_probability = float(_fraud_model.predict_proba(X)[0, 1])
    else:
        # Fallback (should not happen in our setup)
        fraud_probability = float(_fraud_model.predict(X)[0])

    # ----------------------------
    # Anomaly score
    # (lower = more suspicious)
    # ----------------------------
    anomaly_score = float(_anomaly_model.decision_function(X)[0])

    # ----------------------------
    # Extract rule features
    # ----------------------------
    is_qr = int(feature_row["is_qr"].iloc[0])
    beneficiary_is_new = int(feature_row["beneficiary_is_new"].iloc[0])

    # ----------------------------
    # Apply explicit rules
    # ----------------------------
    decision, reason_code = _apply_decision_rules(
        fraud_probability=fraud_probability,
        anomaly_score=anomaly_score,
        is_qr=is_qr,
        beneficiary_is_new=beneficiary_is_new,
    )

    return {
        "decision": decision,
        "risk_score": fraud_probability,
        "anomaly_score": anomaly_score,
        "reason_code": reason_code,
    }


# ---------------------------------------------------------------------
# Internal rule engine
# ---------------------------------------------------------------------

def _apply_decision_rules(
    fraud_probability: float,
    anomaly_score: float,
    is_qr: int,
    beneficiary_is_new: int,
) -> Tuple[str, str]:
    """
    Apply explicit, regulator-friendly decision rules.

    Returns:
        (decision, reason_code)
    """

    # Thresholds (MVP – intentionally simple)
    HIGH_FRAUD_THRESHOLD = 0.7
    SOFT_FRAUD_THRESHOLD = 0.5

    HIGH_ANOMALY_THRESHOLD = -0.15
    SOFT_ANOMALY_THRESHOLD = -0.10

    # Flags
    high_fraud = fraud_probability >= HIGH_FRAUD_THRESHOLD
    high_anomaly = anomaly_score <= HIGH_ANOMALY_THRESHOLD

    soft_fraud = fraud_probability >= SOFT_FRAUD_THRESHOLD
    soft_anomaly = anomaly_score <= SOFT_ANOMALY_THRESHOLD

    qr_payment = is_qr == 1
    new_beneficiary = beneficiary_is_new == 1

    # --------------------------------------------------
    # HARD BLOCK
    # --------------------------------------------------
    if high_fraud and high_anomaly and qr_payment and new_beneficiary:
        return (
            "HARD_BLOCK",
            "QR_NEW_BENEFICIARY_HIGH_FRAUD_HIGH_ANOMALY",
        )

    # --------------------------------------------------
    # SOFT BLOCK
    # --------------------------------------------------
    if soft_fraud or soft_anomaly:
        reasons = []
        if soft_fraud:
            reasons.append("FRAUD_SIGNAL")
        if soft_anomaly:
            reasons.append("ANOMALY_SIGNAL")

        return (
            "SOFT_BLOCK",
            "_".join(reasons),
        )

    # --------------------------------------------------
    # ALLOW
    # --------------------------------------------------
    return "ALLOW", "NO_SIGNIFICANT_RISK"
