# api/audit_logger.py
"""
Audit logging for fraud decision system.

Records decision events to a JSONL file with hashed features for compliance
and audit trail purposes.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def log_decision(
    decision: str,
    risk_score: float,
    anomaly_score: float,
    reason_code: str,
    feature_row: pd.DataFrame,
    model_versions: Dict[str, str],
) -> None:
    """
    Log a fraud decision to the audit trail.

    Args:
        decision: Decision outcome ("ALLOW", "SOFT_BLOCK", "HARD_BLOCK")
        risk_score: Fraud probability from 0.0 to 1.0
        anomaly_score: Anomaly score from IsolationForest
        reason_code: Descriptive reason for decision
        feature_row: Single-row DataFrame with transaction features
        model_versions: Dictionary with model version information
    """
    # Ensure logs directory exists
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Create feature hash (SHA256 of serialized feature values)
    feature_values = feature_row.values.flatten()
    feature_str = ",".join(str(v) for v in feature_values)
    feature_hash = hashlib.sha256(feature_str.encode()).hexdigest()

    # Create audit record
    audit_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "risk_score": float(risk_score),
        "anomaly_score": float(anomaly_score),
        "reason_code": reason_code,
        "feature_hash": feature_hash,
        "model_versions": model_versions,
    }

    # Append to JSONL file (one JSON record per line)
    audit_log_path = logs_dir / "audit_log.jsonl"
    with open(audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit_record) + "\n")