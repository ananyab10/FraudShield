"""
FastAPI application for real-time fraud decision making.

Exposes:
- POST /decision        → real-time fraud decision
- GET  /transactions    → analyst transaction queue
- POST /explain         → post-decision explanation (RAG-based)
- POST /analyst/action  → analyst override actions
- GET  /health          → health check
"""

from datetime import datetime
import uuid
from typing import List, Dict, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from api.decision_engine import make_decision
from rag.explainer import explain_decision

# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------

app = FastAPI(title="FraudShield API", version="1.0.0")

# ---------------------------------------------------------------------
# In-memory stores (MVP ONLY)
# ---------------------------------------------------------------------

TRANSACTIONS: List[Dict] = []
ANALYST_ACTIONS: List[Dict] = []

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SAFE_ALLOW_RESPONSE = {
    "decision": "ALLOW",
    "risk_score": 0.0,
    "anomaly_score": 0.0,
    "reason_code": "INPUT_VALIDATION_FAILED",
}

# ---------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------

class TransactionPayload(BaseModel):
    amount: float = Field(..., gt=0)
    txn_hour: int = Field(..., ge=0, le=23)
    is_qr: int = Field(..., ge=0, le=1)
    beneficiary_age_min: int = Field(..., ge=0)
    device_changed: int = Field(..., ge=0, le=1)
    location_velocity: int = Field(..., ge=0)
    failed_auth_24h: int = Field(..., ge=0)


class ExplainPayload(BaseModel):
    reason_code: str = Field(..., min_length=1)


class AnalystActionPayload(BaseModel):
    txn_id: str
    action: str = Field(..., description="CONFIRM_FRAUD | FALSE_POSITIVE | ESCALATE")
    notes: Optional[str] = None

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def add_transaction(record: Dict):
    TRANSACTIONS.insert(0, record)  # newest first


def list_transactions(filter_decision: str = "ALL") -> List[Dict]:
    if filter_decision == "ALL":
        return TRANSACTIONS
    return [t for t in TRANSACTIONS if t["decision"] == filter_decision]


def log_analyst_action(action: Dict):
    ANALYST_ACTIONS.append({
        **action,
        "timestamp": datetime.utcnow().isoformat()
    })

# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------

@app.post("/decision")
def get_fraud_decision(payload: TransactionPayload) -> dict:
    """
    Real-time fraud decision endpoint.
    Deterministic, fail-safe, and auditable.
    """
    try:
        feature_row = pd.DataFrame([{
            "amount": payload.amount,
            "txn_hour": payload.txn_hour,
            "is_qr": payload.is_qr,
            "beneficiary_age_min": payload.beneficiary_age_min,
            "device_changed": payload.device_changed,
            "location_velocity": payload.location_velocity,
            "failed_auth_24h": payload.failed_auth_24h,
        }])

        decision = make_decision(feature_row)

        txn_record = {
            "txn_id": f"txn_{uuid.uuid4().hex[:8]}",
            "amount": payload.amount,
            "decision": decision["decision"],
            "risk_score": decision["risk_score"],
            "anomaly_score": decision["anomaly_score"],
            "reason_code": decision["reason_code"],
            "timestamp": datetime.utcnow().isoformat(),
        }

        add_transaction(txn_record)
        return decision

    except Exception:
        return SAFE_ALLOW_RESPONSE


@app.get("/transactions")
def get_transactions(filter: str = "ALL") -> List[Dict]:
    """
    Analyst transaction queue.
    filter = ALL | SOFT_BLOCK | HARD_BLOCK | ALLOW
    """
    return list_transactions(filter)


@app.post("/explain")
async def explain(payload: ExplainPayload) -> dict:
    """
    Post-decision explanation endpoint (RAG).
    Never affects decisioning.
    """
    try:
        explanation = explain_decision(payload.reason_code)
        return {
            "reason_code": payload.reason_code,
            "explanation": explanation,
        }
    except Exception:
        return {
            "reason_code": payload.reason_code,
            "explanation": "Explanation temporarily unavailable.",
        }


@app.post("/analyst/action")
def analyst_action(payload: AnalystActionPayload) -> dict:
    """
    Human-in-the-loop analyst override logging.
    Does NOT change original system decision.
    """
    log_analyst_action(payload.dict())
    return {"status": "logged"}


@app.get("/health")
def health_check() -> dict:
    return {"status": "healthy"}
