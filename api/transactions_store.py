from datetime import datetime
from typing import List, Dict

# In-memory store (MVP only)
TRANSACTIONS: List[Dict] = []

def add_transaction(txn: Dict):
    TRANSACTIONS.insert(0, txn)  # newest first

def list_transactions(filter_decision: str | None = None):
    if not filter_decision or filter_decision == "ALL":
        return TRANSACTIONS
    return [t for t in TRANSACTIONS if t["decision"] == filter_decision]
