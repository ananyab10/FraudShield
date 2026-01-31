from datetime import datetime
from typing import Dict, List

ANALYST_ACTIONS: List[Dict] = []

def log_action(action: Dict):
    ANALYST_ACTIONS.append({
        **action,
        "timestamp": datetime.utcnow().isoformat()
    })
