import pandas as pd
from api.decision_engine import make_decision

# Create a fraud-like transaction
row = pd.DataFrame([{
    "amount": 22000,
    "is_qr": 1,
    "device_changed": 1,
    "location_velocity": 1,
    "failed_auth_24h": 4,
    "amount_zscore": 2.1,
    "is_night": 1,
    "beneficiary_is_new": 1,
    "txn_velocity_24h": 5
}])

result = make_decision(row)
print(result)
