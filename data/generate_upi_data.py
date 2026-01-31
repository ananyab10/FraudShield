import random
import uuid
import pandas as pd

rows = []

for i in range(3000):
    is_fraud = random.random() < 0.08  # ~8% fraud

    row = {
        "txn_id": str(uuid.uuid4()),
        "user_id": f"user_{random.randint(1, 400)}",
        "amount": random.randint(100, 5000),
        "txn_hour": random.randint(8, 22),
        "is_qr": 0,
        "beneficiary_age_min": random.randint(1440, 100000),
        "device_changed": 0,
        "location_velocity": 0,
        "failed_auth_24h": random.randint(0, 1),
        "label": 0
    }

    if is_fraud:
        row["amount"] = random.randint(8000, 30000)
        row["txn_hour"] = random.choice([0,1,2,3,23])
        row["is_qr"] = 1
        row["beneficiary_age_min"] = random.randint(1, 10)
        row["device_changed"] = 1
        row["location_velocity"] = 1
        row["failed_auth_24h"] = random.randint(2, 5)
        row["label"] = 1

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("upi_transactions.csv", index=False)
