import pandas as pd
from api.decision_engine import make_decision
from models.features import build_features

df = build_features("data/upi_transactions.csv")

# Test a high-risk transaction
test_row = df.iloc[[4]]  # pick a known fraud-like row
decision = make_decision(test_row)

print(decision)
