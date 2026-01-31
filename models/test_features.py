from models.features import build_features

df = build_features("data/upi_transactions.csv")

print("Columns:")
print(df.columns.tolist())

print("\nSample rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nAny missing values?")
print(df.isna().sum())
