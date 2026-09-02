"""
=============================================================
Credit Card Fraud Detection — Stage 2: Feature Engineering
Author: Sai Ram Yasrapu
=============================================================
This script adds new features on top of the raw data and
prepares a clean train/test split ready for modeling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("C:/Users/Sai Ram/Desktop/git_projects/credit-card-fraud-detection/creditcard.csv")
print(f"Raw shape: {df.shape}")

# ── Feature Engineering ───────────────────────────────────────────────────────
print("\nEngineering features...")

# 1. Hour of day — fraud patterns vary by time of day
df["Hour"] = (df["Time"] / 3600) % 24

# 2. Is it a small amount? — many fraud transactions are small "test" charges
df["Is_Small_Amount"] = (df["Amount"] < 10).astype(int)

# 3. Is it a large amount? — some fraud involves large unusual charges
df["Is_Large_Amount"] = (df["Amount"] > 1000).astype(int)

# 4. Log of amount — reduces the skew in the Amount distribution
#    We add 1 before log to handle zero-value transactions
df["Log_Amount"] = np.log1p(df["Amount"])

# 5. Is it a round number? — fraudsters sometimes use round amounts
df["Is_Round_Amount"] = (df["Amount"] % 1 == 0).astype(int)

# 6. Night transaction flag — between midnight and 6am
df["Is_Night"] = ((df["Hour"] >= 0) & (df["Hour"] < 6)).astype(int)

# 7. Scale Amount and Time — V1-V28 are already scaled (PCA output),
#    but Amount and Time are raw and need scaling
scaler = StandardScaler()
df["Amount_Scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_Scaled"]   = scaler.fit_transform(df[["Time"]])

print(f"New features added: Hour, Is_Small_Amount, Is_Large_Amount,")
print(f"                    Log_Amount, Is_Round_Amount, Is_Night,")
print(f"                    Amount_Scaled, Time_Scaled")

# ── Define Feature Set ────────────────────────────────────────────────────────
# Drop raw Amount and Time — we use the engineered versions instead
v_features      = [f"V{i}" for i in range(1, 29)]
new_features    = ["Amount_Scaled", "Time_Scaled", "Hour", "Log_Amount",
                   "Is_Small_Amount", "Is_Large_Amount", "Is_Round_Amount", "Is_Night"]
all_features    = v_features + new_features

X = df[all_features]
y = df["Class"]

print(f"\nTotal features: {len(all_features)}")
print(f"  - Original V features : {len(v_features)}")
print(f"  - Engineered features : {len(new_features)}")

# ── Train / Test Split ────────────────────────────────────────────────────────
# stratify=y ensures both splits have the same fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # critical for imbalanced datasets
)

print(f"\nTrain size : {len(X_train):,} rows")
print(f"Test size  : {len(X_test):,} rows")
print(f"\nTrain fraud rate : {y_train.mean()*100:.4f}%")
print(f"Test fraud rate  : {y_test.mean()*100:.4f}%")
print("(Both rates should be similar — stratify= is working correctly)")

# ── Save Processed Data ───────────────────────────────────────────────────────
X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

print(f"\n✓ Saved train/test splits to {OUTPUT_DIR}/")
print("\nFeature engineering complete — ready for modeling.")

# ── Print Sample of Engineered Features ──────────────────────────────────────
print("\nSample of engineered features (first 5 rows):")
print(df[new_features].head())
