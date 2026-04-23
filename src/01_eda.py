"""
=============================================================
Credit Card Fraud Detection — Stage 1: Exploratory Data Analysis
Author: Sai Ram Yasrapu
=============================================================
This script loads the dataset and produces visualizations to
understand the data before modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ── Settings ──────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
COLORS = {"legit": "#2E7D8C", "fraud": "#C0392B"}
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/creditcard.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nClass distribution:\n{df['Class'].value_counts()}")
print(f"Fraud rate: {df['Class'].mean()*100:.4f}%")
print(f"Missing values: {df.isnull().sum().sum()}")

# ── Plot 1: Class Imbalance ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Class Distribution — Fraud vs Legitimate Transactions", fontsize=14, fontweight="bold")

counts = df["Class"].value_counts()
labels = ["Legitimate (0)", "Fraud (1)"]
colors = [COLORS["legit"], COLORS["fraud"]]

axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
axes[0].set_title("Transaction Count")
axes[0].set_ylabel("Number of Transactions")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 1000, f"{v:,}", ha="center", fontweight="bold")

axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.4f%%",
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Percentage Split")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_class_imbalance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Saved: 01_class_imbalance.png")

# ── Plot 2: Transaction Amount Distribution ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Transaction Amount: Fraud vs Legitimate", fontsize=14, fontweight="bold")

fraud    = df[df["Class"] == 1]["Amount"]
legit    = df[df["Class"] == 0]["Amount"]

axes[0].hist(legit, bins=80, color=COLORS["legit"], alpha=0.7, label="Legitimate", edgecolor="none")
axes[0].hist(fraud, bins=80, color=COLORS["fraud"], alpha=0.85, label="Fraud", edgecolor="none")
axes[0].set_xlim(0, 500)
axes[0].set_xlabel("Transaction Amount (€)")
axes[0].set_ylabel("Count")
axes[0].set_title("Amount Distribution (0–€500)")
axes[0].legend()

axes[1].boxplot([legit.clip(upper=500), fraud.clip(upper=500)],
                labels=["Legitimate", "Fraud"],
                patch_artist=True,
                boxprops=dict(facecolor=COLORS["legit"], alpha=0.6),
                medianprops=dict(color="black", linewidth=2))
axes[1].set_ylabel("Transaction Amount (€, clipped at 500)")
axes[1].set_title("Amount Boxplot")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_amount_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: 02_amount_distribution.png")

# ── Plot 3: Transaction Time Distribution ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
ax.set_title("Transaction Timing: When Do Fraudulent Transactions Occur?",
             fontsize=13, fontweight="bold")

# Convert seconds to hours
df["Hour"] = (df["Time"] / 3600) % 24

fraud_hours = df[df["Class"] == 1]["Hour"]
legit_hours = df[df["Class"] == 0]["Hour"]

ax.hist(legit_hours, bins=48, color=COLORS["legit"], alpha=0.6, label="Legitimate", density=True)
ax.hist(fraud_hours, bins=48, color=COLORS["fraud"], alpha=0.8, label="Fraud", density=True)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_time_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: 03_time_distribution.png")

# ── Plot 4: Feature Correlation Heatmap ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")

corr = df.drop(columns=["Time"]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0,
            linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.8},
            annot=False, fmt=".1f")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: 04_correlation_heatmap.png")

# ── Plot 5: Top Features Separating Fraud from Legit ─────────────────────────
# Find features with biggest mean difference between fraud and legit
feature_cols = [c for c in df.columns if c.startswith("V")]
fraud_means  = df[df["Class"] == 1][feature_cols].mean()
legit_means  = df[df["Class"] == 0][feature_cols].mean()
diff         = (fraud_means - legit_means).abs().sort_values(ascending=False)
top_features = diff.head(8).index.tolist()

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Top 8 Features — Distribution: Fraud vs Legitimate",
             fontsize=13, fontweight="bold")

for i, feat in enumerate(top_features):
    ax = axes[i // 4][i % 4]
    ax.hist(df[df["Class"] == 0][feat].clip(-10, 10), bins=60,
            color=COLORS["legit"], alpha=0.6, label="Legit", density=True)
    ax.hist(df[df["Class"] == 1][feat].clip(-10, 10), bins=60,
            color=COLORS["fraud"], alpha=0.8, label="Fraud", density=True)
    ax.set_title(feat, fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    if i == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_top_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: 05_top_features.png")

# ── Summary Stats ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("EDA SUMMARY")
print("="*55)
print(f"Total transactions  : {len(df):,}")
print(f"Legitimate          : {(df['Class']==0).sum():,} ({(df['Class']==0).mean()*100:.2f}%)")
print(f"Fraud               : {(df['Class']==1).sum():,} ({(df['Class']==1).mean()*100:.4f}%)")
print(f"Avg legitimate amt  : €{legit.mean():.2f}")
print(f"Avg fraud amount    : €{fraud.mean():.2f}")
print(f"Max fraud amount    : €{fraud.max():.2f}")
print(f"Missing values      : {df.isnull().sum().sum()}")
print(f"\nTop discriminating features: {top_features}")
print("\n✓ All EDA plots saved to outputs/ folder")
