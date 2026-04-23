"""
=============================================================
Credit Card Fraud Detection — Stage 4: Evaluation & Explainability
Author: Sai Ram Yasrapu
=============================================================
Produces all evaluation plots: ROC curve, Precision-Recall curve,
confusion matrix, and SHAP feature importance.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import os
from sklearn.metrics import (roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix, ConfusionMatrixDisplay)

sns.set_style("whitegrid")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})
OUTPUT_DIR = "outputs"
COLORS = {"legit": "#2E7D8C", "fraud": "#C0392B", "xgb": "#1B3A6B", "rf": "#2E7D8C", "lr": "#95A5A6"}

# ── Load Data & Predictions ───────────────────────────────────────────────────
print("Loading predictions and models...")
preds   = pd.read_csv(f"{OUTPUT_DIR}/predictions.csv")
X_test  = pd.read_csv(f"{OUTPUT_DIR}/X_test.csv")
y_test  = preds["y_true"]

with open(f"{OUTPUT_DIR}/xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# ── Plot 1: ROC Curve ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.set_title("ROC Curve — Model Comparison", fontsize=13, fontweight="bold")

for name, col, color in [
    ("XGBoost",             "xgb_probs", COLORS["xgb"]),
    ("Random Forest",       "rf_probs",  COLORS["rf"]),
    ("Logistic Regression", "lr_probs",  COLORS["lr"]),
]:
    fpr, tpr, _ = roc_curve(y_test, preds[col])
    auc = roc_auc_score(y_test, preds[col])
    ax.plot(fpr, tpr, label=f"{name}  (AUC = {auc:.4f})", linewidth=2, color=color)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier (AUC = 0.5)")
ax.set_xlabel("False Positive Rate\n(Legitimate transactions incorrectly flagged)")
ax.set_ylabel("True Positive Rate\n(Fraud cases correctly caught)")
ax.legend(loc="lower right")
ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: 06_roc_curve.png")

# ── Plot 2: Precision-Recall Curve ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.set_title("Precision-Recall Curve\n(Better metric for imbalanced datasets)",
             fontsize=13, fontweight="bold")

for name, col, color in [
    ("XGBoost",             "xgb_probs", COLORS["xgb"]),
    ("Random Forest",       "rf_probs",  COLORS["rf"]),
    ("Logistic Regression", "lr_probs",  COLORS["lr"]),
]:
    prec, rec, _ = precision_recall_curve(y_test, preds[col])
    ap = average_precision_score(y_test, preds[col])
    ax.plot(rec, prec, label=f"{name}  (AP = {ap:.4f})", linewidth=2, color=color)

baseline = y_test.mean()
ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1,
           label=f"Random Classifier (AP = {baseline:.4f})")
ax.set_xlabel("Recall\n(Fraction of fraud cases caught)")
ax.set_ylabel("Precision\n(Fraction of flagged transactions that are real fraud)")
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_precision_recall.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: 07_precision_recall.png")

# ── Plot 3: Confusion Matrix ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.set_title("Confusion Matrix — XGBoost\n(Optimal Threshold)", fontsize=13, fontweight="bold")

cm = confusion_matrix(y_test, preds["y_pred"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraud"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")

# Add annotations
tn, fp, fn, tp = cm.ravel()
ax.set_xlabel(f"Predicted Label\n\nTrue Negatives (correct legit): {tn:,}  |  "
              f"False Positives (legit flagged as fraud): {fp:,}\n"
              f"False Negatives (missed fraud): {fn:,}  |  "
              f"True Positives (fraud caught): {tp:,}")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: 08_confusion_matrix.png")

# ── Plot 4: SHAP Feature Importance ──────────────────────────────────────────
print("\nCalculating SHAP values (this takes ~1-2 minutes)...")

# Use a sample of 1000 rows for speed — SHAP is slow on 50k+ rows
sample_idx = np.random.choice(len(X_test), size=1000, replace=False)
X_sample   = X_test.iloc[sample_idx]

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)

# Summary plot — shows which features matter most overall
fig, ax = plt.subplots(figsize=(10, 9))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False,
                  color=COLORS["xgb"])
plt.title("SHAP Feature Importance — XGBoost\n(Mean impact on fraud prediction)",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: 09_shap_importance.png")

# SHAP beeswarm plot — shows direction of impact
fig, ax = plt.subplots(figsize=(10, 9))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP Beeswarm Plot — Direction of Feature Impact",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: 10_shap_beeswarm.png")

# ── Final Summary ─────────────────────────────────────────────────────────────
tn, fp, fn, tp = cm.ravel()
print("\n" + "="*55)
print("FINAL MODEL PERFORMANCE SUMMARY — XGBoost")
print("="*55)
print(f"ROC-AUC           : {roc_auc_score(y_test, preds['xgb_probs']):.4f}")
print(f"Average Precision : {average_precision_score(y_test, preds['xgb_probs']):.4f}")
print(f"Fraud caught      : {tp} / {tp+fn} ({tp/(tp+fn)*100:.1f}% recall)")
print(f"False alarms      : {fp} legitimate transactions incorrectly flagged")
print(f"Missed fraud      : {fn} fraud cases not caught")
print("\n✓ All evaluation plots saved to outputs/")
