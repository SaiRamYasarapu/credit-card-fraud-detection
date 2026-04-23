"""
=============================================================
Credit Card Fraud Detection — Stage 3: Modeling
Author: Sai Ram Yasrapu
=============================================================
Trains an XGBoost classifier with SMOTE to handle class
imbalance. Saves the trained model for evaluation.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report)
import xgboost as xgb
from imblearn.over_sampling import SMOTE

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Prepared Data ────────────────────────────────────────────────────────
print("Loading prepared data...")
X_train = pd.read_csv(f"{OUTPUT_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{OUTPUT_DIR}/X_test.csv")
y_train = pd.read_csv(f"{OUTPUT_DIR}/y_train.csv").squeeze()
y_test  = pd.read_csv(f"{OUTPUT_DIR}/y_test.csv").squeeze()

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train fraud cases: {y_train.sum():,}")

# ── Handle Class Imbalance with SMOTE ────────────────────────────────────────
# SMOTE = Synthetic Minority Over-sampling Technique
# It creates synthetic fraud examples so the model sees a more balanced dataset
# We only apply SMOTE to the TRAINING set — never to the test set
print("\nApplying SMOTE to training data...")
print(f"Before SMOTE — Legitimate: {(y_train==0).sum():,}  |  Fraud: {(y_train==1).sum():,}")

smote = SMOTE(random_state=42, sampling_strategy=0.1)
# sampling_strategy=0.1 means fraud will be 10% of legit after oversampling
# We don't go to 50/50 because that's too artificial — 10% is more realistic
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"After SMOTE  — Legitimate: {(y_train_sm==0).sum():,}  |  Fraud: {(y_train_sm==1).sum():,}")

# ── Model 1: Logistic Regression (Baseline) ───────────────────────────────────
print("\n" + "="*55)
print("Training Model 1: Logistic Regression (Baseline)")
print("="*55)

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train_sm, y_train_sm)

lr_probs = lr.predict_proba(X_test)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_probs)
lr_ap    = average_precision_score(y_test, lr_probs)

print(f"ROC-AUC          : {lr_auc:.4f}")
print(f"Avg Precision    : {lr_ap:.4f}")
print("(Avg Precision = area under precision-recall curve — better metric for imbalanced data)")

# ── Model 2: Random Forest ────────────────────────────────────────────────────
print("\n" + "="*55)
print("Training Model 2: Random Forest")
print("="*55)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
rf.fit(X_train_sm, y_train_sm)

rf_probs = rf.predict_proba(X_test)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_probs)
rf_ap    = average_precision_score(y_test, rf_probs)

print(f"ROC-AUC          : {rf_auc:.4f}")
print(f"Avg Precision    : {rf_ap:.4f}")

# ── Model 3: XGBoost (Main Model) ─────────────────────────────────────────────
print("\n" + "="*55)
print("Training Model 3: XGBoost (Main Model)")
print("="*55)

# scale_pos_weight handles class imbalance in XGBoost
# It tells the model to penalize missing fraud more than missing legit
fraud_count = (y_train_sm == 1).sum()
legit_count = (y_train_sm == 0).sum()
scale_pos_weight = legit_count / fraud_count
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="aucpr",        # optimize for precision-recall AUC
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20
)

# Use a validation set inside training for early stopping
from sklearn.model_selection import train_test_split as tts
X_tr2, X_val, y_tr2, y_val = tts(X_train_sm, y_train_sm,
                                   test_size=0.1, random_state=42, stratify=y_train_sm)

xgb_model.fit(
    X_tr2, y_tr2,
    eval_set=[(X_val, y_val)],
    verbose=50
)

xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc   = roc_auc_score(y_test, xgb_probs)
xgb_ap    = average_precision_score(y_test, xgb_probs)

print(f"\nROC-AUC          : {xgb_auc:.4f}")
print(f"Avg Precision    : {xgb_ap:.4f}")

# ── Model Comparison ──────────────────────────────────────────────────────────
print("\n" + "="*55)
print("MODEL COMPARISON SUMMARY")
print("="*55)
print(f"{'Model':<25} {'ROC-AUC':>10} {'Avg Precision':>15}")
print("-"*52)
print(f"{'Logistic Regression':<25} {lr_auc:>10.4f} {lr_ap:>15.4f}")
print(f"{'Random Forest':<25} {rf_auc:>10.4f} {rf_ap:>15.4f}")
print(f"{'XGBoost':<25} {xgb_auc:>10.4f} {xgb_ap:>15.4f}")
print(f"\n✓ Best model: XGBoost (ROC-AUC: {xgb_auc:.4f})")

# ── Optimal Threshold Selection ───────────────────────────────────────────────
# Default threshold is 0.5 but for fraud detection we may want to adjust
# to optimize for recall (catching more fraud) vs precision (fewer false alarms)
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, xgb_probs)

# Find threshold that gives best F1 score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx   = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\nOptimal threshold (best F1): {best_threshold:.4f}")
print(f"At this threshold:")
print(f"  Precision : {precision[best_idx]:.4f}  (of all flagged, this % are real fraud)")
print(f"  Recall    : {recall[best_idx]:.4f}  (of all real fraud, this % are caught)")
print(f"  F1 Score  : {f1_scores[best_idx]:.4f}")

# Final predictions using optimal threshold
y_pred_final = (xgb_probs >= best_threshold).astype(int)
print(f"\nClassification Report (threshold={best_threshold:.2f}):")
print(classification_report(y_test, y_pred_final, target_names=["Legitimate", "Fraud"]))

# ── Save Models ───────────────────────────────────────────────────────────────
with open(f"{OUTPUT_DIR}/xgb_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
with open(f"{OUTPUT_DIR}/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Save predictions for evaluation script
pd.DataFrame({
    "y_true"     : y_test.values,
    "xgb_probs"  : xgb_probs,
    "rf_probs"   : rf_probs,
    "lr_probs"   : lr_probs,
    "y_pred"     : y_pred_final
}).to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)

print(f"\n✓ Models and predictions saved to {OUTPUT_DIR}/")
print("Ready for Stage 4: Evaluation & Visualization")
