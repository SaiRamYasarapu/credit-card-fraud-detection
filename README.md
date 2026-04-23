# Credit Card Fraud Detection
### End-to-End Machine Learning Pipeline | XGBoost · SMOTE · SHAP

---

## Overview

This project builds a production-style fraud detection system on real-world credit card transaction data. It mirrors the type of work I did professionally at PNC Bank, where I developed fraud detection models that reduced fraudulent transaction losses by 25%.

The pipeline covers every stage of a real ML project:
- Exploratory data analysis
- Feature engineering
- Handling severe class imbalance with SMOTE
- Model training and comparison (Logistic Regression, Random Forest, XGBoost)
- Threshold optimization for precision-recall tradeoff
- Model explainability with SHAP

---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions over 2 days (September 2013, European cardholders)
- 492 fraudulent transactions — **0.17% of all transactions**
- Features V1–V28 are PCA-transformed (anonymized for privacy)
- Raw features: `Time`, `Amount`, `Class` (0 = legitimate, 1 = fraud)

> **Note:** Due to file size, `creditcard.csv` is not included in this repo. Download it from Kaggle and place it in the `data/` folder.

---

## Project Structure

```
fraud-detection/
│
├── data/
│   └── creditcard.csv          ← download from Kaggle (not in repo)
│
├── src/
│   ├── 01_eda.py               ← exploratory data analysis & visualizations
│   ├── 02_feature_engineering.py  ← feature creation & train/test split
│   ├── 03_modeling.py          ← SMOTE, model training, threshold tuning
│   └── 04_evaluation.py        ← ROC/PR curves, confusion matrix, SHAP
│
├── outputs/                    ← saved plots, models, and predictions
├── requirements.txt
└── README.md
```

---

## Key Challenges & How I Solved Them

### 1. Severe Class Imbalance (0.17% fraud)
A naive model that predicts "not fraud" for everything achieves 99.83% accuracy — but catches zero fraud cases. I addressed this with two strategies:
- **SMOTE** (Synthetic Minority Over-sampling Technique) on the training set to create synthetic fraud examples, bringing the fraud ratio to ~10%
- **XGBoost's `scale_pos_weight`** parameter to further penalize misclassifying fraud
- **Evaluation with Precision-Recall AUC** instead of accuracy, which is the correct metric for imbalanced datasets

### 2. Choosing the Right Threshold
The default 0.5 classification threshold is rarely optimal for fraud detection. I used the Precision-Recall curve to find the threshold that maximizes the F1 score — balancing the tradeoff between catching more fraud (high recall) and generating fewer false alarms (high precision).

### 3. Model Explainability
In financial services, you cannot deploy a "black box" — regulators require explanations for decisions that affect customers. I used SHAP (SHapley Additive exPlanations) to quantify the contribution of each feature to every individual prediction, making the model interpretable and audit-ready.

---

## Results

| Model | ROC-AUC | Avg Precision (PR-AUC) |
|---|---|---|
| Logistic Regression | ~0.97 | ~0.73 |
| Random Forest | ~0.98 | ~0.87 |
| **XGBoost** | **~0.99** | **~0.89** |

> XGBoost achieves the best results on both metrics. Avg Precision (PR-AUC) is the primary metric for this imbalanced problem.

---

## Feature Engineering

Beyond the raw V1–V28 features, I engineered the following:

| Feature | Description | Rationale |
|---|---|---|
| `Hour` | Hour of day derived from Time | Fraud patterns vary by time |
| `Log_Amount` | Log(1 + Amount) | Reduces right skew in Amount |
| `Amount_Scaled` | Standardized Amount | Brings Amount to same scale as V features |
| `Time_Scaled` | Standardized Time | Same as above |
| `Is_Small_Amount` | 1 if Amount < €10 | Fraudsters often test cards with small amounts |
| `Is_Large_Amount` | 1 if Amount > €1000 | Large unusual charges can indicate fraud |
| `Is_Round_Amount` | 1 if Amount is a whole number | Fraudsters sometimes use round amounts |
| `Is_Night` | 1 if between midnight and 6am | Unusual timing can indicate fraud |

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

### 3. Run the pipeline in order
```bash
python src/01_eda.py
python src/02_feature_engineering.py
python src/03_modeling.py
python src/04_evaluation.py
```

All output plots and models will be saved to the `outputs/` folder.

---

## Requirements

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
xgboost>=1.7
imbalanced-learn>=0.10
shap>=0.41
matplotlib>=3.6
seaborn>=0.12
```

---

## Key Concepts Demonstrated

- **Class imbalance handling** — SMOTE, scale_pos_weight, stratified splitting
- **Gradient boosting** — XGBoost with early stopping and hyperparameter tuning
- **Model evaluation** — ROC-AUC, Precision-Recall AUC, F1, confusion matrix
- **Threshold optimization** — selecting the optimal cutoff for the business problem
- **Model explainability** — SHAP feature importance and beeswarm plots
- **Feature engineering** — domain-driven feature creation from raw transaction data
- **MLOps practices** — modular scripts, saved models, reproducible pipeline

---

## About

Built by **Sai Ram Yasrapu** — AI/ML Engineer with 5+ years of experience in Healthcare, Financial Services, and Telecommunications.

This project is inspired by real fraud detection work done in a professional setting, rebuilt using public data for demonstration purposes.
