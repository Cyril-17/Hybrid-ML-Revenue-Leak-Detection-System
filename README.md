1. Overview
This project implements a hybrid machine learning pipeline for detecting and prioritizing revenue leakage in invoice-level business transactions. The system integrates rule-based checks, supervised learning, anomaly detection, price deviation modelling, and salesperson behaviour clustering to generate a unified, explainable risk score. Designed as a lightweight and audit-ready revenue assurance engine.

2. Features
Rule-based leakage detection
Supervised model comparison (RF, LR, GBC)
Isolation Forest anomaly detection
Gradient Boosting price deviation modelling
K-Means salesperson behavioural clustering
Unified invoice-level risk scoring
Threshold analysis & Precision@K ranking
Full visualization suite (ROC, PR, Confusion Matrix, Feature Importance, etc.)

3. Tech Stack
Python
Scikit-learn
Pandas
NumPy
Matplotlib

4. Notebook Structure
data_preparation – cleaning & feature engineering
rule_labeling – generating weak labels from business rules
supervised_models – model training, comparison, evaluation
anomaly_models – Isolation Forest anomaly scoring
price_modeling – expected price prediction using GBR
clustering – salesperson behaviour segmentation
risk_score – unified leak-risk scoring
visualizations – ROC, PR, feature importance, threshold sweep, anomalies

5. Results
Best model: GradientBoostingClassifier
F1 Score: 1.000
Price Model MAE: 0.0159
Precision@10/20/50/100: 1.000
Threshold chosen: 0.40 (optimal precision–recall tradeoff)

6. Visualizations
Included plots:
ROC Curve (model comparison)
Precision–Recall Curve
Confusion Matrix
Feature Importance (GBC)
Threshold Sweep (Precision / Recall / F1)
Risk Score Distribution
Isolation Forest anomaly score histogram
Salesperson PCA cluster plot
Price deviation scatter plot

7. Future Enhancements
Integrating real confirmed leakage labels
Calibration on production audit data
API-based deployment for real-time scoring
Drift monitoring & model retraining automation
