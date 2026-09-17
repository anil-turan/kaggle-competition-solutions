# Playground Series S6E9 — Predicting Electric Vehicle Purchases

- **Competition**: https://www.kaggle.com/competitions/playground-series-s6e9
- **Type**: Binary classification (tabular, synthetic)
- **Target**: `Will_Buy_EV`
- **Deadline**: 2026-09-30
- **Metric**: see competition Overview page

## Problem

Predict whether a person will purchase an electric vehicle from demographic and
behavioural features: age, income, commute distance, cars owned, nearby charging
infrastructure, environmental concern, city type, current car type, home charging
availability, subsidy availability, range anxiety level.

## Approach

1. `01_eda.ipynb` — distributions, missingness, target balance, feature-target relationships.
2. `02_feature_engineering.ipynb` — encoding categoricals, derived features (e.g. charging
   access score, commute-to-income ratio).
3. `03_modeling.ipynb` — baseline (logistic regression) → gradient boosting (XGBoost/LightGBM),
   5-fold stratified CV, class-imbalance handling (`class_weight`/`scale_pos_weight`).
4. Next: Optuna hyperparameter tuning + LightGBM/XGBoost blend (not yet done).

## EDA findings (`01_eda.ipynb`)

- 668,665 train rows / 286,571 test rows, no missing values.
- Target imbalanced: ~17.5% `Yes` (Will_Buy_EV).
- Strongest numeric correlates with purchase: `Environmental_Concern_Level` (r=0.46),
  `Annual_Income_USD` (r=0.23). Commute distance, charging station counts, age, cars owned
  show negligible linear correlation — likely nonlinear/interaction effects better captured
  by tree-based models.
- Categorical purchase-rate spread: `Home_Charging_Possible=Yes` roughly 1.5x more likely to
  buy than `No`; `City_Type=Rural` slightly higher than `Urban`; `Current_Car_Type=SUV`
  highest among car types.

## Results (`03_modeling.ipynb`, 5-fold stratified CV, ROC-AUC)

| Model | Mean AUC | Std |
|---|---|---|
| Logistic Regression (baseline) | 0.9381 | 0.0008 |
| XGBoost | 0.9416 | 0.0008 |
| **LightGBM (best)** | **0.9417** | 0.0008 |

First submission (`submissions/submission_v1.csv`): LightGBM OOF predictions, unblended,
untuned (default learning rate, no Optuna pass yet).

## Key takeaways

- Gradient boosting only edges out the logistic regression baseline by ~0.004 AUC here —
  most of the signal in this dataset is close to linear/monotonic (dominated by
  `Environmental_Concern_Level` and `Annual_Income_USD`), so the ceiling for tree ensembles
  may be modest without more feature engineering.
- LightGBM and XGBoost land within 0.0001 AUC of each other — a simple average blend is
  worth trying before spending Optuna budget on either individually.
- LightGBM hit `n_estimators=1000` (the cap) on 4/5 folds without early stopping triggering —
  worth raising the cap or lowering the learning rate to see if there's AUC left on the table.
- Not yet tried: target/frequency encoding of `Current_Car_Type`/`City_Type` instead of
  one-hot, and interaction features between `Environmental_Concern_Level` and income/charging
  access.
