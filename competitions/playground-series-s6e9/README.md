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
   cross-validated, Optuna-tuned.
4. `04_submission.ipynb` — final model, prediction on test set, submission file.

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

## Results

_TBD — filled in after first submission._

## Key takeaways

_TBD — filled in post-competition._
