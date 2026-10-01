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
4. `04_tuning_and_blend.ipynb` — Optuna tuning of both boosters + rank-average blend.

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

First submission (`submissions/submission_v1.csv`): LightGBM test predictions averaged across
the 5 fold models, unblended, untuned.

## Tuning + blend (`04_tuning_and_blend.ipynb`)

Optuna (TPE) search on a single 80/20 stratified hold-out — 30 trials for LightGBM, 20 for
XGBoost — then the best configuration of each re-trained with the same 5 folds as notebook 03.
The blend averages rank-normalised predictions, with the weight chosen on OOF AUC.

| Model | OOF AUC |
|---|---|
| LightGBM baseline (nb 03) | 0.94170 |
| LightGBM tuned | 0.94192 |
| XGBoost tuned | 0.94199 |
| **Blend (0.45 LightGBM / 0.55 XGBoost)** | **0.94207** |

Both searches converged on shallow, heavily regularised trees (LightGBM `num_leaves=17`,
`min_child_samples=189`; XGBoost `max_depth=3`) — consistent with the EDA finding that the
signal is close to monotonic.

## Leaderboard

| Submission | CV AUC | Public LB | Private LB |
|---|---|---|---|
| v1 — LightGBM baseline | 0.9417 | 0.94144 | 0.94091 |
| v2 — tuned blend (late submission) | 0.94207 | 0.94186 | 0.94122 |

CV and public leaderboard agree to within 0.0003 for both submissions, so the local 5-fold
setup is a reliable proxy for the hidden test set. The blend's +0.0004 OOF gain carried over
to the leaderboard (+0.0003 private). v2 was finished after the 2026-09-30 deadline, so it is
a late submission and does not count towards the final ranking.

## Key takeaways

- Gradient boosting only edges out the logistic regression baseline by ~0.004 AUC here —
  most of the signal in this dataset is close to linear/monotonic (dominated by
  `Environmental_Concern_Level` and `Annual_Income_USD`), so the ceiling for tree ensembles
  may be modest without more feature engineering.
- Tuning and blending together added only ~0.0004 OOF AUC over the untuned baseline — with
  a near-linear signal, model capacity is not the bottleneck; feature engineering is.
- Not yet tried: target/frequency encoding of `Current_Car_Type`/`City_Type` instead of
  one-hot, and interaction features between `Environmental_Concern_Level` and income/charging
  access.
