---
title: Gradient boosting for tabular data (XGBoost, LightGBM, CatBoost)
category: ml
tags: [xgboost, lightgbm, catboost, gradient-boosting, tabular, early-stopping, shap]
use_cases:
  - "predict credit default or fraud from transaction tables with the best possible accuracy"
  - "rank leads or predict customer lifetime value from CRM data"
  - "build a demand or price regression model on mixed categorical and numeric features"
  - "win a Kaggle-style tabular competition baseline quickly"
  - "explain feature contributions of a risk model for compliance"
status: draft
last_verified: 2026-10-03
sources:
  - https://xgboost.readthedocs.io/en/stable/python/python_intro.html
  - https://lightgbm.readthedocs.io/en/stable/
  - https://catboost.ai/docs/en/
---

# Gradient boosting for tabular data (XGBoost, LightGBM, CatBoost)

## Summary
Gradient-boosted decision trees build many small trees sequentially, each correcting the previous errors. On structured/tabular data they are still the strongest general default, usually beating neural networks with far less tuning. The three main libraries are XGBoost, LightGBM and CatBoost; all have scikit-learn-compatible Python APIs.

## Key concepts
- Boosting: additive trees fitted to gradients of the loss; `learning_rate` x `n_estimators` trade off.
- Early stopping on a validation set picks the number of trees automatically.
- Tree complexity controls: `max_depth` / `num_leaves`, `min_child_weight` / `min_data_in_leaf`, subsampling, L1/L2 regularisation.
- Native handling of missing values (all three); native categorical support in LightGBM and especially CatBoost (ordered target statistics).
- Objectives: binary/multiclass logloss, regression (L2/L1/Huber), ranking (LambdaRank), quantile, Poisson/Tweedie for counts and claims.
- Interpretation: feature importance, SHAP (`TreeExplainer`), partial dependence.

## When to use / scenarios
- Signals: tabular data, 1k to 100M rows, mixed types, nonlinear interactions, need for accuracy.
- Finance: fraud and credit scoring (with monotonic constraints for regulators). Insurance: Tweedie claim cost. E-commerce: conversion and ranking models. Logistics: ETA regression.
- NOT for: images/audio/text (use [[pytorch-basics]] or [[nlp-classic-tasks]]), extrapolating trends beyond training range (trees cannot extrapolate; see [[time-series-forecasting]]), tiny datasets where a linear model suffices ([[classic-ml-scikit-learn]]).

## Setup & code
```bash
pip install xgboost lightgbm catboost scikit-learn
```
```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, stratify=y, random_state=0)

model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=31,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
print(roc_auc_score(y_va, model.predict_proba(X_va)[:, 1]), model.best_iteration_)
```
XGBoost equivalent: `xgboost.XGBClassifier(n_estimators=2000, learning_rate=0.03, early_stopping_rounds=50).fit(X_tr, y_tr, eval_set=[(X_va, y_va)])`. CatBoost: `CatBoostClassifier(iterations=2000, early_stopping_rounds=50, cat_features=[...])`.

## Choosing / trade-offs
- LightGBM: fastest on large data, leaf-wise growth, good default; can overfit small data if `num_leaves` is high.
- XGBoost: very mature, strong GPU and distributed support, broad ecosystem.
- CatBoost: best out-of-the-box with many high-cardinality categoricals, less tuning, symmetric trees give fast inference.
- Tuning: Optuna over learning rate, leaves/depth, regularisation; log runs with [[experiment-tracking]].
- Versus neural nets: only consider them for very large data, multimodal inputs, or embeddings-heavy features.

## Gotchas
- Using the test set for early stopping leaks; keep a separate validation set or nested CV.
- High-cardinality categoricals one-hot encoded blow up memory; use native categorical handling.
- Target/ID leakage and time leakage: split by time or group where rows are related.
- Trees cannot extrapolate; add trend features or model differences for forecasting.
- `scale_pos_weight` / class weights distort probabilities; calibrate if you need real probabilities.
- Default feature importance is biased toward high-cardinality features; prefer SHAP or permutation importance.
- Same hyperparameter name means different things across libraries (`num_leaves` vs `max_depth`).

## Related
- [[classic-ml-scikit-learn]] - preprocessing, CV, metrics; boosters plug into sklearn pipelines.
- [[experiment-tracking]] - log tuning runs.
- [[anomaly-detection]] - supervised boosting beats unsupervised when labelled fraud exists.
- [[recommender-systems]] - boosting is common as a ranking stage.

## References
- https://xgboost.readthedocs.io/en/stable/python/python_intro.html
- https://lightgbm.readthedocs.io/en/stable/
- https://catboost.ai/docs/en/
