---
title: Classic ML with scikit-learn
category: ml
tags: [scikit-learn, sklearn, classification, regression, clustering, pipeline, cross-validation, tabular]
use_cases:
  - "build a baseline churn classifier on customer CSV data"
  - "predict house prices or insurance claim cost from tabular features"
  - "segment retail customers into groups with clustering"
  - "build a leak-free preprocessing + model pipeline with cross-validation"
  - "score loan applications with an explainable model for a bank"
status: draft
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/user_guide.html
  - https://scikit-learn.org/stable/common_pitfalls.html
---

# Classic ML with scikit-learn

## Summary
scikit-learn is the default Python library for non-deep-learning ML on tabular data: preprocessing, classification, regression, clustering, dimensionality reduction, model selection and metrics, all behind one `fit/predict/transform` API. Start here for any structured-data problem before reaching for gradient boosting libraries or neural nets; it gives a fast, honest baseline.

## Key concepts
- Estimator API: `fit(X, y)`, `predict(X)`, `transform(X)`; every model and transformer follows it.
- `Pipeline` + `ColumnTransformer`: chain imputing, scaling, encoding and the model so all fitting happens only on the training fold (prevents data leakage).
- Cross-validation (`cross_val_score`, `StratifiedKFold`, `TimeSeriesSplit`) estimates generalisation; a single train/test split is noisy.
- Hyperparameter search: `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV`.
- Metrics must match the business goal: ROC-AUC / PR-AUC / F1 for imbalance, MAE/RMSE for regression, never plain accuracy on skewed classes.
- Persist with `joblib.dump(pipeline, path)`; always persist the whole pipeline, not just the model.

## When to use / scenarios
- Signals: rows x columns data, thousands to millions of rows, need for interpretability, small team, tight latency (microseconds to ms).
- Banking: logistic regression scorecard for credit risk; regulators like coefficients.
- Telecom/SaaS: churn prediction, uplift baselines. Retail: customer segmentation with KMeans, basket clustering.
- Healthcare ops: readmission risk from structured fields (validate carefully, see Gotchas).
- NOT for: images/audio/raw text at scale (see [[pytorch-basics]], [[nlp-classic-tasks]]); best-in-class accuracy on large tabular data (see [[gradient-boosting-tabular]]); forecasting with time dependence (see [[time-series-forecasting]]).

## Setup & code
```bash
pip install scikit-learn pandas joblib
```
```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("customers.csv")           # columns incl. target "churned"
X, y = df.drop(columns="churned"), df["churned"]
num = X.select_dtypes("number").columns
cat = X.select_dtypes(exclude="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])
clf = Pipeline([("pre", pre),
                ("model", RandomForestClassifier(n_estimators=300, random_state=0,
                                                 class_weight="balanced"))])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)
print(cross_val_score(clf, X_tr, y_tr, cv=5, scoring="roc_auc").mean())
clf.fit(X_tr, y_tr)
print(clf.score(X_te, y_te))
```

## Choosing / trade-offs
- Linear/logistic models: fast, explainable, strong with many sparse features; weak on interactions.
- Random forest: robust default, little tuning; larger and slower than boosting, rarely the top accuracy.
- Gradient boosting (`HistGradientBoosting*` in sklearn, or XGBoost/LightGBM): usually best accuracy; see [[gradient-boosting-tabular]].
- KMeans needs scaled numeric features and a chosen k; use silhouette score as a sanity check, not truth. DBSCAN/HDBSCAN when clusters are irregular.
- Calibration matters if outputs are used as probabilities: `CalibratedClassifierCV`.

## Gotchas
- Fitting scalers/encoders/imputers on the full data before splitting leaks test information; use a Pipeline.
- Target leakage: features that exist only after the outcome (e.g. "account closed date" for churn).
- Class imbalance: use stratified splits, PR-AUC, `class_weight`, threshold tuning; resampling only inside CV folds.
- Random splits on time- or group-structured data inflate scores; use `TimeSeriesSplit` / `GroupKFold`.
- Pickled models are tied to the scikit-learn version and are unsafe to load from untrusted sources.
- Always keep a dummy baseline (`DummyClassifier`) to see if the model learns anything.

## Related
- [[gradient-boosting-tabular]] - the usual next step for accuracy.
- [[experiment-tracking]] - record CV runs and parameters.
- [[anomaly-detection]] - unsupervised models also live in sklearn.
- [[time-series-forecasting]] - when rows are ordered in time.

## References
- https://scikit-learn.org/stable/user_guide.html
- https://scikit-learn.org/stable/common_pitfalls.html
