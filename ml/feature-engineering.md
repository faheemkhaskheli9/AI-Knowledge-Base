---
title: Feature engineering and data preprocessing
category: ml
tags: [feature-engineering, preprocessing, encoding, scaling, imputation, pandas, scikit-learn, tabular, eda]
use_cases:
  - "turn a raw customer or transaction table into model-ready features"
  - "handle missing values, categories and skewed numbers before training"
  - "create date, aggregate and ratio features that improve a churn or fraud model"
  - "encode a high-cardinality column like city, product ID or merchant"
  - "make sure training and serving compute features the same way"
status: draft
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/preprocessing.html
  - https://scikit-learn.org/stable/modules/impute.html
  - https://scikit-learn.org/stable/modules/compose.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
---

# Feature engineering and data preprocessing

## Summary
Feature engineering turns raw records into numeric inputs a model can learn from: cleaning, imputing missing values, encoding categories, scaling numbers and creating new features (dates, aggregates, ratios) that expose the signal. On tabular data it usually moves the metric more than switching algorithms. Deep learning learns features from raw images, audio and text, so this file is mainly about tabular and event data.

## Key concepts
- **Explore first (EDA).** Look at types, missing rates, cardinality, distributions, target rate per category and obvious leaks before writing any transform.
- **Missing values.** Impute (median/most-frequent, `SimpleImputer`; model-based, `IterativeImputer`) and often add a "was missing" indicator (`add_indicator=True`) because missingness itself can be predictive. Gradient boosting libraries handle NaN natively.
- **Categorical encoding.** One-hot for low cardinality; ordinal for ordered levels or tree models; target encoding (`TargetEncoder`, cross-fitted) or hashing for high cardinality; native categorical support in LightGBM/CatBoost/`HistGradientBoosting`.
- **Scaling.** `StandardScaler` / `MinMaxScaler` for distance- and gradient-based models (linear, SVM, k-NN, k-means, neural nets). Trees do not need it.
- **Skew and outliers.** Log / Box-Cox / Yeo-Johnson (`PowerTransformer`) for heavy-tailed amounts; clip or use `RobustScaler` for outliers.
- **Created features.** Date parts and "days since" (recency), ratios (debt/income), aggregates per entity over time windows (transactions in last 7 days), counts, text length, geographic distance, interaction terms.
- **Leak-free fitting.** Every transform that learns from data (imputer, scaler, encoder) is fit on the training fold only; put them in a `Pipeline` + `ColumnTransformer`.
- **Feature selection.** Drop constant, duplicate and ID-like columns; use permutation importance or L1 to prune; fewer features means less drift surface.

## When to use / scenarios
- Any tabular project: churn, credit scoring, fraud, insurance claims, lead scoring, demand.
- Event logs (clicks, transactions, sensor readings): aggregate to one row per entity per prediction time with rolling windows.
- Retail/e-commerce: recency-frequency-monetary (RFM) features for customers; price relative to category average for products.
- Manufacturing/IoT: rolling mean, std, min/max and slope of sensor signals per machine.
- NOT needed much for: images, audio and long text with deep models (they learn representations; see [[pytorch-basics]], [[embeddings]]). Text in a tabular model can become an embedding column.

## Setup & code
```bash
pip install scikit-learn pandas
```
```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, TargetEncoder

rng = np.random.default_rng(0)
n = 3000
df = pd.DataFrame({
    "signup": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 600, n), "D"),
    "income": rng.lognormal(10, 1, n),
    "debt": rng.lognormal(9, 1, n),
    "plan": rng.choice(["basic", "pro", "team"], n),
    "city": rng.choice([f"city_{i}" for i in range(300)], n),   # high cardinality
})
df.loc[rng.random(n) < 0.1, "income"] = np.nan
y = (df["debt"] / df["income"].fillna(df["income"].median()) > 0.4).astype(int)

# Hand-made features: computed the same way at training and serving time.
def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    as_of = pd.Timestamp("2026-10-01")
    d["tenure_days"] = (as_of - d["signup"]).dt.days
    d["signup_month"] = d["signup"].dt.month
    d["debt_to_income"] = d["debt"] / d["income"]
    return d.drop(columns="signup")

num = ["income", "debt", "tenure_days", "signup_month", "debt_to_income"]
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                      ("log", FunctionTransformer(np.log1p))]), num),
    ("plan", OneHotEncoder(handle_unknown="ignore"), ["plan"]),
    ("city", TargetEncoder(target_type="binary"), ["city"]),  # cross-fitted internally
])
model = Pipeline([
    ("feat", FunctionTransformer(add_features)),
    ("pre", pre),
    ("clf", HistGradientBoostingClassifier(random_state=0)),
])
print(cross_val_score(model, df, y, cv=5, scoring="roc_auc").mean())
```

## Choosing / trade-offs
- **One-hot vs target encoding.** One-hot is safe and interpretable but explodes with hundreds of levels; target encoding is compact but leaks unless cross-fitted (sklearn's `TargetEncoder` does this in `fit_transform`).
- **Hand-made vs learned features.** Hand-made features encode domain knowledge cheaply and explain well; learned embeddings (entity embeddings, text embeddings) win when there is lots of data and many raw signals.
- **Rich features vs serving cost.** Aggregates over long windows need a feature pipeline or feature store at serving time; a slightly weaker model with features available in the request can be the better product.
- **Linear models** need more engineering (interactions, binning, scaling); **boosted trees** find interactions themselves and need little scaling, so engineering effort goes into aggregates and domain ratios.

## Gotchas
- Computing aggregates over the whole dataset (including the future) leaks: an entity's "average spend" must use only events before the prediction time (point-in-time correctness).
- Fitting imputers/scalers/encoders before the train/test split leaks; keep them inside the pipeline.
- Train/serve skew: features re-implemented separately for production (SQL vs pandas) silently drift apart. Ship the same pipeline object or one shared feature definition.
- Unseen categories at serving time crash `OneHotEncoder` unless `handle_unknown="ignore"`.
- IDs, row numbers and timestamps used raw act as leaky or meaningless features.
- Dividing by a value that can be zero or missing (ratios) yields inf/NaN; guard it.
- Too many weak features increase variance and maintenance; prune with permutation importance on validation data.

## Related
- [[ml-fundamentals]] - leakage and splits explained.
- [[classic-ml-scikit-learn]] - pipelines and models that consume these features.
- [[gradient-boosting-tabular]] - native categorical and missing-value handling.
- [[model-evaluation-and-metrics]] - measure whether a feature actually helps.
- [[time-series-forecasting]] - lag and rolling-window features for forecasting.

## References
- Preprocessing data: https://scikit-learn.org/stable/modules/preprocessing.html
- Imputation: https://scikit-learn.org/stable/modules/impute.html
- Pipelines and composite estimators: https://scikit-learn.org/stable/modules/compose.html
- TargetEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
