---
title: Data leakage and validation splits (stratified, group, time-series CV)
category: ml
tags: [data-leakage, cross-validation, train-test-split, stratified-kfold, group-kfold, time-series-split, pipeline, target-leakage, nested-cv, scikit-learn]
use_cases:
  - "my model scores far higher offline than it does in production"
  - "split data where one customer, patient or device has many rows"
  - "validate a forecasting or churn model without using the future to predict the past"
  - "scale, impute or select features without leaking test data into training"
  - "tune hyperparameters and still report an honest test score"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/common_pitfalls.html
  - https://scikit-learn.org/stable/modules/cross_validation.html
  - https://scikit-learn.org/stable/modules/compose.html
  - https://www.kaggle.com/code/alexisbcook/data-leakage
---

# Data leakage and validation splits (stratified, group, time-series CV)

## Summary
Data leakage is any information in training or validation that would not be available at prediction time. It makes offline scores look better than production. Most leakage comes from two places: the **split** (the same customer, patient or time period on both sides) and **preprocessing** (scaling, imputing, encoding or selecting features on the full dataset before splitting). The fix is to split the way the model will be used, and to fit every data-dependent step inside the training fold, usually with a scikit-learn `Pipeline`.

## Key concepts
- **Three sets.** Train fits parameters, validation (or CV) picks hyperparameters and features, test is touched once for the final number. Every look at the test set that changes a decision turns it into a validation set.
- **Split matches deployment.** Ask "what will the model see when it predicts?" and make validation look like that:
  - **KFold / StratifiedKFold**: rows are independent. Stratified keeps class ratios per fold (classification default).
  - **GroupKFold / StratifiedGroupKFold**: many rows per entity (user, patient, device, document). All rows of a group go to the same side, so the model is scored on *unseen* entities.
  - **TimeSeriesSplit**: training always comes before validation. Add `gap` to skip the rows whose labels are not yet known at prediction time.
- **Target leakage.** A feature that is a consequence of the label or only filled in after it: `refund_issued` when predicting fraud, `discharge_code` when predicting readmission, aggregates computed over the whole period including the future.
- **Train-test contamination.** Preprocessing that learns from all rows (mean for scaling, vocabulary, target encoding, SMOTE, feature selection) and is then evaluated on rows it has already seen.
- **Duplicates and near-duplicates.** The same image resized, the same text with a typo, or repeated measurements land on both sides and inflate scores.
- **Nested CV.** An inner CV loop tunes hyperparameters; an outer loop scores the whole tuning procedure. Needed when the dataset is too small to spare a test set.

## When to use / scenarios
- Healthcare: several scans or visits per patient -> group by patient, never by scan.
- Retail and finance forecasting, churn, credit default: time-ordered split with a gap equal to the label horizon.
- Recommenders and per-user models: group by user to measure cold-start, time split to measure ongoing use.
- Fraud and anomaly: time split plus stratification, since positives are rare and fraud patterns drift.
- Kaggle-style random splits are fine only when rows truly are independent and identically distributed.
- NOT needed: a plain random split is already correct for i.i.d. data with one row per entity; do not invent groups.

## Setup & code
```bash
pip install "scikit-learn>=1.4" numpy
```
Leakage from feature selection outside the CV loop, and the `Pipeline` fix. The data is pure noise, so honest accuracy is ~50%:
```python
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 5000))
y = rng.integers(0, 2, 200)

# WRONG: selection sees every label, including the validation folds.
X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
print("leaky :", cross_val_score(LogisticRegression(), X_sel, y, cv=5).mean())   # ~0.8

# RIGHT: every data-dependent step lives inside the pipeline and refits per fold.
pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=20), LogisticRegression())
print("honest:", cross_val_score(pipe, X, y, cv=5).mean())                        # ~0.5
```
Group and time-ordered splits:
```python
# 50 patients with 4 rows each: score on patients the model has never seen.
groups = np.repeat(np.arange(50), 4)
print("group :", cross_val_score(pipe, X, y, cv=GroupKFold(n_splits=5), groups=groups).mean())

# Rows sorted by time; gap=5 drops rows whose labels would not be known yet.
for tr, va in TimeSeriesSplit(n_splits=3, gap=5).split(X):
    print(f"train 0..{tr[-1]}  ->  validate {va[0]}..{va[-1]}")
```
Nested CV for an honest score of a tuned model:
```python
from sklearn.model_selection import GridSearchCV
inner = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1]}, cv=3)
print("nested:", cross_val_score(inner, X, y, cv=5).mean())
```

## Choosing / trade-offs
- **Random vs group vs time.** Use the strictest split that matches deployment. It gives lower but trustworthy numbers; a random split on grouped or temporal data gives higher numbers you cannot ship.
- **Hold-out vs k-fold.** A single hold-out is cheap and fine for large data (100k+ rows). k-fold (5 or 10) gives a variance estimate and uses small data better, at k times the training cost.
- **Nested CV vs a held-out test set.** Nested CV costs inner x outer fits; a held-out test set costs data. Pick by which one you can afford.
- **Time gap size.** Set it to the label horizon (predicting 30-day churn -> 30 days). Too small leaks; too large wastes recent data.

## Gotchas
- `fit_transform` on the full dataset before `train_test_split` is the most common leak. Fit on train, `transform` on validation and test, or use a `Pipeline`.
- Oversampling (SMOTE) before splitting puts synthetic copies of validation points in training. Use `imblearn.pipeline.Pipeline` so it runs only on training folds ([[imbalanced-data]]).
- Target encoding must be computed out-of-fold; scikit-learn's `TargetEncoder` does this inside `fit_transform` ([[feature-engineering]]).
- `cross_val_score` on a `GridSearchCV` that was already fitted on all data is not nested CV; pass the unfitted search object.
- Shuffling a time series before `KFold` (`shuffle=True`) silently trains on the future.
- A feature that looks "too good" in importance plots is usually target leakage; check when it is recorded relative to the label ([[model-interpretability]]).
- Deduplicate (exact hashes, perceptual hashes for images, MinHash for text) before splitting, not after.
- Early stopping on the test set is tuning on the test set; use a validation split.

## Related
- [[ml-fundamentals]] - why train/validation/test exist at all.
- [[model-evaluation-and-metrics]] - metrics and threshold choice once the split is right.
- [[time-series-forecasting]] - backtesting and rolling-origin evaluation.
- [[feature-engineering]] - preprocessing steps that must sit inside the pipeline.
- [[hyperparameter-tuning]] - search that needs its own validation loop.
- [[imbalanced-data]] - resampling without leaking.

## References
- scikit-learn, Common pitfalls (inconsistent preprocessing, data leakage): https://scikit-learn.org/stable/common_pitfalls.html
- scikit-learn, Cross-validation: evaluating estimator performance: https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn, Pipelines and composite estimators: https://scikit-learn.org/stable/modules/compose.html
- Kaggle Learn, Data Leakage: https://www.kaggle.com/code/alexisbcook/data-leakage
