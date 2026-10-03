---
title: Model evaluation and metrics
category: ml
tags: [evaluation, metrics, cross-validation, precision, recall, roc-auc, pr-auc, calibration, threshold, scikit-learn]
use_cases:
  - "pick the right metric for an imbalanced fraud or disease classifier"
  - "choose a decision threshold that balances false alarms against missed cases"
  - "compare two models fairly with cross-validation instead of one lucky split"
  - "report regression error in business units (e.g. rupees or units of stock)"
  - "check whether predicted probabilities can be trusted as real probabilities"
status: draft
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/model_evaluation.html
  - https://scikit-learn.org/stable/modules/cross_validation.html
  - https://scikit-learn.org/stable/modules/calibration.html
  - https://scikit-learn.org/stable/modules/classification_threshold.html
---

# Model evaluation and metrics

## Summary
Evaluation answers "how good will this model be on data it has not seen, measured in a way the business cares about". It has two parts: a validation scheme that gives an honest estimate (hold-out, cross-validation, time-based splits) and a metric that matches the cost of each kind of error. Picking the wrong one of either is the most common reason an ML project ships a model that looks good and is not. For LLM output quality see [[llm-evaluation]] instead.

## Key concepts
- **Confusion matrix** (binary): TP, FP, TN, FN. Every classification metric is built from it.
- **Precision** = TP / (TP + FP): of the cases flagged, how many were right. Matters when false alarms are costly (manual review queues, spam folders).
- **Recall** (sensitivity) = TP / (TP + FN): of the real positives, how many were caught. Matters when misses are costly (fraud, cancer screening).
- **F1**: harmonic mean of precision and recall; F-beta weights recall beta times as much.
- **ROC-AUC**: ranking quality across all thresholds; threshold-free but optimistic under heavy imbalance.
- **PR-AUC / average precision**: precision-recall area; the better summary when positives are rare.
- **Log loss / Brier score**: quality of the probabilities themselves, not just their ranking.
- **Regression**: MAE (median-ish, robust, same units as y), RMSE (penalises large errors), MAPE (relative, breaks near zero), R² (variance explained, not an error in business units).
- **Threshold.** Classifiers output scores; the 0.5 cut-off is arbitrary. Choose it from the cost of FP vs FN on validation data.
- **Calibration.** A calibrated model's "0.8" is right 80% of the time. Trees, boosting and SVMs often are not; fix with `CalibratedClassifierCV`.
- **Cross-validation (k-fold).** Train k models on k-1 folds, score on the held-out fold; report mean and spread. Stratify for classification, group or time-order when rows are not independent.

## When to use / scenarios
- Imbalanced detection (fraud, defects, rare disease): PR-AUC to compare models, then precision at a fixed recall (or recall at a fixed review capacity) to pick the threshold.
- Medical screening: high recall first; report sensitivity and specificity with confidence intervals.
- Lead scoring / ranking: ROC-AUC, precision@k (top k leads a sales team can call).
- Demand / price regression: MAE or weighted MAPE in units the planner understands; RMSE when big misses are disproportionately costly.
- Credit or insurance pricing: calibration (Brier, reliability plot) because the probability itself drives a price or a limit.
- NOT this file: generative/LLM outputs ([[llm-evaluation]]), forecasting backtests ([[time-series-forecasting]]), ranking recommendations at scale ([[recommender-systems]]).

## Setup & code
```bash
pip install scikit-learn
```
```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

# 2% positives, like fraud.
X, y = make_classification(n_samples=20000, weights=[0.98], random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)

model = LogisticRegression(max_iter=1000)
cv = cross_validate(model, X_tr, y_tr, scoring=["roc_auc", "average_precision"],
                    cv=StratifiedKFold(5, shuffle=True, random_state=0))
print({k: f"{v.mean():.3f} ± {v.std():.3f}" for k, v in cv.items() if k.startswith("test_")})

model.fit(X_tr, y_tr)
p = model.predict_proba(X_te)[:, 1]
print("ROC-AUC", roc_auc_score(y_te, p), "PR-AUC", average_precision_score(y_te, p))

# Pick the threshold that keeps recall >= 0.80 with the best precision.
# (Do this on a validation split in practice; the test set is used here for brevity.)
prec, rec, thr = precision_recall_curve(y_te, p)
ok = rec[:-1] >= 0.80
t = thr[ok][prec[:-1][ok].argmax()]
print(f"threshold={t:.3f}")
print(classification_report(y_te, (p >= t).astype(int), digits=3))
```
scikit-learn also ships `TunedThresholdClassifierCV` to tune the threshold inside cross-validation.

## Choosing / trade-offs
- **Hold-out vs k-fold.** Large data (hundreds of thousands of rows): one validation split is stable and cheap. Small data: k-fold (5 or 10), optionally repeated.
- **Nested CV** when you both tune hyperparameters and want an unbiased score on small data; otherwise tune with CV and keep a final untouched test set.
- **ROC-AUC vs PR-AUC.** ROC-AUC for balanced classes or when both classes matter equally; PR-AUC when the positive class is rare and is what you act on.
- **Single number vs cost model.** If error costs are known (a missed fraud costs X, a review costs Y), optimise expected cost directly instead of F1.
- **Micro vs macro averaging** (multi-class): micro favours frequent classes; macro treats every class equally and exposes weak rare classes.

## Gotchas
- Accuracy on imbalanced data: predicting "never fraud" scores 98% above.
- Tuning the threshold or hyperparameters on the test set leaks it; keep test untouched until the end.
- Random k-fold on time series or on repeated entities (same customer, same patient, same device) inflates scores; use `TimeSeriesSplit` or `GroupKFold`.
- Resampling (SMOTE, undersampling) applied before splitting leaks synthetic copies into validation; do it inside the CV pipeline (imbalanced-learn `Pipeline`).
- A metric difference smaller than the CV standard deviation is noise, not a better model.
- Offline metric ≠ business impact; confirm with an A/B test or shadow deployment where possible.
- `score()` defaults differ by estimator (accuracy for classifiers, R² for regressors); pass `scoring=` explicitly.

## Related
- [[ml-fundamentals]] - why held-out evaluation and splits matter at all.
- [[classic-ml-scikit-learn]] - pipelines that keep preprocessing inside CV folds.
- [[gradient-boosting-tabular]] - early stopping uses a validation set too.
- [[experiment-tracking]] - log every metric and split for comparison.
- [[llm-evaluation]] - evaluation for LLM applications.

## References
- Metrics and scoring: https://scikit-learn.org/stable/modules/model_evaluation.html
- Cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- Probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- Tuning the decision threshold: https://scikit-learn.org/stable/modules/classification_threshold.html
