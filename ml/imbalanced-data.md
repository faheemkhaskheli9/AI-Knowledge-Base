---
title: Imbalanced classification (rare classes)
category: ml
tags: [imbalanced-data, class-imbalance, class-weight, smote, resampling, threshold-tuning, imbalanced-learn, scikit-learn]
use_cases:
  - "train a fraud, default or rare-disease classifier where positives are under 1-5% of rows"
  - "decide between class weights, resampling and threshold tuning for a skewed dataset"
  - "pick the decision threshold that matches a fixed review capacity or recall target"
  - "stop a model from predicting the majority class for everything"
status: draft
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/classification_threshold.html
  - https://imbalanced-learn.org/stable/
  - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html
---

# Imbalanced classification (rare classes)

## Summary
In fraud, defects, churn, default and rare-disease data the class you care about may be 0.1-5% of rows. Most models still rank cases well; what goes wrong is the **evaluation** (accuracy looks great when the model predicts "no" for everything) and the **decision threshold** (0.5 is almost never right). The usual fix, in order: measure with PR-AUC and recall/precision at a threshold, tune the threshold to the business cost, then try class weights; resampling such as SMOTE comes last and helps less often than its popularity suggests.

## Key concepts
- **Imbalance ratio.** Minority share of rows. Below ~10% needs care; below ~1% the number of positives (not the ratio) usually limits what can be learned.
- **Metrics.** PR-AUC (average precision) to compare models; precision and recall at the chosen threshold to decide; ROC-AUC stays optimistic under heavy imbalance. See [[model-evaluation-and-metrics]].
- **Threshold moving.** Scores are a ranking; choose the cut-off from costs (a missed fraud vs a false alarm) or capacity ("analysts can review 200 cases a day"). `TunedThresholdClassifierCV` picks it by cross-validation.
- **Class weights.** `class_weight="balanced"` (or `scale_pos_weight` in XGBoost/LightGBM) up-weights minority errors in the loss. Same effect as oversampling, no extra rows, but it distorts predicted probabilities.
- **Resampling.** Random undersampling of the majority (fast, loses data), random oversampling of the minority, or **SMOTE** (synthesises minority points between neighbours). Lives in `imbalanced-learn`.
- **Stratification.** Use `stratify=y` in splits and `StratifiedKFold` so every fold contains positives.
- **Calibration.** After weighting or resampling, probabilities are inflated; recalibrate (`CalibratedClassifierCV`) if anyone reads them as real risks.

## When to use / scenarios
- Card and payment fraud, insurance claims fraud, AML alerts: tiny positive rate, fixed analyst capacity, so threshold by capacity.
- Manufacturing defect detection and predictive maintenance: rare failures, high cost of misses, so threshold for high recall.
- Medical screening for rare conditions: recall target set clinically, then maximise precision.
- Churn and conversion prediction: moderate imbalance (5-20%); class weights plus a tuned threshold are usually enough.
- When positives are extremely rare or unlabelled, consider [[anomaly-detection]] instead of supervised classification.

## Setup & code
```bash
pip install scikit-learn imbalanced-learn
```
```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_pipeline
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import TunedThresholdClassifierCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1% positives, like a fraud dataset.
X, y = make_classification(n_samples=20000, n_features=20, n_informative=6,
                           weights=[0.99], flip_y=0.005, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)

def lr(**kw):
    return LogisticRegression(max_iter=1000, **kw)

models = {
    "plain": make_pipeline(StandardScaler(), lr()),
    "class_weight": make_pipeline(StandardScaler(), lr(class_weight="balanced")),
    # Resampling inside the pipeline: applied to training folds only.
    "SMOTE": imb_pipeline(StandardScaler(), SMOTE(random_state=0), lr()),
    # Tune the cut-off by CV; swap scoring for a cost function or F-beta.
    "tuned threshold": TunedThresholdClassifierCV(
        make_pipeline(StandardScaler(), lr()),
        scoring="balanced_accuracy", cv=5),
}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    pred, score = m.predict(X_te), m.predict_proba(X_te)[:, 1]
    print(f"{name:15s} PR-AUC {average_precision_score(y_te, score):.3f}  "
          f"recall {recall_score(y_te, pred):.2f}  "
          f"precision {precision_score(y_te, pred, zero_division=0):.2f}")
```
Output on this run: the plain model has PR-AUC 0.29 but catches only 8% of positives at the default 0.5 threshold. Class weights and SMOTE raise recall to about 0.8 but lower PR-AUC to about 0.22 (the ranking got worse); the tuned threshold reaches recall 0.72 and keeps the plain model's PR-AUC, because it only moves the cut-off. That is the core lesson: most imbalance techniques just move the operating point, which a tuned threshold does directly without damaging the model.

## Choosing / trade-offs
- **Start with threshold tuning.** It keeps calibrated probabilities and costs nothing at training time. Pick the scoring that matches the decision (cost matrix, recall at capacity, F-beta).
- **Class weights vs SMOTE.** Class weights are simpler, deterministic and work with any loss-based model; SMOTE can help weak or linear models on low-dimensional numeric data, rarely helps gradient boosting, and is meaningless for text, images or categorical-heavy data.
- **Undersampling** is useful when the data is too large to train on comfortably: keep all positives and a sample of negatives, then correct probabilities for the sampling rate.
- **More positives beat any trick.** Spending effort on labelling more minority cases ([[data-labeling-and-synthetic-data]]) usually helps more than resampling.

## Gotchas
- Resampling before the train/test split (or outside the CV pipeline) leaks synthetic copies of test points into training and inflates scores. Use `imblearn.pipeline`, never `sklearn.pipeline`, for samplers.
- Never resample the validation or test set; it must reflect real-world prevalence.
- Weighted or resampled models output inflated probabilities; a "70% fraud risk" may really be 5%. Recalibrate before displaying or thresholding on probability.
- Precision depends on prevalence: a model validated at 1% fraud will show different precision when live fraud rate changes. Monitor it ([[mlops-lifecycle]]).
- Too few positives in a fold makes CV metrics noisy; use repeated stratified CV and report the spread.

## Related
- [[model-evaluation-and-metrics]] - PR-AUC, threshold choice and calibration in detail.
- [[anomaly-detection]] - when labels for the rare class are missing.
- [[gradient-boosting-tabular]] - `scale_pos_weight` and the usual strong model for these tasks.
- [[ml-fundamentals]] - splits, baselines and leakage.
- [[finance]] - fraud and credit scenarios.

## References
- scikit-learn decision-threshold tuning: https://scikit-learn.org/stable/modules/classification_threshold.html
- imbalanced-learn documentation: https://imbalanced-learn.org/stable/
- TunedThresholdClassifierCV API: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html
