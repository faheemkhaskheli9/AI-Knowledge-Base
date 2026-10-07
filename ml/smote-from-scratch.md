---
title: SMOTE oversampling from scratch (vs random oversampling, class weights and threshold tuning)
category: ml
tags: [smote, oversampling, imbalanced-data, class-weights, threshold, resampling, classification, numpy, from-scratch]
use_cases:
  - "handle a rare positive class (fraud, defects, churn, disease) in a classifier"
  - "decide between SMOTE, random oversampling, class weights and threshold tuning"
  - "understand how SMOTE generates synthetic minority samples"
  - "explain why resampling must happen inside cross-validation folds"
status: stable
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1106.1813
  - https://imbalanced-learn.org/stable/over_sampling.html
  - https://scikit-learn.org/stable/modules/classification_threshold.html
---

# SMOTE oversampling from scratch (vs random oversampling, class weights and threshold tuning)

## Summary
SMOTE (Synthetic Minority Over-sampling Technique) balances a dataset by creating new minority-class points on the line segments between each minority point and one of its `k` nearest minority neighbours. It is the best-known fix for imbalanced classification. This file implements SMOTE in a few lines of NumPy and compares it on a 5%-positive problem with three alternatives: random duplication, class weights and simply moving the decision threshold. For a logistic regression, all three rebalancing methods produce almost the same model: recall rises from `0.67` to `0.98` and precision falls from `0.92` to about `0.35`. Rebalancing mostly moves the threshold. Whether that trade is worth it depends on the cost of each error, not on the method.

## Key concepts
- **Generation rule.** For a minority point `x_i`, pick one of its `k` nearest minority neighbours `x_j` and a random `λ ∈ [0, 1]`. The new point is `x_i + λ (x_j − x_i)`. Repeat until the desired count is reached.
- **Interpolation, not new information.** Synthetic points lie inside the region the minority class already covers. They smooth the class region, but they cannot reveal minority cases that look different from the ones you have.
- **Rebalancing ≈ reweighting ≈ threshold shift.** For a well-specified probabilistic model, training on balanced data shifts the predicted probabilities toward the minority class, which is close to lowering the threshold on the original model. That is why the three rebalancing rows below agree.
- **Precision-recall trade-off.** Any of these methods trades precision for recall. The right operating point comes from the business cost of false positives vs false negatives.
- **Resample only the training fold.** The validation and test sets must keep the real class ratio, or the metrics are meaningless.

## When to use / scenarios
- Fraud, defect, churn or rare-disease detection where positives are 0.1-10% of the data and the model's decision boundary ignores them by default.
- Models that have no class-weight option or that underfit the minority region (k-NN, some tree ensembles on very small minority classes).
- Not as a first resort: try class weights and threshold tuning first. They are simpler, add no synthetic data, and give the same effect for linear models and gradient boosting.
- Not for: high-dimensional sparse data such as text (interpolating TF-IDF vectors is meaningless), images (use data augmentation, see [[data-augmentation]]), or categorical features without SMOTE-NC.
- Not when the real problem is the metric. Accuracy on a 95/5 split is useless with or without SMOTE. See [[imbalanced-data]].

## Setup & code
NumPy only. Runs in a few seconds.

```python
import numpy as np

rng = np.random.default_rng(0)


def smote(X_min, n_new, k=5):
    """Synthetic minority points on segments between a point and one of its k nearest minority neighbours."""
    d = ((X_min[:, None] - X_min[None]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    i = rng.integers(0, len(X_min), n_new)
    j = nn[i, rng.integers(0, k, n_new)]
    return X_min[i] + rng.random((n_new, 1)) * (X_min[j] - X_min[i])


def make_data(n_maj, n_min):
    X = np.vstack([rng.normal(0, 1, (n_maj, 2)), rng.normal(1.8, 0.7, (n_min, 2))])
    return X, np.r_[np.zeros(n_maj), np.ones(n_min)]


def fit_logreg(X, y, w=None, lr=0.1, epochs=2000):
    w = np.ones(len(y)) if w is None else w
    Xb = np.c_[X, np.ones(len(X))]
    theta = np.zeros(3)
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-Xb @ theta))
        theta -= lr * Xb.T @ (w * (p - y)) / w.sum()
    return lambda Z: 1 / (1 + np.exp(-np.c_[Z, np.ones(len(Z))] @ theta))


def f1(pred, y):
    tp = np.sum(pred & (y == 1))
    return 2 * tp / (pred.sum() + y.sum()), tp / y.sum(), tp / max(pred.sum(), 1)


X_tr, y_tr = make_data(950, 50)
X_va, y_va = make_data(950, 50)            # for picking a threshold
X_te, y_te = make_data(1900, 100)
X_min, n_new = X_tr[y_tr == 1], 900        # 50 -> 950 minority points

dup = X_min[rng.integers(0, len(X_min), n_new)]
syn = smote(X_min, n_new)
models = {
    "imbalanced": fit_logreg(X_tr, y_tr),
    "random oversample": fit_logreg(np.vstack([X_tr, dup]), np.r_[y_tr, np.ones(n_new)]),
    "smote": fit_logreg(np.vstack([X_tr, syn]), np.r_[y_tr, np.ones(n_new)]),
    "class weights": fit_logreg(X_tr, y_tr, w=np.where(y_tr == 1, 19.0, 1.0)),
}
for name, m in models.items():
    f, r, p = f1(m(X_te) > 0.5, y_te)
    print(f"{name:18s} recall {r:.2f}  precision {p:.2f}  f1 {f:.2f}")

base = models["imbalanced"]
t = max(np.linspace(0.05, 0.95, 91), key=lambda t: f1(base(X_va) > t, y_va)[0])
f, r, p = f1(base(X_te) > t, y_te)
print(f"{'imbalanced, t=%.2f' % t:18s} recall {r:.2f}  precision {p:.2f}  f1 {f:.2f}")
```

Output (Python 3.14, NumPy 2.5):
```
imbalanced         recall 0.67  precision 0.92  f1 0.77
random oversample  recall 0.98  precision 0.35  f1 0.51
smote              recall 0.98  precision 0.36  f1 0.53
class weights      recall 0.98  precision 0.35  f1 0.52
imbalanced, t=0.29 recall 0.84  precision 0.65  f1 0.73
```

SMOTE, duplication and a 19× class weight give the same recall and nearly the same precision: for logistic regression they are three ways of moving the boundary toward the majority class. At the default 0.5 threshold they catch 98% of positives at the price of about two false alarms per true positive. Choosing the threshold on a separate validation set gives an intermediate trade-off without any resampling. Here it scored slightly lower F1 on the test set than the untouched model, because 50 validation positives make the chosen threshold noisy. The point is not which row has the best F1, but that each row is a different point on the same precision-recall curve.

## Choosing / trade-offs
- **Start with the metric and the threshold.** Pick a metric that reflects costs (precision at a fixed recall, cost-weighted error, PR-AUC), train normally, then choose the threshold on validation data. scikit-learn's `TunedThresholdClassifierCV` does this with cross-validation.
- **Class weights next.** `class_weight="balanced"` in scikit-learn, `scale_pos_weight` in XGBoost and LightGBM. No change to the data, no extra training time.
- **SMOTE when the model benefits from a denser minority region.** Flexible models (k-NN, deep trees) can overfit duplicated points, and interpolation gives them a smoother region instead. Gains over class weights are usually small. Measure them.
- **Variants.** Borderline-SMOTE and ADASYN generate more points near the class boundary. SMOTE-NC handles mixed categorical and numeric features. SMOTE + Tomek links or ENN also cleans overlapping majority points.
- **Undersampling.** Dropping majority points is fast and works when the majority class is huge and redundant. It throws away data, so use it with an ensemble (balanced bagging, `BalancedRandomForestClassifier`).
- **Probabilities.** Any rebalancing biases the predicted probabilities. If downstream code needs calibrated probabilities (expected loss, pricing), recalibrate afterwards, see [[platt-and-isotonic-calibration-from-scratch]].
- **Library.** `imbalanced-learn` (`SMOTE`, `ADASYN`, `SMOTENC`) with its `Pipeline`, which applies samplers to training folds only.

## Gotchas
- Applying SMOTE before the train/test split, or before cross-validation, leaks synthetic copies of test points into training and inflates the scores. Use `imblearn.pipeline.Pipeline` so resampling happens inside each fold.
- Scale features before SMOTE. The nearest-neighbour search uses distances, so an unscaled large-range feature dominates which neighbours are chosen.
- Interpolating integer, one-hot or bounded features gives impossible values (2.4 children, half a category). Use SMOTE-NC or round and clip.
- Minority outliers and mislabelled points get copied into many synthetic points. Clean labels first, see [[imbalanced-data]].
- Never evaluate on resampled data. The test set must have the real class ratio.
- With very few minority points (fewer than `k + 1`), SMOTE cannot find `k` neighbours. Lower `k` or collect more data.

## Related
- [[imbalanced-data]] - metrics, sampling and cost-sensitive learning for skewed classes.
- [[knn-from-scratch]] - the neighbour search SMOTE uses.
- [[logistic-regression-from-scratch]] - the classifier used in the comparison.
- [[classification-metrics-and-cross-validation-from-scratch]] - precision, recall, F1 and fold handling.
- [[platt-and-isotonic-calibration-from-scratch]] - fixing probabilities after rebalancing.
- [[data-leakage-and-validation-splits]] - why resampling belongs inside the folds.

## References
- Chawla et al. (2002), "SMOTE: Synthetic Minority Over-sampling Technique", JAIR: https://arxiv.org/abs/1106.1813
- imbalanced-learn over-sampling guide: https://imbalanced-learn.org/stable/over_sampling.html
- scikit-learn, tuning the decision threshold: https://scikit-learn.org/stable/modules/classification_threshold.html
