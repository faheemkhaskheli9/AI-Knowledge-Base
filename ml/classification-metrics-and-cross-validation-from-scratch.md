---
title: Classification metrics and cross-validation from scratch (confusion matrix, F1, ROC-AUC by ranks, stratified k-fold)
category: ml
tags: [evaluation, metrics, confusion-matrix, precision, recall, f1, roc-auc, mann-whitney, cross-validation, stratified-k-fold, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement precision, recall, F1 and ROC-AUC from scratch for learning or an interview"
  - "understand what ROC-AUC actually measures and why it ignores the threshold"
  - "write a stratified k-fold splitter and report a metric as mean plus spread"
  - "check hand-written metrics against scikit-learn on every fold"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/model_evaluation.html
  - https://scikit-learn.org/stable/modules/cross_validation.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
---

# Classification metrics and cross-validation from scratch (confusion matrix, F1, ROC-AUC by ranks, stratified k-fold)

## Summary
Every binary classification metric is built from four counts (TP, FP, FN, TN) at a chosen threshold, except ROC-AUC, which is a ranking metric: the probability that a random positive scores higher than a random negative. That makes AUC computable from score ranks alone (the Mann-Whitney U statistic), with no curve and no threshold. A stratified k-fold splitter is a dozen lines: deal each class round-robin into folds. The code below matches scikit-learn's `f1_score` and `roc_auc_score` on every fold.

## Key concepts
- **Confusion matrix.** At threshold `t`, `ŷ = score ≥ t`. TP/FP/FN/TN count the four outcomes. Precision = TP/(TP+FP) ("of what I flagged, how much was right"); recall = TP/(TP+FN) ("of what exists, how much I found"); F1 is their harmonic mean.
- **Threshold dependence.** Precision, recall, F1 and accuracy all change with `t`. Lower `t` raises recall and usually lowers precision. 0.5 is a default, not a law ([[model-evaluation-and-metrics]]).
- **ROC-AUC as ranking.** AUC = P(score⁺ > score⁻), with ties counted as ½. Rank all scores (average rank for ties), sum the positives' ranks `R⁺`, then `AUC = (R⁺ − n⁺(n⁺+1)/2) / (n⁺·n⁻)`. One sort, `O(n log n)`.
- **AUC is scale-free.** Any monotone transform of the scores (logits, probabilities, ranks) gives the same AUC. It says nothing about calibration ([[probability-calibration]]).
- **Stratified k-fold.** Shuffle each class's indices, deal them round-robin into `k` folds, so every fold keeps the class ratio. Train on `k−1` folds, score on the held-out one, report mean ± std.
- **Zero-division.** Precision is undefined when nothing is predicted positive. Pick a convention (0 here, like scikit-learn's default with a warning) and be explicit.

## When to use / scenarios
- Learning and interviews: deriving F1 and AUC, and explaining why AUC is threshold-free.
- Custom metrics: once you can write F1 by hand, cost-weighted metrics (fraud: a missed case costs 100×, a false alarm 1×) are a one-line change.
- Small datasets, where one train/test split is noisy and k-fold mean ± std is the honest number ([[model-selection-and-comparison]]).
- Not for: grouped or time-ordered data. Rows from the same patient or later dates must not be split freely; use group or time-series splits ([[data-leakage-and-validation-splits]]).

## Setup & code
`pip install numpy scikit-learn`. Runs in about a second on CPU.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def confusion(y, yhat):
    tp = np.sum((yhat == 1) & (y == 1)); fp = np.sum((yhat == 1) & (y == 0))
    fn = np.sum((yhat == 0) & (y == 1)); tn = np.sum((yhat == 0) & (y == 0))
    return tp, fp, fn, tn


def prf(y, yhat):
    tp, fp, fn, _ = confusion(y, yhat)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def roc_auc(y, score):
    """AUC = P(score of random positive > random negative), ties count 1/2 (Mann-Whitney U)."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):                       # average ranks over tied scores
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(s)); r[order] = ranks
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def stratified_kfold(y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):                  # deal each class round-robin across folds
        idx = rng.permutation(np.flatnonzero(y == c))
        for i, j in enumerate(idx):
            folds[i % k].append(j)
    for f in range(k):
        te = np.array(sorted(folds[f]))
        tr = np.setdiff1d(np.arange(len(y)), te)
        yield tr, te


X, y = load_breast_cancer(return_X_y=True)
y = 1 - y                                   # make "malignant" the positive class
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

aucs, f1s = [], []
for tr, te in stratified_kfold(y, k=5):
    p = model.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    yhat = (p >= 0.5).astype(int)
    a, f = roc_auc(y[te], p), prf(y[te], yhat)[2]
    assert np.isclose(a, roc_auc_score(y[te], p)) and np.isclose(f, f1_score(y[te], yhat))
    aucs.append(a); f1s.append(f)
    print(f"fold: n_test={len(te)} pos_rate={y[te].mean():.3f} AUC={a:.4f} F1={f:.4f}")
print(f"5-fold AUC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}, F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

# Threshold moves precision/recall; AUC does not depend on it.
for t in (0.1, 0.5, 0.9):
    print(f"threshold {t}: P/R/F1 =", np.round(prf(y[te], (p >= t).astype(int)), 3))
print("ties check:", roc_auc(np.array([0, 1, 0, 1]), np.array([0.2, 0.2, 0.1, 0.9])),
      roc_auc_score([0, 1, 0, 1], [0.2, 0.2, 0.1, 0.9]))
```

Output (numpy 2.5, scikit-learn 1.9):
```
fold: n_test=115 pos_rate=0.374 AUC=1.0000 F1=1.0000
fold: n_test=115 pos_rate=0.374 AUC=0.9880 F1=0.9524
fold: n_test=113 pos_rate=0.372 AUC=0.9983 F1=0.9647
fold: n_test=113 pos_rate=0.372 AUC=0.9936 F1=0.9639
fold: n_test=113 pos_rate=0.372 AUC=0.9973 F1=0.9762
5-fold AUC 0.9955 +/- 0.0043, F1 0.9714 +/- 0.0162
threshold 0.1: P/R/F1 = [0.891 0.976 0.932]
threshold 0.5: P/R/F1 = [0.976 0.976 0.976]
threshold 0.9: P/R/F1 = [1.    0.833 0.909]
ties check: 0.875 0.875
```

Every fold keeps the 37% positive rate, and the from-scratch AUC and F1 pass the `assert` against scikit-learn on all five folds. One fold scores a perfect 1.000 and another 0.952 F1: a single split could have reported either, which is why the mean ± spread is the number to quote. On the last fold, moving the threshold from 0.1 to 0.9 trades recall (0.976 → 0.833) for precision (0.891 → 1.0) while AUC stays fixed. The tie check shows the ½ credit for a positive and negative with equal scores.

## Choosing / trade-offs
- **F1 vs AUC.** AUC judges ranking over all thresholds, good for comparing models before a threshold is chosen. F1 (or precision at a fixed recall) judges the deployed decision. Report both: a model can have high AUC and poor F1 at a badly chosen threshold.
- **ROC-AUC vs PR-AUC.** With rare positives (1% fraud), ROC-AUC stays high because true negatives are plentiful; PR-AUC (average precision) shows the precision you will actually get ([[imbalanced-data]]).
- **Micro vs macro averaging (multiclass).** Macro averages per-class F1 equally, so rare classes count as much as common ones; micro pools all counts and tracks accuracy ([[multiclass-classification-strategies]]).
- **k = 5 vs 10 vs repeated.** 5 is the usual default. 10 lowers the bias of each estimate at twice the cost. Repeated k-fold with different seeds shrinks the variance of the mean on small data.

## Gotchas
- Tune the threshold on validation folds, never on the test set; otherwise the reported F1 is optimistic.
- Fit scalers, encoders and feature selection inside each fold (a `Pipeline` does it). Fitting them on all rows first leaks the test fold ([[data-leakage-and-validation-splits]]).
- The std across folds understates true uncertainty because folds share training data. Use it to compare models on the same folds, not as a confidence interval.
- `roc_auc` needs both classes in the fold; a tiny fold with no positives is undefined. Stratification prevents it.
- Sorting scores without averaging tie ranks gives a different AUC on discrete scores (tree leaves, rounded probabilities). Check with a tie case like the one above.
- Which class is "positive" matters for precision/recall/F1 but not for AUC up to `1 − AUC`. Here `y = 1 − y` makes the malignant class positive; scikit-learn's breast-cancer labels code it as 0.

## Related
- [[model-evaluation-and-metrics]] - the scikit-learn version and how to pick a metric for a business goal.
- [[data-leakage-and-validation-splits]] - group and time-series splits when rows are not independent.
- [[imbalanced-data]] - PR-AUC, resampling and thresholds for rare positives.
- [[probability-calibration]] - AUC ignores calibration; this covers it.
- [[logistic-regression-from-scratch]] - the classifier these metrics are usually first applied to.
- [[regression-metrics-and-residual-analysis]] - the regression counterpart.

## References
- scikit-learn, Metrics and scoring: https://scikit-learn.org/stable/modules/model_evaluation.html
- scikit-learn, Cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn `roc_auc_score` API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
