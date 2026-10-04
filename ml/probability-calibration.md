---
title: Probability calibration (Platt scaling, isotonic, temperature, reliability diagrams)
category: ml
tags: [calibration, probability-calibration, platt-scaling, isotonic-regression, temperature-scaling, reliability-diagram, brier-score, ece, log-loss, scikit-learn]
use_cases:
  - "make a classifier's 0.8 actually mean right about 80% of the time"
  - "turn model scores into probabilities a pricing, risk or triage rule can use directly"
  - "fix a random forest, SVM or boosted model whose probabilities are too extreme or too timid"
  - "compare models on probability quality, not just ranking (AUC)"
  - "keep probabilities honest after resampling or class weighting for imbalanced data"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/calibration.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.frozen.FrozenEstimator.html
  - https://arxiv.org/abs/1706.04599
  - https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
---

# Probability calibration (Platt scaling, isotonic, temperature, reliability diagrams)

## Summary
A classifier is calibrated when, among all cases it scores 0.8, about 80% are positive. Many models rank well (high AUC) but output badly calibrated scores: naive Bayes and boosted trees push toward 0 and 1, random forests and bagged models pull toward the middle, SVMs output margins rather than probabilities, and class weighting or resampling shifts everything. Calibration fits a small monotone map from raw scores to probabilities on held-out data (Platt/sigmoid, isotonic, or temperature scaling). Do it whenever a downstream decision uses the probability value itself: expected cost, pricing, thresholds shared across models, or risk shown to a person.

## Key concepts
- **Ranking vs calibration.** AUC only checks order; a monotone remap leaves it (almost) unchanged. Calibration fixes the *values*. A model can have AUC 0.98 and still say 0.29 where the true rate is 0.14.
- **Reliability diagram.** Bin predictions, plot mean predicted probability vs observed positive rate. Diagonal = calibrated; S-shape = under-confident (forests); inverse S = over-confident (naive Bayes, deep nets, boosting with many rounds). `calibration_curve` / `CalibrationDisplay` in scikit-learn.
- **Metrics.** *Brier score* (mean squared error of probabilities, proper scoring rule), *log loss* (proper, punishes confident mistakes hard), *ECE* (expected calibration error: bin-weighted gap between confidence and accuracy; easy to read, depends on binning, not a proper score). Report Brier or log loss plus a reliability plot.
- **Platt / sigmoid scaling.** Fit a 1-D logistic regression on the model's score. Two parameters, works with little data (hundreds of points), assumes the distortion is sigmoid-shaped.
- **Isotonic regression.** Fit a non-decreasing step function. Fixes any monotone distortion but needs more data (roughly 1000+ calibration points) and can overfit, producing ties and flat steps.
- **Temperature scaling.** Divide logits by one learned scalar T before softmax. The standard fix for over-confident deep nets, natively multi-class, never changes the argmax. In scikit-learn as `method="temperature"` since 1.8 ([[bayesian-deep-learning-and-uncertainty]] covers the PyTorch version).
- **Held-out data is mandatory.** Calibrating on the training set learns the training-set over-confidence. Use a separate calibration split, or `CalibratedClassifierCV` with `cv=k` (fits k model+calibrator pairs and averages them with `ensemble=True`).
- **Prior shift.** Calibration holds for the class balance it was fitted on. If the deployment base rate differs (resampled training data, seasonality), recalibrate or apply a prior correction.

## When to use / scenarios
- Credit, insurance and fraud scoring where expected loss = probability x exposure ([[finance]]).
- Clinical risk scores and triage, where clinicians read the number as a risk ([[healthcare]]).
- Ad click / conversion bidding, where a bid is value x predicted rate ([[ecommerce-retail]]).
- Combining or comparing probabilities from several models, or applying one threshold to models retrained over time.
- After [[imbalanced-data]] tricks (SMOTE, class weights, undersampling), which deliberately distort probabilities.
- Before [[conformal-prediction-and-uncertainty]] or cost-sensitive thresholds that assume meaningful scores.
- NOT needed when you only rank (top-k leads, search results) or only use a threshold tuned on validation data for that same model.
- Logistic regression and well-regularised gradient boosting with log loss are often already close; check the reliability plot before adding a calibrator.

## Setup & code
```bash
pip install "scikit-learn>=1.6" numpy   # FrozenEstimator needs 1.6+, method="temperature" 1.8+
```
Fit a model, then calibrate it on a separate held-out split with `FrozenEstimator` (the replacement for the removed `cv="prefit"`). Compare Brier, log loss, ECE and AUC on a test split:
```python
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=20000, n_features=20, n_informative=6,
                           n_redundant=10, random_state=0)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.5, random_state=0)
X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)


def ece(y_true, p, n_bins=10):
    """Expected calibration error: bin-weighted |accuracy - confidence|."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return sum(abs(y_true[bins == b].mean() - p[bins == b].mean()) * (bins == b).mean()
               for b in range(n_bins) if (bins == b).any())


for name, base in {"naive bayes": GaussianNB(),
                   "random forest": RandomForestClassifier(n_estimators=200, random_state=0)}.items():
    base.fit(X_tr, y_tr)
    models = {"raw": base}
    for method in ["sigmoid", "isotonic"]:
        # Calibrate the already-fitted model on a held-out set it never saw.
        models[method] = CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(X_cal, y_cal)
    for m_name, m in models.items():
        p = m.predict_proba(X_te)[:, 1]
        print(f"{name:13s} {m_name:8s} brier {brier_score_loss(y_te, p):.4f}  "
              f"logloss {log_loss(y_te, p):.4f}  ECE {ece(y_te, p):.4f}  AUC {roc_auc_score(y_te, p):.4f}")

# Reliability-diagram data: mean predicted vs observed frequency per bin.
frac_pos, mean_pred = calibration_curve(y_te, models["raw"].predict_proba(X_te)[:, 1], n_bins=5)
print("RF raw  pred:", mean_pred.round(2), " observed:", frac_pos.round(2))
```
Output with scikit-learn 1.9.0:
```text
naive bayes   raw      brier 0.0943  logloss 0.3756  ECE 0.0614  AUC 0.9454
naive bayes   sigmoid  brier 0.0931  logloss 0.3149  ECE 0.0197  AUC 0.9454
naive bayes   isotonic brier 0.0889  logloss 0.3310  ECE 0.0153  AUC 0.9448
random forest raw      brier 0.0422  logloss 0.1795  ECE 0.0345  AUC 0.9837
random forest sigmoid  brier 0.0394  logloss 0.1462  ECE 0.0089  AUC 0.9837
random forest isotonic brier 0.0391  logloss 0.1764  ECE 0.0070  AUC 0.9832
RF raw  pred: [0.04 0.29 0.51 0.72 0.95]  observed: [0.01 0.14 0.46 0.76 0.98]
```
Calibration cuts ECE 3-5x while AUC stays put. The raw forest is under-confident at the extremes (says 0.29 where the rate is 0.14). For a plot, use `CalibrationDisplay.from_estimator(model, X_te, y_te, n_bins=10)` with matplotlib installed.

Without a spare split, let cross-validation produce the calibration data: `CalibratedClassifierCV(RandomForestClassifier(), method="isotonic", cv=5).fit(X_train, y_train)`.

## Choosing / trade-offs
- **Sigmoid** with fewer than ~1000 calibration points, or when the reliability plot is a clean S. Safe default.
- **Isotonic** with plenty of calibration data and an irregular distortion; watch for it lowering log loss less than Brier (its flat steps can produce near-0/1 values).
- **Temperature** for multi-class neural nets and anything that outputs logits; one parameter, keeps predictions, cannot fix non-uniform miscalibration across classes.
- **Separate split vs `cv=k`.** A separate split is simple and calibrates the exact model you ship; `cv=k` uses all data but ships k models (or one refit model with `ensemble=False`).
- **Fix at the source when you can.** Train with log loss, regularise, use early stopping, avoid resampling (weight the loss or move the threshold instead), and calibration may become unnecessary.
- **Calibration vs conformal.** Calibration gives a good probability on average; conformal prediction gives sets or intervals with a coverage guarantee. They solve different problems and combine well.

## Gotchas
- Calibrating on training data, or on the same split used for early stopping or threshold tuning: the calibrator learns optimistic scores.
- Assuming calibration holds per subgroup: a model can be calibrated overall but over-confident for one region, product or demographic. Check reliability diagrams per important segment ([[model-evaluation-and-metrics]]).
- Isotonic on small data gives a staircase with a handful of distinct probabilities; ties break downstream ranking.
- ECE depends on the number and kind of bins and can be gamed; report Brier/log loss too.
- Base-rate drift after deployment silently breaks calibration; monitor predicted vs observed rates over time ([[online-learning-and-concept-drift]]).
- `cv="prefit"` was deprecated in scikit-learn 1.6 and is rejected by 1.9; wrap the fitted model in `FrozenEstimator` instead.
- Calibrating a model trained on SMOTE-balanced data with a calibration set that is *also* balanced keeps the wrong prior; the calibration set must have the real class mix.

## Related
- [[model-evaluation-and-metrics]] - Brier, log loss, ROC/PR and threshold choice.
- [[imbalanced-data]] - resampling and class weights that distort probabilities.
- [[bayesian-deep-learning-and-uncertainty]] - temperature scaling and uncertainty for neural nets.
- [[conformal-prediction-and-uncertainty]] - coverage-guaranteed sets and intervals.
- [[svm-knn-naive-bayes]] - models whose raw scores are notoriously miscalibrated.
- [[decision-trees-and-random-forests]] - forests' under-confident probabilities.
- [[gradient-boosting-tabular]] - boosting with log loss, usually close to calibrated.

## References
- scikit-learn, Probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- `CalibratedClassifierCV` (method `temperature` added in 1.8): https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
- `FrozenEstimator` (1.6+): https://scikit-learn.org/stable/modules/generated/sklearn.frozen.FrozenEstimator.html
- Guo et al., On Calibration of Modern Neural Networks (ICML 2017): https://arxiv.org/abs/1706.04599
- Niculescu-Mizil & Caruana, Predicting Good Probabilities With Supervised Learning (ICML 2005): https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf
