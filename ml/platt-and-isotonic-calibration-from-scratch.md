---
title: Platt scaling and isotonic regression calibration from scratch (PAV, ECE, Brier)
category: ml
tags: [probability-calibration, platt-scaling, isotonic-regression, pool-adjacent-violators, ece, brier-score, log-loss, naive-bayes, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement Platt scaling (Newton's method) and isotonic regression (pool-adjacent-violators) in NumPy"
  - "fix an overconfident classifier's probabilities and measure ECE, Brier score and log loss"
  - "match scikit-learn's IsotonicRegression exactly"
  - "choose between Platt scaling, isotonic regression and temperature scaling"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/calibration.html
  - https://doi.org/10.1145/1102351.1102430
  - https://doi.org/10.1145/775047.775151
  - https://arxiv.org/abs/1706.04599
---

# Platt scaling and isotonic regression calibration from scratch (PAV, ECE, Brier)

## Summary
A classifier is calibrated when its predicted probabilities match observed frequencies: of all the cases it scores 0.8, about 80% are positive. Many models rank well but are badly calibrated. Naive Bayes double-counts correlated features and pushes scores to 0 or 1; boosted trees and SVMs squash them toward the middle. Calibration fits a monotone map from the model's score to a probability on held-out data, without changing the ranking. Below, Gaussian naive Bayes is trained on data with 10 redundant features, and 57% of its test predictions are below 0.01 or above 0.99. Its expected calibration error (ECE) is 0.102. Platt scaling (a 2-parameter logistic fit) brings ECE to 0.017 and log loss from 0.577 to 0.386. Isotonic regression (a non-parametric monotone step function, fitted with pool-adjacent-violators) reaches ECE 0.011 and matches scikit-learn to 1e-16. Accuracy stays at about 0.836, because calibration changes probabilities, not rankings.

## Key concepts
- **Reliability.** Bin predictions by score and compare each bin's mean score with its positive rate. ECE is the size-weighted average gap, `Σ_b (n_b/n)·|mean(p_b) − mean(y_b)|`. It depends on the binning, so report it next to Brier score and log loss, which are proper scoring rules.
- **Platt scaling.** Fit `p = σ(a·f + b)` on a calibration set, where f is the model's score (a margin or log-odds). It is logistic regression with one feature, fitted by Newton's method. Platt's version uses smoothed targets `(N+ + 1)/(N+ + 2)` and `1/(N− + 2)` instead of 1 and 0 to avoid overfitting on small sets. Here `a = 0.348` means naive Bayes' log-odds are about 3× too large.
- **Isotonic regression.** Fit the best non-decreasing function from score to label (least squares). **Pool-adjacent-violators (PAV)** sorts by score, then merges neighbouring blocks whose means are out of order until the sequence is monotone, in O(n) after the sort. The result is a step function, interpolated between points.
- **Temperature scaling.** For neural networks: divide the logits by one learned T > 0. It is Platt scaling without the bias term, extended to softmax over many classes, and it is the usual default for deep classifiers.
- **Held-out data.** The calibrator must be fitted on data the model did not train on. Fitting it on training predictions learns the overconfidence of a fitted model, not of new data.

## When to use / scenarios
- Learning: the cleanest example of "ranking quality vs probability quality", and a real use of PAV ([[probability-calibration]]).
- Interviews: "is a 0.9 from your model really 90%", "Platt vs isotonic", "why is naive Bayes overconfident", "what is ECE and what is wrong with it".
- Practice: any decision that uses the probability itself, not only the ranking: expected-value thresholds (fraud cost vs review cost), risk scores shown to clinicians or underwriters, combining models, and inputs to [[conformal-prediction-and-uncertainty]] or bandits.
- Not for: problems where only the ranking or a fixed top-k matters (calibration leaves AUC unchanged); or fixing a model whose ranking is poor (calibrate after the model is good).

## Setup & code
NumPy and scikit-learn (data, the base model and a reference isotonic fit). Runs in about 4 seconds.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.isotonic import IsotonicRegression
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=30000, n_features=20, n_informative=6, n_redundant=10,
                           random_state=0)
Xtr, ytr = X[:10000], y[:10000]                  # fit the model
Xcal, ycal = X[10000:15000], y[10000:15000]      # fit the calibrator (held out!)
Xte, yte = X[15000:], y[15000:]                  # evaluate

nb = GaussianNB().fit(Xtr, ytr)                  # redundant features -> overconfident
s_cal, s_te = nb.predict_proba(Xcal)[:, 1], nb.predict_proba(Xte)[:, 1]


def log_odds(model, X):
    """Unsaturated score: predict_proba rounds to exactly 0/1, the joint log-likelihoods do not."""
    jll = model.predict_joint_log_proba(X)
    return jll[:, 1] - jll[:, 0]


f_cal, f_te = log_odds(nb, Xcal), log_odds(nb, Xte)
sigmoid = lambda z: np.exp(-np.logaddexp(0, -z))


def platt_fit(f, y, iters=100):
    """Fit p = sigmoid(a*f + b): Newton steps, halved until the loss drops, Platt's smoothed targets."""
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    F = np.column_stack([f, np.ones_like(f)])
    loss = lambda w: np.sum(t * np.logaddexp(0, -F @ w) + (1 - t) * np.logaddexp(0, F @ w))
    w = np.zeros(2)
    for _ in range(iters):
        p = sigmoid(F @ w)
        g = F.T @ (p - t)
        H = F.T @ (F * (p * (1 - p))[:, None]) + 1e-12 * np.eye(2)
        step = np.linalg.solve(H, g)
        while loss(w - step) > loss(w) and np.abs(step).max() > 1e-12:
            step /= 2
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w


def pav_fit(s, y):
    """Pool-adjacent-violators: non-decreasing step function from score to probability."""
    order = np.argsort(s, kind="mergesort")
    xs, ys = s[order], y[order].astype(float)
    vals, wts = [], []                           # blocks: mean label, number of points
    for v in ys:
        vals.append(v); wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:          # merge violating neighbours
            w = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            wts[-2] = w
            vals.pop(); wts.pop()
    fitted = np.repeat(vals, np.array(wts, dtype=int))
    return xs, fitted


def pav_predict(xs, fitted, s):
    return np.interp(s, xs, fitted)              # linear between points, clipped at ends


def ece(p, y, bins=15):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
               for b in range(bins) if (idx == b).any())


def logloss(p, y, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


a, b = platt_fit(f_cal, ycal)
p_platt = sigmoid(a * f_te + b)
xs, fitted = pav_fit(f_cal, ycal)
p_iso = pav_predict(xs, fitted, f_te)
p_sk = IsotonicRegression(out_of_bounds="clip").fit(f_cal, ycal).predict(f_te)

print(f"GaussianNB on 20 features (10 redundant); calibrator fit on {len(ycal)} held-out rows")
print(f"Platt on log-odds: a={a:.3f}, b={b:.3f}")
print(f"isotonic: {len(np.unique(fitted))} steps, max |ours - sklearn| on test = {np.abs(p_iso - p_sk).max():.2e}")
print("method          ECE    Brier  log loss  accuracy")
for name, p in [("uncalibrated", s_te), ("Platt", p_platt), ("isotonic", p_iso)]:
    print(f"{name:<13} {ece(p, yte):.3f}  {np.mean((p - yte) ** 2):.4f}  {logloss(p, yte):>8.3f}"
          f"  {np.mean((p > 0.5) == yte):.4f}")
print(f"share of uncalibrated test scores < 0.01 or > 0.99: {np.mean((s_te < .01) | (s_te > .99)):.2f}")
```

Output (Python 3.14, NumPy 2.5, scikit-learn 1.9):
```
GaussianNB on 20 features (10 redundant); calibrator fit on 5000 held-out rows
Platt on log-odds: a=0.348, b=0.084
isotonic: 45 steps, max |ours - sklearn| on test = 1.11e-16
method          ECE    Brier  log loss  accuracy
uncalibrated  0.102  0.1336     0.577  0.8367
Platt         0.017  0.1204     0.386  0.8352
isotonic      0.011  0.1202     0.398  0.8367
share of uncalibrated test scores < 0.01 or > 0.99: 0.57
```

Both calibrators fix most of the problem. Platt has the best log loss because its smooth sigmoid never outputs exactly 0 or 1. Isotonic has the best ECE and Brier because it can follow any monotone shape, but its 45 flat steps mean that many different scores receive the same probability, and its extreme steps can be very close to 0 or 1, which log loss punishes. Platt's accuracy moves slightly (0.8367 to 0.8352) because a non-zero bias b shifts which scores cross 0.5. The ranking, and so the AUC, is unchanged by both.

The `log_odds` helper matters. A first version fed `logit(predict_proba)` into Platt: so many naive Bayes probabilities were exactly 0.0 or 1.0 in float64 that the logits saturated at the clip value, and Newton's method without a line search diverged to `a ≈ 8.7e11`. For comparison, scikit-learn's `CalibratedClassifierCV(method="sigmoid")` on this naive Bayes uses `predict_proba` (GaussianNB has no `decision_function`) and gets Brier 0.1255, worse than 0.1204 on the unsaturated log-odds.

## Choosing / trade-offs
- **Platt scaling.** Use it with small calibration sets (hundreds of rows), when the distortion is roughly sigmoid-shaped (SVMs, boosted trees), and when you need smooth, strictly increasing output. It cannot fix non-sigmoid distortions.
- **Isotonic regression.** Use it with thousands of calibration rows or more, and when the distortion has an unknown shape. With little data it overfits: steps follow noise, and many inputs collapse to the same probability, creating ties that can slightly change AUC.
- **Temperature scaling.** The default for deep networks and multiclass softmax ([[bayesian-deep-learning-and-uncertainty]]). For multiclass with sklearn models, `CalibratedClassifierCV` calibrates one-vs-rest and renormalizes.
- **Beta calibration and spline calibration** sit between the two: more flexible than a sigmoid, smoother than steps.
- **Getting calibration data.** Use a separate split, or `CalibratedClassifierCV(cv=5)`, which fits calibrators on out-of-fold predictions and averages them. Re-calibrate when the base rate shifts ([[online-learning-and-concept-drift]]).

## Gotchas
- Calibrate on held-out predictions only. Calibrating on the training set barely changes an overfitted model's scores.
- Feed Platt scaling an unsaturated score (margin, log-odds, raw logits), not a probability that has already been rounded to 0 or 1.
- Use Newton with a line search, or a library logistic regression, for the Platt fit. Plain Newton from a bad start can diverge, as it did here.
- ECE changes with the number and placement of bins and can hide large errors in sparse bins. Always plot the reliability diagram, and report Brier score or log loss alongside.
- Class rebalancing (oversampling, `class_weight`) miscalibrates on purpose. Calibrate afterwards on data with the real class ratio, or correct the prior analytically ([[imbalanced-data]]).
- A calibrated model is calibrated on average, not for every subgroup. Check reliability per segment when decisions differ by segment.

## Related
- [[probability-calibration]] - overview, reliability diagrams and library usage.
- [[naive-bayes-from-scratch]] - why naive Bayes is overconfident with correlated features.
- [[logistic-regression-from-scratch]] - the Newton/IRLS fit behind Platt scaling.
- [[split-conformal-prediction-from-scratch]] - coverage guarantees that complement calibration.
- [[classification-metrics-and-cross-validation-from-scratch]] - Brier score, log loss and AUC.

## References
- scikit-learn user guide, "Probability calibration": https://scikit-learn.org/stable/modules/calibration.html
- Niculescu-Mizil and Caruana (2005), "Predicting Good Probabilities with Supervised Learning", ICML: https://doi.org/10.1145/1102351.1102430
- Zadrozny and Elkan (2002), "Transforming Classifier Scores into Accurate Multiclass Probability Estimates", KDD: https://doi.org/10.1145/775047.775151
- Guo, Pleiss, Sun and Weinberger (2017), "On Calibration of Modern Neural Networks": https://arxiv.org/abs/1706.04599
