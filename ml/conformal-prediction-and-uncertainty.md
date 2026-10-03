---
title: Conformal prediction and uncertainty quantification
category: ml
tags: [conformal-prediction, uncertainty, prediction-intervals, prediction-sets, calibration, coverage, mapie, quantile-regression]
use_cases:
  - "put a guaranteed 90% interval around any regression model's prediction"
  - "return a set of plausible labels instead of one, and send ambiguous cases to a human"
  - "decide when a model is unsure enough to abstain or escalate"
  - "add uncertainty to a gradient boosting or deep model without making it Bayesian"
  - "report forecast ranges rather than point estimates to stakeholders"
status: draft
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/2107.07511
  - https://mapie.readthedocs.io/en/stable/
  - https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
---

# Conformal prediction and uncertainty quantification

## Summary
Conformal prediction wraps any trained model and turns its point predictions
into intervals (regression) or label sets (classification) that contain the
true answer with a chosen probability, for example 90%. The guarantee holds
for any model and any data distribution, as long as calibration and test data
are exchangeable (roughly: drawn from the same distribution). It needs only a
held-out calibration set and a few lines of code.

## Key concepts
- Kinds of uncertainty: aleatoric (noise in the data, does not shrink with
  more data) and epistemic (model ignorance, shrinks with data). Intervals
  from conformal cover both, without separating them.
- Nonconformity score: how wrong the model is on a calibration row. Regression:
  `|y - y_hat|`. Classification: `1 - p(true class)`.
- Split conformal: fit on train, score on a separate calibration set, take the
  `ceil((n+1)(1-alpha))/n` quantile `q` of the scores. Test interval is
  `y_hat +- q`; test set is every class with `p >= 1 - q`.
- Coverage guarantee is marginal: on average over test rows, at least
  `1 - alpha`. It is not per row, per subgroup or conditional on x.
- Adaptive variants: conformalized quantile regression (CQR) gives wider
  intervals where the data is noisier; APS/RAPS scores give better-sized
  classification sets; Mondrian (class-conditional) conformal gives coverage
  per group.
- Cross-conformal / jackknife+ reuse data instead of holding out a
  calibration set, at the cost of refitting K models.
- Calibration (probabilities that match frequencies, see
  [[model-evaluation-and-metrics]]) is a different property. A calibrated
  model can still need conformal intervals, and conformal does not fix
  miscalibrated probabilities.

## When to use / scenarios
- Finance/insurance: price or loss ranges for underwriting; flag wide
  intervals for manual review.
- Healthcare: label sets for triage; a set with more than one diagnosis goes
  to a clinician.
- Retail/supply chain: demand ranges for safety stock
  ([[time-series-forecasting]], with time-series variants such as EnbPI or
  adaptive conformal inference).
- Document/ticket classification: auto-route singleton sets, escalate the rest.
- Any model that is already trained and you cannot or will not retrain.
- NOT for: small data where you also want to reason about parameters (use
  [[bayesian-and-gaussian-processes]]); data with strong drift between
  calibration and serving (the guarantee breaks; see
  [[online-learning-and-concept-drift]] and recalibrate often).

## Setup & code
```bash
pip install scikit-learn numpy
```
```python
import numpy as np
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

alpha = 0.1  # target 90% coverage

# --- Split conformal regression ---
X, y = fetch_california_housing(return_X_y=True)
X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.4, random_state=0)
X_cal, X_te, y_cal, y_te = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)

model = HistGradientBoostingRegressor(random_state=0).fit(X_tr, y_tr)
scores = np.abs(y_cal - model.predict(X_cal))                 # nonconformity
n = len(scores)
q = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")
pred = model.predict(X_te)
covered = (y_te >= pred - q) & (y_te <= pred + q)
print(f"regression: width={2*q:.3f} coverage={covered.mean():.3f}")

# --- Split conformal classification (prediction sets) ---
X, y = load_digits(return_X_y=True)
X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.5, random_state=0)
X_cal, X_te, y_cal, y_te = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
clf = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)
p_cal = clf.predict_proba(X_cal)
scores = 1 - p_cal[np.arange(len(y_cal)), y_cal]
n = len(scores)
q = np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher")
sets = clf.predict_proba(X_te) >= 1 - q                       # boolean (n_test, n_classes)
print(f"classification: coverage={sets[np.arange(len(y_te)), y_te].mean():.3f} "
      f"avg set size={sets.sum(1).mean():.2f} singletons={(sets.sum(1) == 1).mean():.3f}")
```
Output (scikit-learn 1.9.0): regression interval width 1.388 (house value in
$100k units) with 0.891 test coverage; classification coverage 0.902, 92% of
sets are singletons and the average set size is 0.92, because the remaining 8%
are empty sets. An empty set means "no class is plausible": treat it as an
abstain, not as a prediction.

The constant-width regression interval is the weakness of this simplest form.
For width that follows the noise level, use CQR: fit two quantile models
(`HistGradientBoostingRegressor(loss="quantile", quantile=0.05/0.95)`) and
conformalize their gap the same way. MAPIE packages split, cross, CQR, APS
and time-series variants behind a scikit-learn API; check its current docs,
since the API changed between major versions.

## Choosing / trade-offs
- Enough data for a calibration split (a few hundred rows or more): split
  conformal. Little data: cross-conformal / jackknife+ (K refits).
- Noise varies with x: CQR or a normalized score (`|y - y_hat| / sigma_hat(x)`).
- Need coverage per class or per customer segment: Mondrian conformal, with
  enough calibration rows in each group.
- Need parameter uncertainty or priors: Bayesian models; deep ensembles or MC
  dropout for deep nets. These give uncertainty but no coverage guarantee.
- alpha is a business choice: lower alpha means wider intervals and larger sets.

## Gotchas
- Reusing training rows for calibration gives intervals that are far too
  narrow. The calibration set must be unseen by the model.
- Coverage is on average, not per row; small subgroups can be badly covered.
  Report coverage per segment.
- Distribution shift (new season, new product line) breaks exchangeability;
  recalibrate on recent data, or use adaptive conformal for time series.
- With very few calibration rows the quantile index can exceed n; the
  interval is then infinite. Use at least ~100 rows for alpha = 0.1.
- Empty prediction sets surprise downstream code; decide what they mean.
- Coverage that matches the target on one split can vary from run to run;
  check across several random splits.

## Related
- [[bayesian-and-gaussian-processes]] - model-based uncertainty for small data.
- [[model-evaluation-and-metrics]] - calibration, which is a different property.
- [[gradient-boosting-tabular]] - the usual base model; supports quantile loss.
- [[time-series-forecasting]] - forecast intervals.
- [[online-learning-and-concept-drift]] - shift that breaks the guarantee.
- [[finance]], [[healthcare]] - human-in-the-loop on uncertain cases.

## References
- Angelopoulos & Bates, A Gentle Introduction to Conformal Prediction: https://arxiv.org/abs/2107.07511
- MAPIE documentation: https://mapie.readthedocs.io/en/stable/
- scikit-learn quantile gradient boosting example: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
