---
title: Regression metrics and residual analysis (MAE, RMSE, R², MAPE)
category: ml
tags: [regression, evaluation, metrics, mae, rmse, r2, mape, residuals, baseline, scikit-learn]
use_cases:
  - "choose the right error metric for a price, demand or duration prediction model"
  - "explain regression error to a business owner in rupees, units or minutes"
  - "find where a regression model is systematically wrong (biased segments, capped targets)"
  - "check whether a regression model is actually better than predicting the average"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
  - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
  - https://otexts.com/fpp3/accuracy.html
---

# Regression metrics and residual analysis (MAE, RMSE, R², MAPE)

## Summary
A regression model predicts a number, and the metric decides what "good" means. MAE, RMSE, MAPE and R² rank models differently because they punish different errors. A single score also hides *where* a model is wrong. Residual analysis (actual minus predicted, broken down by segment and prediction range) shows systematic bias that one aggregate number averages away. Always compare against a predict-the-mean baseline.

## Key concepts
- **MAE** = mean |y − ŷ|. Same units as the target, robust to outliers, minimised by predicting the **median**. The easiest to explain ("off by 31k on average").
- **RMSE** = √mean (y − ŷ)². Same units, squares errors so big misses dominate, minimised by the **mean**. Always ≥ MAE. A large RMSE/MAE ratio means a few big errors.
- **MedAE** (median absolute error). The typical error, ignoring the tail entirely.
- **MAPE** = mean |y − ŷ| / |y|. Relative error, scale-free, but explodes when y is near 0 and penalises over-prediction more than under-prediction. Prefer **WAPE** (Σ|y − ŷ| / Σ|y|) for demand.
- **R²** = 1 − SSE/SST. The share of variance explained relative to predicting the mean. 0 means no better than the mean, and it is negative when worse. It is not an error in business units, and it depends on how spread out the test set is.
- **Residual** = y − ŷ. Its mean is overall bias. Its pattern against ŷ, features or time shows what the model has not learned.
- **Loss ≠ metric.** Train with the loss that targets the statistic you want (MSE → mean, MAE → median, pinball → quantile, see [[quantile-regression]]), then report the business metric.

## When to use / scenarios
- House or car price prediction: MAE in currency for stakeholders, RMSE when a large miss (e.g. a mispriced loan collateral) is much worse than several small ones.
- Retail demand forecasting: WAPE or MAE per SKU. MAPE breaks on slow movers with zero sales ([[time-series-forecasting]]).
- Delivery ETA or service duration: MAE in minutes plus residuals by city or hour, to find segments that are always late.
- Model comparison: report R² only together with an error in units. R² alone hides whether the error is acceptable.
- Not for ranked outputs or probabilities. Use ranking metrics ([[learning-to-rank]]) or classification metrics ([[model-evaluation-and-metrics]]).

## Setup & code
```bash
pip install "scikit-learn>=1.4" numpy
```
```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, root_mean_squared_error, r2_score,
                             mean_absolute_percentage_error, median_absolute_error)
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True)  # target: median house value, $100k
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

for name, m in [("mean baseline", DummyRegressor()),
                ("linear", LinearRegression()),
                ("boosting", HistGradientBoostingRegressor(random_state=0))]:
    p = m.fit(X_tr, y_tr).predict(X_te)
    print(f"{name:14s} MAE={mean_absolute_error(y_te, p):.3f} "
          f"RMSE={root_mean_squared_error(y_te, p):.3f} "
          f"MedAE={median_absolute_error(y_te, p):.3f} "
          f"MAPE={mean_absolute_percentage_error(y_te, p):.3f} R2={r2_score(y_te, p):.3f}")

# Residual analysis on the boosting model: where is it wrong?
res = y_te - p
print("mean residual (bias):", round(res.mean(), 3))
bins = np.quantile(p, [0, .25, .5, .75, 1])
idx = np.digitize(p, bins[1:-1])
for i in range(4):
    r = res[idx == i]
    print(f"pred quartile {i + 1}: mean resid={r.mean():+.3f} MAE={np.abs(r).mean():.3f}")
# Capped target: the data stops at 5.0 ($500k), so the top is under-predicted.
print("rows at cap 5.0:", int((y_te >= 5.0).sum()), "mean resid there:",
      round(res[y_te >= 5.0].mean(), 3))
```
Output (scikit-learn 1.9, target in $100k):

| Model | MAE | RMSE | MedAE | MAPE | R² |
|---|---|---|---|---|---|
| Mean baseline | 0.914 | 1.150 | 0.770 | 0.632 | 0.000 |
| Linear | 0.537 | 0.735 | 0.419 | 0.327 | 0.591 |
| Gradient boosting | 0.313 | 0.467 | 0.208 | 0.182 | 0.835 |

The boosting model's overall mean residual is −0.015 (nearly unbiased), but its MAE grows from 0.17 in the cheapest quartile of predictions to 0.47 in the most expensive, so errors scale with price. The 241 test houses at the $500k cap are under-predicted by 0.55 ($55k) on average. The cap is a data artefact that no metric total shows.

## Choosing / trade-offs
- **MAE vs RMSE.** Is a 100k miss as bad as ten 10k misses (MAE) or worse (RMSE)? Ask the business, then train with the matching loss.
- **Absolute vs relative.** Targets spanning orders of magnitude (prices from 10k to 10M) make absolute errors dominated by expensive items. Use relative error, or fit on `log(y)` and report RMSLE.
- **Errors that grow with the target** (heteroscedastic, as above): log-transform the target, or give intervals that widen with ŷ ([[quantile-regression]], [[conformal-prediction-and-uncertainty]]).
- **Report a set.** One error in units (MAE), one tail-sensitive metric (RMSE or P90 absolute error), and the baseline beside them.

## Gotchas
- **Skipping the baseline.** A 0.5 MAE sounds fine until the mean predictor scores 0.6. `DummyRegressor` costs one line.
- **MAPE with zeros or tiny values** divides by ~0 and one row dominates. Use WAPE, sMAPE or MAE.
- **R² across datasets.** The same model gets a higher R² on a test set with more spread. Do not compare R² between datasets or time periods.
- **Negative R² is allowed** and means worse than predicting the mean, often from a train/test distribution shift.
- **Log-target models predict the median, not the mean.** `exp(prediction of log y)` is biased low for the mean. Apply a smearing correction if totals must add up (e.g. summed demand).
- **Aggregates hide segments.** Always slice residuals by key features (region, product line, time) and by prediction quantile, and plot residuals against ŷ.

## Related
- [[model-evaluation-and-metrics]] - classification metrics, thresholds and bootstrap confidence intervals.
- [[linear-models]] - the linear baseline and its assumptions about residuals.
- [[quantile-regression]] - predicting intervals and quantiles instead of a single mean.
- [[time-series-forecasting]] - MASE, WAPE and backtesting for forecasts.
- [[loss-functions]] - MSE, MAE, Huber and which statistic each one targets.
- [[bias-variance-and-learning-curves]] - diagnosing whether error comes from bias or variance.

## References
- scikit-learn, regression metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
- California housing dataset (target capped at 5.0): https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, forecast accuracy: https://otexts.com/fpp3/accuracy.html
