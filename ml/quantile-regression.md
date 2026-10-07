---
title: Quantile regression and prediction intervals
category: ml
tags: [quantile-regression, pinball-loss, prediction-intervals, uncertainty, gradient-boosting, lightgbm, scikit-learn, forecasting]
use_cases:
  - "predict a range (P10-P90), not just a single number"
  - "forecast demand so stock covers 95% of days"
  - "estimate worst-case delivery time or latency (P95)"
  - "model a target whose spread grows with the input (heteroscedastic noise)"
  - "replace mean regression when the cost of under- and over-predicting differs"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/linear_model.html#quantile-regression
  - https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_pinball_loss.html
  - https://lightgbm.readthedocs.io/en/latest/Parameters.html
---

# Quantile regression and prediction intervals

## Summary
Ordinary regression predicts the conditional **mean**. Quantile regression predicts a conditional **quantile**: the value the target falls below with probability q. Training one model at q=0.05 and one at q=0.95 gives a 90% prediction interval whose width can change from row to row. It is the standard tool for demand forecasting (stock to the P90), SLA estimates (P95 latency), and any task where under- and over-predicting cost different amounts.

## Key concepts
- **Pinball (quantile) loss.** For residual r = y - ŷ: loss = q·r if r ≥ 0, else (q-1)·r. At q=0.5 it is half the absolute error, so the median model is the MAE model. At q=0.9 under-predicting costs 9x more than over-predicting, which pushes the prediction up to the 90th percentile.
- **Asymmetric cost = quantile.** If a stock-out costs `c_under` and overstock costs `c_over`, the cost-minimizing order is the quantile q = c_under / (c_under + c_over) (the newsvendor result).
- **Heteroscedasticity.** Quantile models capture spread that varies with the inputs. A mean model plus a constant ± band does not.
- **One model per quantile.** Most libraries fit each quantile separately. Nothing forces q=0.05 to sit below q=0.95, so quantiles can **cross**. Sort them or fit a multi-quantile model.
- **Coverage vs width.** A 90% interval should contain ~90% of held-out targets (coverage). Among intervals with correct coverage, narrower is better. Check both. Pinball loss scores a single quantile.

## When to use / scenarios
- Retail and supply chain: order quantities, safety stock, staffing at P80/P90.
- Logistics and ops: delivery ETA ranges, P95 job runtime, capacity planning.
- Energy: probabilistic load and solar/wind forecasts (P10/P50/P90 is the industry format).
- Finance: Value-at-Risk is a low quantile of the return distribution.
- Real estate and insurance: price or claim ranges instead of point estimates.
- NOT the right tool when you need a **guaranteed** coverage level. Quantile models often under-cover on new data. Wrap them in conformalized quantile regression (CQR) ([[conformal-prediction-and-uncertainty]]).
- NOT needed if a single point and its average error are enough (use MAE/RMSE regression, [[linear-models]], [[gradient-boosting-tabular]]).

## Setup & code
```bash
pip install "scikit-learn>=1.4" numpy
```
Gradient-boosted quantiles on data whose noise grows with x, plus a linear quantile model:
```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, size=(2000, 1))
y = 2 * X[:, 0] + rng.normal(scale=0.5 + 0.3 * X[:, 0])   # noise grows with x
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

preds = {}
for q in (0.05, 0.5, 0.95):
    m = HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=0)
    preds[q] = m.fit(X_tr, y_tr).predict(X_te)
    print(f"q={q}: pinball={mean_pinball_loss(y_te, preds[q], alpha=q):.3f}")

lo, hi = np.minimum(preds[0.05], preds[0.95]), np.maximum(preds[0.05], preds[0.95])
print(f"90% interval coverage on test: {np.mean((y_te >= lo) & (y_te <= hi)):.3f}")
narrow, wide = X_te[:, 0] < 2, X_te[:, 0] > 8
print(f"mean width x<2: {np.mean((hi - lo)[narrow]):.2f}  x>8: {np.mean((hi - lo)[wide]):.2f}")

lin = QuantileRegressor(quantile=0.9, alpha=0.0, solver="highs").fit(X_tr, y_tr)
print(f"linear q=0.9: coef={lin.coef_[0]:.2f} intercept={lin.intercept_:.2f}")
```
Output (scikit-learn 1.9.0):
```
q=0.05: pinball=0.207
q=0.5: pinball=0.838
q=0.95: pinball=0.240
90% interval coverage on test: 0.828
mean width x<2: 2.22  x>8: 9.65
linear q=0.9: coef=2.39 intercept=0.67
```
The interval widens where the noise is larger (2.2 vs 9.7). The nominal 90% interval covers only 83% of test points, which is typical. Calibrate it with CQR before promising a coverage level.

Other libraries: LightGBM/XGBoost (`objective="quantile", alpha=q`; XGBoost ≥2.0 also has `reg:quantileerror` with several quantiles in one model), CatBoost (`loss_function="MultiQuantile:alpha=0.1,0.5,0.9"`), statsmodels `QuantReg` (classic linear model with standard errors), and in PyTorch any network trained with pinball loss on K output heads.

## Choosing / trade-offs
- **Linear (`QuantileRegressor`, `QuantReg`) vs boosted trees.** Linear models are interpretable and give per-quantile coefficients. "Income effect at the 10th vs 90th percentile" is a classic econometrics use. Trees capture non-linearity and interactions; use them for prediction. `QuantileRegressor` solves a linear program and gets slow beyond ~100k rows; use the `highs` solver, or subsample.
- **Separate models vs multi-quantile.** Separate models are simplest but can cross and cost K fits. Multi-quantile (CatBoost MultiQuantile, XGBoost quantile list, a neural net with K heads) shares structure and crosses less.
- **Quantile regression vs conformal vs distributional models.** Quantile regression gives adaptive width but no guarantee. Split conformal on a point model gives a guarantee but constant width. CQR gives both. Distributional models (NGBoost, Gaussian/Tweedie heads) give a full distribution but assume its shape ([[generalized-linear-models]]).
- **Which quantiles.** Derive q from business costs (newsvendor) instead of defaulting to 0.05/0.95.

## Gotchas
- Extreme quantiles (0.01, 0.99) need lots of data. With few points in the tail the estimate is noisy and boosting overfits it. Raise `min_samples_leaf` and regularize more than for the median.
- Report coverage **and** mean width on a held-out set, broken down by segment. Global 90% coverage can hide 70% on the segment that matters.
- Quantile crossing: sort the predicted quantiles per row (`np.sort(preds, axis=1)`), or the interval can have lo > hi.
- Do not judge a quantile model with RMSE or R². Use `mean_pinball_loss(alpha=q)` and coverage.
- Gradient boosting's quantile loss has zero second derivative. LightGBM/XGBoost work around it, but convergence is slower and leaf values can be crude. Use more trees and a lower learning rate than for squared error.
- Sums of quantiles are not quantiles: the P90 of total demand is not the sum of per-store P90s. Aggregate the samples or forecast at the level you decide on.
- In time series, interval coverage decays with horizon and under drift. Re-check coverage on recent windows ([[time-series-forecasting]], [[online-learning-and-concept-drift]]).

## Related
- [[conformal-prediction-and-uncertainty]] - CQR turns quantile models into intervals with guaranteed coverage.
- [[loss-functions]] - pinball loss next to MSE, MAE, Huber.
- [[gradient-boosting-tabular]] - the usual engine for tabular quantile models.
- [[time-series-forecasting]] - P10/P50/P90 probabilistic forecasts.
- [[generalized-linear-models]] - distributional alternative when the noise family is known.
- [[model-evaluation-and-metrics]] - regression metrics and segment-level evaluation.

## References
- scikit-learn, Quantile regression (linear): https://scikit-learn.org/stable/modules/linear_model.html#quantile-regression
- scikit-learn, Prediction intervals for gradient boosting regression: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
- scikit-learn, `mean_pinball_loss`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_pinball_loss.html
- LightGBM parameters (`objective=quantile`, `alpha`): https://lightgbm.readthedocs.io/en/latest/Parameters.html
- Koenker & Bassett, "Regression Quantiles", Econometrica 1978.
- Romano, Patterson & Candès, "Conformalized Quantile Regression", NeurIPS 2019: https://arxiv.org/abs/1905.03222
