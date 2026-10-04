---
title: AR and ARIMA(p,d,0) forecasting from scratch (least squares, AIC, differencing)
category: ml
tags: [arima, autoregressive, ar-model, time-series, forecasting, differencing, aic, stationarity, rolling-origin, numpy, from-scratch, ml-basics]
use_cases:
  - "fit an autoregressive AR(p) model with least squares in NumPy"
  - "choose the AR order with AIC and forecast several steps ahead recursively"
  - "difference a trending series (ARIMA d=1) and integrate the forecast back"
  - "compare ARIMA with naive and drift baselines using rolling-origin evaluation"
  - "explain stationarity, differencing and AR/MA terms in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://otexts.com/fpp3/arima.html
  - https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
  - https://otexts.com/fpp3/tscv.html
---

# AR and ARIMA(p,d,0) forecasting from scratch (least squares, AIC, differencing)

## Summary
An autoregressive model AR(p) predicts the next value of a series as a linear combination of its last `p` values plus a constant: ordinary linear regression on lagged copies of the series. ARIMA(p, d, q) differences the series `d` times to remove a trend, fits AR(p) plus q moving-average terms to the result, and sums the forecasts back up. The NumPy version below covers p and d (q = 0, so plain least squares suffices). On a simulated AR(2) series it recovers the coefficients (0.59 and −0.32 against 0.6 and −0.3), and AIC picks the right order. On a trending series, fitting AR to the raw levels gives a 20-step RMSE of 2.49; differencing first (ARIMA(2,1,0)) brings it to 1.69. In a rolling-origin test over 100 forecast origins, ARIMA beats the drift baseline clearly one step ahead (MAE 0.73 vs 0.82) but only barely at five steps (1.43 vs 1.44), because AR memory fades with the horizon. Always compare against naive baselines.

## Key concepts
- **AR(p).** `y_t = c + φ_1 y_{t−1} + … + φ_p y_{t−p} + ε_t`. Fit it by least squares on a lag matrix; that is close to the maximum-likelihood estimate for Gaussian noise.
- **Stationarity.** Mean and autocovariance do not change over time. AR models assume it; a trend or a random walk violates it.
- **Differencing (the "I").** `∇y_t = y_t − y_{t−1}` turns a random walk with drift into a stationary series. Forecast the differences, then cumulative-sum them from the last observed level. `d = 1` is enough for most trending data; seasonal differencing (`y_t − y_{t−m}`) handles seasonality.
- **MA(q).** Moving-average terms regress on past forecast errors, which are not observed directly, so fitting needs iterative maximum likelihood (statsmodels does it). They are omitted here.
- **Recursive forecasting.** For horizon `h > 1`, feed each forecast back in as a lag. Forecasts decay toward the series mean (or, after differencing, toward a straight line with slope equal to the mean difference).
- **AIC.** `n log σ̂² + 2k`: fit penalised by the number of parameters. Compare orders on the same rows of data, or the comparison is biased.
- **Rolling origin.** Refit at many cut-off points and forecast forward from each. Time-series cross-validation never trains on the future.

## When to use / scenarios
- Learning: forecasting as regression on lags, and why differencing exists.
- Interviews: "what does the I in ARIMA do", "how do you choose p", "why do long-horizon ARIMA forecasts go flat", "AR vs MA", "how do you validate a forecast".
- Short univariate series (hundreds of points) with a stable structure: demand, sensor readings, KPIs, as a strong classical baseline.
- Not for: many related series with shared patterns or many covariates (use gradient boosting on lag features or a global neural model, see [[time-series-forecasting]]), strong multiple seasonality (use ETS, Prophet-style or MSTL decompositions), or regime changes.

## Setup & code
`pip install numpy`. Runs in under a second on CPU.

```python
import numpy as np


def lagmat(y, p):
    """Rows [1, y_{t-1}, ..., y_{t-p}] for t = p..n-1."""
    return np.column_stack([np.ones(len(y) - p)] + [y[p - k:len(y) - k] for k in range(1, p + 1)])


def fit_ar(y, p):
    X, t = lagmat(y, p), y[p:]
    coef, *_ = np.linalg.lstsq(X, t, rcond=None)
    resid = t - X @ coef
    sigma2 = resid @ resid / len(t)
    aic = len(t) * np.log(sigma2) + 2 * (p + 1)
    return coef, sigma2, aic


def forecast_ar(y, coef, h):
    """Recursive multi-step forecast: feed each prediction back in as a lag."""
    p, hist = len(coef) - 1, list(y)
    for _ in range(h):
        hist.append(coef[0] + sum(coef[k] * hist[-k] for k in range(1, p + 1)))
    return np.array(hist[len(y):])


def fit_forecast_arima(y, p, d, h):
    """ARIMA(p, d, 0): difference d times, fit AR(p), forecast, integrate back."""
    z = y.copy()
    lasts = []
    for _ in range(d):
        lasts.append(z[-1])
        z = np.diff(z)
    coef, _, _ = fit_ar(z, p)
    fc = forecast_ar(z, coef, h)
    for last in reversed(lasts):
        fc = last + np.cumsum(fc)
    return fc, coef


rng = np.random.default_rng(0)
n, h = 400, 20
# AR(2) process: y_t = 0.6 y_{t-1} - 0.3 y_{t-2} + e_t (part 2 sums it into a trending series)
e = rng.normal(0, 1, n + h)
ar = np.zeros(n + h)
for t in range(2, n + h):
    ar[t] = 0.6 * ar[t - 1] - 0.3 * ar[t - 2] + e[t]
print("1) stationary AR(2), true coefficients [0.6, -0.3]")
y, y_future = ar[:n], ar[n:]
P_MAX = 8                                   # compare AICs on the same rows: drop P_MAX - p extra points
for p in [1, 2, 3, 5, 8]:
    coef, s2, aic = fit_ar(y[P_MAX - p:], p)
    print(f"   AR({p}): AIC {aic:7.1f} | lag coefs {np.round(coef[1:], 3).tolist()}")
best_p = min(range(1, P_MAX + 1), key=lambda p: fit_ar(y[P_MAX - p:], p)[2])
print(f"   AIC picks p = {best_p}")

print("2) forecasting a trending series (AR(2) noise on the differences: an integrated process)")
trend = np.cumsum(0.3 + ar)                 # y_t = y_{t-1} + 0.3 + AR(2) noise -> needs d = 1
y, y_future = trend[:n], trend[n:]
rmse = lambda fc: np.sqrt(np.mean((fc - y_future) ** 2))
naive = np.full(h, y[-1])
drift = y[-1] + (y[-1] - y[0]) / (n - 1) * np.arange(1, h + 1)
fc_d0, _ = fit_forecast_arima(y, 2, 0, h)
fc_d1, coef = fit_forecast_arima(y, 2, 1, h)
print(f"   naive (last value)      RMSE {rmse(naive):6.2f}")
print(f"   drift (last + avg step) RMSE {rmse(drift):6.2f}")
print(f"   ARIMA(2,0,0) on levels  RMSE {rmse(fc_d0):6.2f}")
print(f"   ARIMA(2,1,0)            RMSE {rmse(fc_d1):6.2f} | coefs [c, phi1, phi2] {np.round(coef, 3).tolist()}")

print("3) rolling origin: refit at each of 100 origins (t = 300..399), MAE by horizon")
for H in [1, 5]:
    errs = {"naive": [], "drift": [], "ARIMA(2,1,0)": []}
    for origin in range(300, 400):
        yt, yf = trend[:origin], trend[origin:origin + H]
        errs["naive"].append(np.abs(yt[-1] - yf).mean())
        errs["drift"].append(np.abs(yt[-1] + (yt[-1] - yt[0]) / (origin - 1) * np.arange(1, H + 1) - yf).mean())
        errs["ARIMA(2,1,0)"].append(np.abs(fit_forecast_arima(yt, 2, 1, H)[0] - yf).mean())
    print(f"   {H}-step: " + " | ".join(f"{k} {np.mean(v):.3f}" for k, v in errs.items()))
```

Output (numpy 2.5):
```
1) stationary AR(2), true coefficients [0.6, -0.3]
   AR(1): AIC    46.5 | lag coefs [0.447]
   AR(2): AIC     6.0 | lag coefs [0.591, -0.319]
   AR(3): AIC     8.0 | lag coefs [0.592, -0.321, 0.002]
   AR(5): AIC    11.5 | lag coefs [0.592, -0.309, -0.019, 0.037, -0.002]
   AR(8): AIC    15.7 | lag coefs [0.592, -0.309, -0.02, 0.05, -0.017, 0.005, 0.05, -0.057]
   AIC picks p = 2
2) forecasting a trending series (AR(2) noise on the differences: an integrated process)
   naive (last value)      RMSE   3.10
   drift (last + avg step) RMSE   1.68
   ARIMA(2,0,0) on levels  RMSE   2.49
   ARIMA(2,1,0)            RMSE   1.69 | coefs [c, phi1, phi2] [0.178, 0.594, -0.322]
3) rolling origin: refit at each of 100 origins (t = 300..399), MAE by horizon
   1-step: naive 0.838 | drift 0.819 | ARIMA(2,1,0) 0.734
   5-step: naive 1.609 | drift 1.441 | ARIMA(2,1,0) 1.426
```

Part 1: AR(2) recovers the true coefficients to within 0.02, and AR(1) misses badly (AIC 46.5 against 6.0). Higher orders add near-zero coefficients that AIC penalises, so it picks p = 2. Part 2: the second series accumulates AR(2) steps plus a drift of 0.3, so its level wanders without a fixed mean. AR on the raw levels treats that as a near-unit-root process and does poorly over 20 steps (RMSE 2.49). Differencing once makes the series the original AR(2) plus a constant, and ARIMA(2,1,0) recovers the same φ values from it. Over a single 20-step window, though, it only ties the drift baseline (1.69 vs 1.68). Part 3 shows why. Across 100 refits, the AR terms cut one-step error by 10% against drift (0.734 vs 0.819) because the last two changes carry real information about the next one. By five steps the AR(2) memory (φ₁ = 0.6) has mostly decayed, and both methods reduce to "last value plus the average slope". A gain at short horizons that disappears at longer ones is typical of ARIMA; measure it at the horizon you actually need.

## Choosing / trade-offs
- **Order selection.** Use AIC/AICc or rolling-origin error over a small grid; `auto_arima` (pmdarima) and statsforecast's `AutoARIMA` automate this, including unit-root tests to pick `d`.
- **d.** Choose it with a unit-root test (KPSS or ADF) or by eye on the plot, not by trial and error on test error. Over-differencing adds noise; `d = 2` is rarely needed.
- **MA terms and seasonality.** Need maximum-likelihood fitting: use `statsmodels.tsa.arima.model.ARIMA` or statsforecast. SARIMA adds seasonal AR, MA and differencing at lag `m`.
- **ARIMA vs ETS vs ML.** ETS (exponential smoothing) is often as accurate for level/trend/seasonal series. Gradient boosting on lag features wins with many related series or covariates. Deep global models win with thousands of series. A seasonal-naive or drift baseline should always be in the comparison.
- **Intervals.** ARIMA gives analytic prediction intervals that widen with horizon (they assume Gaussian, constant-variance errors); check them with backtesting or use conformal intervals.

## Gotchas
- Never use a random train/test split or K-fold on time series; it leaks the future. Split by time and evaluate with a rolling origin ([[data-leakage-and-validation-splits]]).
- Comparing AIC across orders fitted on different numbers of rows is invalid. Drop the first `p_max − p` points so every model sees the same targets (`P_MAX` above).
- A constant in a differenced model means a linear trend in the levels; with `d = 2` it means a quadratic trend that can explode over long horizons.
- Multi-step recursive forecasts compound errors. The alternative is direct forecasting: one model per horizon.
- Log-transform positive series with multiplicative growth or variance that grows with the level before fitting, and transform back (with a bias correction if you report means).
- Outliers and level shifts distort least-squares AR fits; clean them or model them with intervention dummies.
- If the drift baseline is as good as ARIMA, report that. It is a common and honest outcome (part 2 above).

## Related
- [[time-series-forecasting]] - statsforecast, ETS, Prophet-style, gradient boosting and neural forecasters.
- [[linear-regression-from-scratch]] - the least-squares solver that AR fitting reuses.
- [[kalman-filter-from-scratch]] - state-space models; ARIMA can be written in state-space form and fit with a Kalman filter.
- [[hmm-from-scratch]] - a discrete-state sequence model, an alternative to linear dynamics.
- [[data-leakage-and-validation-splits]] - time-based splits and leakage.
- [[regression-metrics-and-residual-analysis]] - MAE/RMSE and checking that residuals look like white noise.

## References
- Hyndman and Athanasopoulos, Forecasting: Principles and Practice, 3rd ed., Chapter 9 (ARIMA models): https://otexts.com/fpp3/arima.html
- Hyndman and Athanasopoulos, FPP3 Section 5.10, time series cross-validation: https://otexts.com/fpp3/tscv.html
- statsmodels ARIMA reference: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
