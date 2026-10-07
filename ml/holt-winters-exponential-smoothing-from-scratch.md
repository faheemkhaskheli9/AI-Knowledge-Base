---
title: Holt-Winters exponential smoothing from scratch (level, trend, additive season)
category: ml
tags: [time-series, forecasting, exponential-smoothing, holt-winters, seasonality, trend, ets, numpy, from-scratch, ml-basics]
use_cases:
  - "forecast a monthly series with trend and yearly seasonality in plain NumPy"
  - "see what each smoothing parameter (alpha, beta, gamma) controls"
  - "compare simple exponential smoothing, Holt and Holt-Winters against a seasonal-naive baseline"
  - "explain exponential smoothing / ETS in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://otexts.com/fpp3/holt-winters.html
  - https://otexts.com/fpp3/ses.html
  - https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
---

# Holt-Winters exponential smoothing from scratch (level, trend, additive season)

## Summary
Exponential smoothing forecasts with a weighted average of past observations whose weights decay geometrically, so recent points count most. Simple exponential smoothing (SES) tracks only a level. Holt's method adds a trend. Holt-Winters adds a seasonal component as well, which gives three smoothing parameters: α (level), β (trend) and γ (season). Below, all three are fitted by grid search on one-step-ahead squared error, using ten years of synthetic monthly data with a linear trend and a two-harmonic yearly season. Each method then forecasts the next 24 months. SES and Holt cannot represent the season and score an MAE of 14.0 and 10.8. Seasonal naive ("same month last year") ignores the trend and scores 14.0. Holt-Winters scores 3.58, close to the 3.17 noise floor of the true mean curve.

## Key concepts
- **Level update.** `ℓ_t = α (y_t − s_{t−m}) + (1 − α)(ℓ_{t−1} + b_{t−1})`. This blends the deseasonalized observation with the previous one-step forecast. High α reacts fast and is noisy; low α is smooth and lags.
- **Trend update.** `b_t = β (ℓ_t − ℓ_{t−1}) + (1 − β) b_{t−1}`. This smooths the change in level.
- **Seasonal update.** `s_t = γ (y_t − ℓ_t) + (1 − γ) s_{t−m}`. Each of the `m` seasonal slots is updated once per cycle, which is why γ usually needs to be small.
- **Forecast.** `ŷ_{t+k} = ℓ_t + k b_t + s_{t+k−m(⌊(k−1)/m⌋+1)}`. The level plus k trend steps plus the matching seasonal slot. Additive seasons keep a fixed amplitude. Multiplicative seasons (`y / s`) grow with the level.
- **Initialization.** Here the level is the first season's mean, the trend is the difference between the first two season means divided by `m`, and the season is the first cycle minus that level. The first cycle is skipped when computing the fit error, because it still reflects that crude start.
- **ETS view.** Holt-Winters is the point forecast of a state-space model with additive errors (ETS(A,A,A)). That model gives a likelihood, AIC and prediction intervals, which the plain recursions do not.

## When to use / scenarios
- Learning: the simplest model with explicit level, trend and season states, and a natural bridge to Kalman filters ([[kalman-filter-from-scratch]]) and state-space time-series models.
- Interviews: "what does alpha do", "SES vs Holt vs Holt-Winters", "additive vs multiplicative seasonality", "why is seasonal naive a hard baseline".
- Practice: demand, call-volume, energy-load and sales forecasts for many short, regular series where you need a fast, robust baseline for each one. ETS is a standard benchmark in forecasting competitions.
- Not for: series driven by external regressors such as price, promotions or weather (use regression with lags, or gradient boosting on features: [[time-series-forecasting]]); several interacting seasonalities such as daily plus weekly patterns (use TBATS, Prophet or MSTL); or irregular, intermittent demand (use Croston-type methods).

## Setup & code
NumPy only. The 19³-point grid search over (α, β, γ) runs in about 20 seconds.

```python
import itertools
import numpy as np

rng = np.random.default_rng(0)
m = 12                                                   # monthly data, yearly season
n = 12 * 10
t = np.arange(n + 24)
truth = 100 + 0.8 * t + 15 * np.sin(2 * np.pi * t / m) + 6 * np.cos(4 * np.pi * t / m)
y_all = truth + rng.normal(0, 4, len(t))
y, y_test = y_all[:n], y_all[n:]


def holt_winters(y, alpha, beta, gamma, m, h):
    """Additive Holt-Winters. Returns one-step-ahead fitted values and an h-step forecast."""
    level = y[:m].mean()
    trend = (y[m:2 * m].mean() - y[:m].mean()) / m
    season = list(y[:m] - level)
    fitted = np.zeros(len(y))
    for i in range(len(y)):
        s = season[i % m]
        fitted[i] = level + trend + s
        prev_level = level
        level = alpha * (y[i] - s) + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        season[i % m] = gamma * (y[i] - level) + (1 - gamma) * s
    k = np.arange(1, h + 1)
    forecast = level + k * trend + np.array([season[(len(y) + j - 1) % m] for j in k])
    return fitted, forecast


def fit(y, m, h, use_trend=True, use_season=True):
    grid = np.linspace(0.05, 0.95, 19)
    best = None
    for a, b, g in itertools.product(grid, grid if use_trend else [0.0], grid if use_season else [0.0]):
        if use_season:
            fitted, fc = holt_winters(y, a, b, g, m, h)
        else:
            fitted, fc = holt(y, a, b, h, use_trend)
        sse = np.sum((y[m:] - fitted[m:]) ** 2)        # skip the warm-up season
        if best is None or sse < best[0]:
            best = (sse, (a, b, g), fc)
    return best


def holt(y, alpha, beta, h, use_trend):
    """Simple exponential smoothing (use_trend=False) or Holt's linear trend."""
    level, trend = y[0], (y[1] - y[0]) if use_trend else 0.0
    fitted = np.zeros(len(y))
    for i in range(len(y)):
        fitted[i] = level + trend
        prev = level
        level = alpha * y[i] + (1 - alpha) * (level + trend)
        if use_trend:
            trend = beta * (level - prev) + (1 - beta) * trend
    return fitted, level + np.arange(1, h + 1) * trend


h = 24
results = {
    "simple exp. smoothing": fit(y, m, h, use_trend=False, use_season=False),
    "Holt (level + trend)": fit(y, m, h, use_trend=True, use_season=False),
    "Holt-Winters additive": fit(y, m, h),
}
seasonal_naive = np.tile(y[-m:], 2)
mae = lambda f: np.mean(np.abs(y_test - f))
print(f"train {n} months, forecast {h} months ahead")
print("model                    alpha  beta  gamma   MAE (24 m)")
for name, (sse, (a, b, g), fc) in results.items():
    print(f"{name:<24} {a:>5.2f}  {b:>4.2f}  {g:>5.2f}   {mae(fc):>9.2f}")
print(f"{'seasonal naive':<24} {'':>5}  {'':>4}  {'':>5}   {mae(seasonal_naive):>9.2f}")
print(f"{'noise floor (true mean)':<24} {'':>5}  {'':>4}  {'':>5}   {mae(truth[n:]):>9.2f}")
```

Output (Python 3.14, NumPy 2.5):
```
train 120 months, forecast 24 months ahead
model                    alpha  beta  gamma   MAE (24 m)
simple exp. smoothing     0.95  0.00   0.00       14.04
Holt (level + trend)      0.95  0.05   0.00       10.81
Holt-Winters additive     0.10  0.05   0.20        3.58
seasonal naive                                    14.02
noise floor (true mean)                            3.17
```

The fitted parameters show what each model is missing. Without a seasonal state, SES and Holt choose α = 0.95: the best they can do is chase the seasonal swing by copying the last observation, so their forecast is a flat (or sloped) line from wherever the last month happened to sit. With a seasonal state, Holt-Winters chooses α = 0.10, a slow level, because the season now explains the monthly swings and the level only has to follow the trend. Seasonal naive gets the shape right but is a whole year behind on the trend: about 0.8 × 12 ≈ 10 units, growing over the second forecast year. Holt-Winters is within 0.4 MAE of the true mean curve. The rest is irreducible noise.

## Choosing / trade-offs
- **SES vs Holt vs Holt-Winters.** Match the components to the data: no trend and no season → SES; trend only → Holt (usually with a damped trend for long horizons); trend and season → Holt-Winters. Extra components on a series that lacks them add variance and can extrapolate trends that are not there.
- **Additive vs multiplicative season.** If the seasonal swing grows with the level (retail sales, airline passengers), use multiplicative seasons, or model `log(y)` additively.
- **Damped trend.** Holt's linear trend extrapolates forever. A damping factor φ < 1 (`ℓ + (φ + φ² + … + φ^k) b`) flattens long-horizon forecasts. It is often the most accurate single default in forecasting benchmarks.
- **Grid search vs MLE.** The grid here is for transparency. statsmodels `ExponentialSmoothing` / `ETSModel` and R `forecast::ets` optimize the parameters and initial states by likelihood, pick the model by AIC, and give prediction intervals.
- **vs ARIMA.** ETS models level, trend and season states explicitly. ARIMA models autocorrelation of the differenced series ([[arima-from-scratch]]). Both overlap for linear models. Try both and compare on a rolling-origin backtest.

## Gotchas
- Always compare against seasonal naive and naive baselines. Many "fancy" forecasts lose to "same month last year".
- Evaluate on a time-ordered holdout or rolling origin, never on a random split ([[data-leakage-and-validation-splits]]).
- Initialization matters for short series. A bad initial season leaks error into the first cycles. Skip the warm-up in the loss, or estimate the initial states as parameters.
- γ near 1 lets one unusual month overwrite that seasonal slot. Outliers and holidays move between years, so clean or flag them first.
- The trend β is easy to overfit. A high β plus a long horizon gives explosive forecasts. Prefer a small β or a damped trend.
- The recursion gives point forecasts only. For intervals, use the ETS state-space form or bootstrap the residuals.

## Related
- [[time-series-forecasting]] - forecasting landscape: baselines, ML models and libraries.
- [[arima-from-scratch]] - the other classical linear forecaster.
- [[kalman-filter-from-scratch]] - the state-space machinery behind ETS.
- [[data-leakage-and-validation-splits]] - time-ordered evaluation.
- [[regression-metrics-and-residual-analysis]] - MAE, RMSE and residual checks for forecasts.

## References
- Hyndman and Athanasopoulos, "Forecasting: Principles and Practice" (3rd ed.), Holt-Winters chapter: https://otexts.com/fpp3/holt-winters.html
- Same book, simple exponential smoothing: https://otexts.com/fpp3/ses.html
- statsmodels `ExponentialSmoothing`: https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
