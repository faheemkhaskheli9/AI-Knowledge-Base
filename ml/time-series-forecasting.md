---
title: Time-series forecasting
category: ml
tags: [forecasting, time-series, statsforecast, prophet, arima, backtesting, lightgbm, foundation-models]
use_cases:
  - "forecast weekly product demand for a retail chain"
  - "predict electricity load or energy consumption for a utility"
  - "forecast cash flow or revenue for the next 12 months"
  - "predict server traffic to plan capacity"
  - "build a forecast with prediction intervals and proper backtesting"
status: draft
last_verified: 2026-10-03
sources:
  - https://nixtlaverse.nixtla.io/statsforecast/index.html
  - https://otexts.com/fpp3/
  - https://facebook.github.io/prophet/
---

# Time-series forecasting

## Summary
Forecasting predicts future values of a series (sales, load, traffic, cash) from its past and known future inputs (calendar, price, promotions). Strong baselines are statistical models (ETS, ARIMA, seasonal naive); at scale, global ML models (gradient boosting on lag features) or pretrained time-series foundation models are common. Evaluation discipline (time-ordered backtesting) matters more than model choice.

## Key concepts
- Components: trend, seasonality (possibly multiple), holidays/events, noise.
- Horizon and frequency: forecast h steps ahead; choose the metric per horizon.
- Local vs global models: one model per series (ARIMA/ETS) vs one model across many series (LightGBM, deep nets, foundation models).
- Exogenous variables: those known in the future (price plan, holidays) are usable; unknown ones are not.
- Backtesting = rolling-origin cross-validation (`TimeSeriesSplit`, `cross_validation` in Nixtla libs). Never shuffle.
- Metrics: MAE, RMSE, MASE (scale-free vs naive), sMAPE/WAPE; pinball loss for quantiles. Always compare against a seasonal-naive baseline.
- Probabilistic output: prediction intervals / quantiles, essential for inventory and capacity decisions.
- Intermittent demand (many zeros): Croston/TSB or count-aware models.

## When to use / scenarios
- Retail/CPG: SKU-store demand with promotions, thousands of series -> global LightGBM or statsforecast at scale.
- Energy/utilities: load with strong daily and weekly seasonality, weather as exogenous.
- Finance/FP&A: revenue and cash forecasting; few series, interpretable models, intervals.
- IT ops: capacity planning from traffic metrics.
- NOT for: anomaly flagging (see [[anomaly-detection]]) or when the series is too short/changes regime constantly; then use scenario planning, not a model.

## Setup & code
```bash
pip install statsforecast pandas
```
```python
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive

# long format: unique_id, ds (datetime), y
df = pd.DataFrame({
    "unique_id": "store1",
    "ds": pd.date_range("2022-01-01", periods=104, freq="W"),
    "y": [100 + (i % 52) * 0.5 + (i % 4) * 3 for i in range(104)],
})
sf = StatsForecast(models=[SeasonalNaive(season_length=52), AutoETS(season_length=52)],
                   freq="W")
fcst = sf.forecast(df=df, h=12, level=[80])   # point + 80% interval
print(fcst.head())
```
Global ML route: build lag/rolling/calendar features and train LightGBM (see [[gradient-boosting-tabular]]), or use `mlforecast`. Prophet (`pip install prophet`) suits business series with holidays and missing data, but is not always more accurate than ETS.

## Choosing / trade-offs
- Few series, interpretability needed: ETS/ARIMA/Theta (statsforecast) or Prophet.
- Many related series, rich covariates: global gradient boosting or deep models (N-HiTS, PatchTST via neuralforecast).
- Zero-shot start with little history: pretrained foundation models (e.g. Chronos, TimesFM, Moirai) can be a strong baseline; check current model cards and licenses, and validate against seasonal naive on your own backtest.
- LLM prompting for numeric forecasting is generally a poor fit versus the above.
- Hierarchies (SKU -> category -> total) need reconciliation so forecasts add up.

## Gotchas
- Random train/test splits leak the future; always split by time and backtest with several origins.
- Features computed with future info (centered rolling windows, global scaling on all data).
- Forgetting that exogenous features must be known at forecast time; use forecasts or plans of them.
- Tree models cannot extrapolate trend; detrend or model differences.
- Metrics like MAPE explode near zero values; use WAPE/MASE.
- Structural breaks (COVID, price changes, stockouts: censored demand) make history misleading; mark or drop them.
- Irregular timestamps and missing periods must be regularised before modelling.

## Related
- [[gradient-boosting-tabular]] - global lag-feature models.
- [[anomaly-detection]] - residual-based detection on forecasts.
- [[experiment-tracking]] - record backtest results.
- [[pytorch-basics]] - for deep forecasting models.

## References
- https://otexts.com/fpp3/ (Hyndman, Forecasting: Principles and Practice)
- https://nixtlaverse.nixtla.io/statsforecast/index.html
- https://facebook.github.io/prophet/
