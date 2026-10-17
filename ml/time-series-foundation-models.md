---
title: Time-series foundation models (Chronos, TimesFM, Moirai) vs classical baselines
category: ml
tags: [time-series, forecasting, foundation-model, zero-shot, chronos, chronos-2, timesfm, moirai, uni2ts, backtesting, mase, seasonal-naive, ets, statsmodels]
use_cases:
  - "get a zero-shot forecast for many series with little history and no per-series model fitting"
  - "decide whether Chronos, TimesFM or Moirai beats seasonal naive and ETS on my own data before adopting it"
  - "set up a rolling-origin backtest with MASE that any forecaster, pretrained or classical, plugs into"
  - "check whether a time-series foundation model's weight licence allows commercial production use"
status: draft
last_verified: 2026-10-17
sources:
  - https://github.com/amazon-science/chronos-forecasting
  - https://arxiv.org/abs/2403.07815
  - https://arxiv.org/abs/2510.15821
  - https://github.com/google-research/timesfm
  - https://arxiv.org/abs/2310.10688
  - https://github.com/SalesforceAIResearch/uni2ts
  - https://arxiv.org/abs/2402.02592
  - https://github.com/SalesforceAIResearch/gift-eval
  - https://github.com/autogluon/fev
  - https://otexts.com/fpp3/
---

# Time-series foundation models (Chronos, TimesFM, Moirai) vs classical baselines

## Summary
A time-series foundation model (TSFM) is a transformer pretrained on a large
corpus of real and synthetic series that forecasts a new series zero-shot:
you pass the recent history (and, in newer models, covariates) and get point
and quantile forecasts with no fitting on your data. The main open families
are Amazon's Chronos (Chronos, Chronos-Bolt, Chronos-2), Google's TimesFM
(1.0 to 3.0) and Salesforce's Moirai (1.0, 1.1, Moirai-MoE, 2.0). They are a
strong default when you have many series, short histories or no time to tune,
but they must still beat seasonal naive and ETS in a rolling-origin backtest on
your own data, and weight licences differ by version.

## Key concepts
- **Zero-shot forecasting.** No per-series training: the context window is the
  input. Fine-tuning is optional (TimesFM ships a LoRA example; uni2ts has a
  fine-tuning module) and only worth it if the backtest says so.
- **How they tokenise series.** Original Chronos scales and quantises values
  into tokens and trains a T5-style language model with cross-entropy, then
  samples future trajectories. Chronos-Bolt, TimesFM and Moirai instead feed
  patches of observations; Chronos-Bolt predicts multi-step quantiles directly
  (faster, no sampling).
- **Probabilistic output.** Most return quantiles (e.g. 0.1/0.5/0.9). Score
  them with pinball loss / weighted quantile loss, not only point error.
- **Context length.** Each model has a maximum context; longer history is
  truncated to the most recent points. TimesFM 2.5 supports up to 16k context
  (up from 2,048 in 2.0); TimesFM 3.0 truncates beyond 15,360 points.
- **Covariates and multivariate.** Chronos-2 and TimesFM 3.0 support
  multivariate series and covariates (past-only and known-future) natively;
  TimesFM 2.5 does covariates through its XReg add-on. Original Chronos and
  Chronos-Bolt are univariate; Moirai was designed for any number of variates.
- **Baselines that matter.** Seasonal naive (repeat last season) and ETS /
  ARIMA / Theta. A TSFM that does not beat them on your backtest is not worth
  its serving cost.
- **MASE.** Mean absolute error divided by the in-sample MAE of the seasonal
  naive forecast. Scale-free, so it averages across series; < 1 means better
  than in-sample seasonal naive. Aggregate relative scores across series with a
  geometric mean, not an arithmetic one.
- **Rolling-origin backtest.** Several forecast origins, each model sees only
  data before its origin, scored on the next `h` points. One split is noise.
- **Benchmarks.** GIFT-Eval (Salesforce) and fev-bench (AutoGluon `fev`) are the
  shared leaderboards most model releases report on. GIFT-Eval tracks test-data
  leakage and tags truly zero-shot submissions separately.

### Model families (as stated in each project's README, 2026-10-17)

| Family | Versions / checkpoints | Package | Licence |
|---|---|---|---|
| Chronos (Amazon) | `amazon/chronos-2` (120M), `autogluon/chronos-2-small` (28M); Chronos-Bolt tiny/mini/small/base (9M–205M); original `chronos-t5-*` (8M–710M) | `pip install chronos-forecasting` | repo: Apache-2.0; check each model card for the weights |
| TimesFM (Google) | 3.0 `google/timesfm-3.0-pytorch` (latest); 2.5 (200M); 1.0/2.0 archived (`pip install timesfm==1.3.0`) | `pip install timesfm[torch]` | code and weights up to 2.5: Apache-2.0; 3.0 weights: `timesfm-non-commercial-license-v1.0`, non-commercial, non-production only (commercial use via Google Cloud, e.g. BigQuery ML) |
| Moirai (Salesforce) | `Salesforce/moirai-2.0-R-small`; Moirai-MoE 1.0-R small/base; Moirai 1.1-R small/base/large | `pip install uni2ts` | code: Apache-2.0; README says the release is "for research purposes only"; check each model card for the weights |

Other open TSFMs exist (for example Lag-Llama, MOMENT, TimeGPT as an API); the
leaderboards above are the quickest way to see what is current.

## When to use / scenarios
- Retail/CPG with thousands of SKUs, many with short or new histories (cold
  start): a TSFM gives a usable forecast on day one. See [[ecommerce-retail]],
  [[logistics-and-supply-chain]].
- Ops/IT capacity, energy load, sensor series where you need a decent forecast
  across many heterogeneous series without a modelling team per series. See
  [[manufacturing-iot]].
- A quick, strong baseline to beat before investing in global gradient
  boosting with features ([[gradient-boosting-tabular]]).
- Forecasts with known future covariates (prices, promotions, holidays) and
  related series: Chronos-2 or TimesFM 3.0 rather than older univariate models.
- Not when: a few long, stable series where ETS/ARIMA already score well (cheaper,
  interpretable, see [[holt-winters-exponential-smoothing-from-scratch]],
  [[arima-from-scratch]]); hard latency or CPU-only budgets at huge scale where
  statistical models are orders of magnitude cheaper; rich tabular features
  that drive the target (global gradient boosting); or when the licence of the
  best-scoring checkpoint forbids your use.

## Setup & code
`pip install numpy statsmodels`. A rolling-origin backtest with MASE that any
forecaster with the signature `f(history) -> h-step forecast` plugs into. Here
it compares seasonal naive and ETS on 12 synthetic daily series with weekly
seasonality (some with a late level shift); about 10 s on CPU.

```python
import warnings

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")      # statsmodels convergence chatter
rng = np.random.default_rng(0)
M, H = 7, 14               # weekly seasonality on daily data, 14-day horizon

# 12 synthetic daily series: level + trend + weekly pattern + noise (+ a level shift in some)
def make_series(n=420):
    t = np.arange(n)
    level = rng.uniform(50, 200)
    weekly = rng.normal(0, 0.15, M) * level
    y = level + rng.normal(0, 0.05) * t + weekly[t % M] + rng.normal(0, 0.05 * level, n)
    if rng.random() < 0.3:                       # structural break late in the series
        y[int(n * 0.8):] += 0.3 * level
    return np.maximum(y, 0)
series = [make_series() for _ in range(12)]

# forecasters: context (1-D array) -> H-step point forecast
def seasonal_naive(y, h=H):
    return np.resize(y[-M:], h)                  # repeat the last full season

def ets(y, h=H):                                 # additive trend (damped) + additive season
    fit = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add",
                               seasonal_periods=M).fit()
    return fit.forecast(h)

def mase(y_train, y_true, y_pred, m=M):
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))   # in-sample seasonal-naive MAE
    return np.mean(np.abs(y_true - y_pred)) / scale

# rolling-origin backtest: several cut-offs, model sees only data before each one
def backtest(forecaster, series, n_origins=6, step=H):
    scores = np.zeros((len(series), n_origins))
    for i, y in enumerate(series):
        for j in range(n_origins):
            cut = len(y) - H - (n_origins - 1 - j) * step
            train, test = y[:cut], y[cut:cut + H]
            scores[i, j] = mase(train, test, forecaster(train))
    return scores                                # shape: series x origins

# foundation models plug in with the same signature: FORECASTERS["chronos2"] = chronos2
FORECASTERS = {"seasonal_naive": seasonal_naive, "ets": ets}
results = {name: backtest(f, series) for name, f in FORECASTERS.items()}

base = results["seasonal_naive"].mean(1)        # per-series mean MASE of the baseline
print(f"{'model':16s} {'mean MASE':>9s} {'geo-mean rel. to SN':>20s} {'series beating SN':>18s}")
for name, s in results.items():
    per_series = s.mean(1)
    rel = np.exp(np.mean(np.log(per_series / base)))   # < 1 means better than seasonal naive
    wins = int(np.sum(per_series < base))
    print(f"{name:16s} {s.mean():9.3f} {rel:20.3f} {wins:>12d} / {len(series)}")
assert results["ets"].mean() < results["seasonal_naive"].mean()
```

Output: seasonal naive mean MASE 1.227, ETS 1.025; ETS is 0.807x seasonal
naive (geometric mean) and beats it on 12 of 12 series. A foundation model is
worth adopting only if it beats the better of these two by a margin that
covers its serving cost, on your series and horizons.

Adapter for Chronos-2 (the `Chronos2Pipeline.from_pretrained` / `predict_df`
calls and the `predictions` output column follow the project README; add it to
`FORECASTERS` above):

<!-- skip-check: needs torch, chronos-forecasting and a model download -->
```python
import pandas as pd
from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

def chronos2(y, h=H):
    ctx = pd.DataFrame({"id": "s", "timestamp": pd.date_range("2020-01-01", periods=len(y), freq="D"),
                        "target": y})
    pred = pipeline.predict_df(ctx, prediction_length=h, quantile_levels=[0.1, 0.5, 0.9],
                               id_column="id", timestamp_column="timestamp", target="target")
    return pred["predictions"].to_numpy()

FORECASTERS["chronos2"] = chronos2
```

For real backtests batch all series per origin into one call (every TSFM
accepts a batch), and score the quantiles too. TimesFM 3.0 (`TimesFM3Evaluator`,
`predict_batch`) and Moirai (`uni2ts`, GluonTS-style predictors) need their own
thin adapters with the same `f(history) -> forecast` shape; copy the call from
each README, as their APIs change between major versions. `statsforecast`,
`fev` and AutoGluon-TimeSeries provide the same backtesting loop at scale.

## Choosing / trade-offs
- **TSFM vs ETS/ARIMA.** TSFM: no per-series fitting, handles short histories
  and many series, quantiles out of the box; costs a GPU or slow CPU inference
  and is a black box. ETS/ARIMA: cheap, interpretable, strong on long stable
  seasonal series.
- **TSFM vs global gradient boosting.** Boosting with lag, calendar and
  promotion features wins when you have rich covariates and enough history to
  train; TSFMs win on cold start and setup time. Newer covariate-aware TSFMs
  narrow the gap; check on your backtest.
- **Which family.** Licence first (TimesFM 3.0 weights are non-commercial for
  self-hosting; Moirai is released for research; check every card), then
  covariate/multivariate support, then size and latency (Chronos-Bolt and
  Chronos-2-small are small; TimesFM 2.5 is 200M parameters), then the
  leaderboard score on domains like yours.
- **Self-host vs managed.** TimesFM is offered in BigQuery ML; Chronos-2 can be
  deployed through SageMaker JumpStart or AutoGluon-Cloud. Managed removes
  serving work and, for TimesFM 3.0, is the licensed commercial route. See
  [[managed-cloud-ai-platforms]].
- **Zero-shot vs fine-tuned.** Fine-tune only when the zero-shot backtest loses
  to the best baseline on a large, homogeneous set of series.

## Gotchas
- Leaderboard ranks do not transfer to your data; public benchmark datasets may
  also be in a model's pretraining corpus (GIFT-Eval tracks this as test-data
  leakage). Only your own rolling-origin backtest counts.
- Comparing on one origin, or with MAPE on series near zero, gives noise or
  explodes; use several origins and MASE/WAPE, plus a quantile loss for
  intervals.
- Context truncation: passing years of hourly data silently drops all but the
  most recent points the model accepts.
- Frequency and seasonality are inferred from the context; a context shorter
  than two seasonal cycles gives the model nothing to infer the season from.
- Missing timestamps and irregular sampling must be regularised before the call,
  as with any forecaster ([[time-series-forecasting]]).
- Model versions change APIs and licences (TimesFM 2.5 to 3.0 changed both).
  Pin the package and the checkpoint, and record both with backtest results.
- Scoring the median quantile against a point-forecast baseline is fair; scoring
  a sampled trajectory or the mean of samples is not the same thing.
- Leakage in the backtest itself: covariates must be values known at the
  origin (plans), not realised values ([[data-leakage-and-validation-splits]]).

## Related
- [[time-series-forecasting]] - the overview: classical, global ML and
  foundation-model routes, metrics and backtesting.
- [[holt-winters-exponential-smoothing-from-scratch]] - the ETS baseline built
  by hand.
- [[arima-from-scratch]] - the other classical baseline.
- [[tabular-foundation-models]] - the same zero-shot, in-context idea for
  tabular data.
- [[gradient-boosting-tabular]] - global lag-feature models, the main
  alternative when covariates are rich.
- [[conformal-prediction-and-uncertainty]] - calibrating the intervals a
  forecaster returns.
- [[model-licenses]] - reading weight licences before production use.
- [[logistics-and-supply-chain]] - demand forecasting in practice.
- [[ecommerce-retail]] - SKU-level forecasting at scale.

## References
- Ansari et al., Chronos: Learning the Language of Time Series: https://arxiv.org/abs/2403.07815
- Chronos-2 technical report: https://arxiv.org/abs/2510.15821
- Chronos repository (models, licence, usage): https://github.com/amazon-science/chronos-forecasting
- Das et al., A decoder-only foundation model for time-series forecasting (TimesFM): https://arxiv.org/abs/2310.10688
- TimesFM repository (versions, licence notice for 3.0 weights): https://github.com/google-research/timesfm
- Woo et al., Unified Training of Universal Time Series Forecasting Transformers (Moirai): https://arxiv.org/abs/2402.02592
- uni2ts repository (Moirai versions): https://github.com/SalesforceAIResearch/uni2ts
- GIFT-Eval benchmark: https://github.com/SalesforceAIResearch/gift-eval
- fev / fev-bench: https://github.com/autogluon/fev
- Hyndman and Athanasopoulos, Forecasting: Principles and Practice (MASE, time-series cross-validation): https://otexts.com/fpp3/
