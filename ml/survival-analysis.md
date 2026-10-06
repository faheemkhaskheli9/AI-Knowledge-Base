---
title: Survival analysis (time-to-event)
category: ml
tags: [survival-analysis, time-to-event, censoring, kaplan-meier, cox, churn, predictive-maintenance, lifelines, scikit-survival]
use_cases:
  - "predict when a customer will churn, not just whether"
  - "estimate time until machine failure for maintenance planning"
  - "analyse patient time-to-event outcomes with follow-up that ends early"
  - "estimate loan default or prepayment timing"
  - "compare retention curves between cohorts or experiment arms"
status: draft
last_verified: 2026-10-03
sources:
  - https://lifelines.readthedocs.io/en/latest/
  - https://scikit-survival.readthedocs.io/en/stable/
---

# Survival analysis (time-to-event)

## Summary
Survival analysis models the time until an event (churn, failure, default,
death) when many subjects have not had the event yet. Those rows are
"censored": you know the event had not happened by their last observation.
Dropping them or treating them as "no event" biases results; survival models
use them correctly and output a survival curve S(t) = P(event after t).

## Key concepts
- Right censoring: follow-up ended (still a customer, study ended). Most
  common. Left/interval censoring exist too.
- Survival function S(t), hazard h(t) (instantaneous event rate given
  survival so far), cumulative hazard.
- Kaplan-Meier: non-parametric S(t) estimate; compare groups with the log-rank test.
- Cox proportional hazards: semi-parametric regression; exp(coef) is a
  hazard ratio. Assumes covariate effects are constant over time.
- Parametric models (Weibull, log-normal, log-logistic AFT) when you need
  extrapolation beyond observed follow-up.
- ML models: random survival forests, gradient-boosted Cox (scikit-survival,
  XGBoost `survival:cox` / `survival:aft`), DeepSurv-style nets.
- Metrics: concordance index (C-index, ranking), time-dependent AUC,
  integrated Brier score (calibration).
- Competing risks: another event prevents the one you study (e.g. death
  before relapse); use cumulative incidence, not 1 - KM.

## When to use / scenarios
- SaaS/telecom: churn timing and customer lifetime; pairs with retention offers.
- Manufacturing/IoT: remaining useful life of equipment; plan maintenance windows.
- Healthcare: clinical outcomes with patients lost to follow-up.
- Finance: time to default or prepayment on loans.
- HR: employee attrition timing.
- NOT for: events with fixed horizons and complete labels (plain
  classification is fine), or forecasting an aggregate series (use
  [[time-series-forecasting]]).

## Setup & code
```bash
pip install lifelines
```
```python
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.datasets import load_rossi

df = load_rossi()            # 432 released prisoners, week of re-arrest; arrest=0 means censored
km = KaplanMeierFitter().fit(df["week"], event_observed=df["arrest"])
print("median survival:", km.median_survival_time_)
print("P(no arrest by week 52):", round(float(km.predict(52)), 3))

cph = CoxPHFitter().fit(df, duration_col="week", event_col="arrest")
print(cph.summary[["exp(coef)", "p"]].round(3))
print("concordance:", round(cph.concordance_index_, 3))
```
Output (lifelines 0.30.3): median survival is `inf` (fewer than half were
re-arrested within the 52-week follow-up), S(52) = 0.736, concordance 0.64.
Hazard ratios: financial aid (`fin`) 0.68 (p=0.047), each prior conviction
(`prio`) 1.10 (p=0.001), each year of age 0.94.

For ML models with the scikit-learn API use scikit-survival
(`RandomSurvivalForest`, `GradientBoostingSurvivalAnalysis`), which takes `y`
as a structured array of (event, time).

## Choosing / trade-offs
- Describe and compare groups: Kaplan-Meier + log-rank.
- Explain drivers to stakeholders: Cox PH (hazard ratios), after checking the
  proportional-hazards assumption (`cph.check_assumptions(df)`).
- Best ranking on many features: random survival forest or boosted survival models.
- Need predictions past the observed follow-up: parametric AFT, with caution.
- Discrete time (monthly billing): a per-period classifier on person-period
  rows is simple and works with any tabular model.

## Gotchas
- Treating censored rows as "no event" or dropping them both bias survival upward or downward.
- Leakage through time: features must be known at the start of the risk
  window, not measured later (e.g. "number of support tickets" over the whole life).
- Immortal time bias: a group defined by something that happens later
  (e.g. "users who upgraded") gets guaranteed survival until it happens.
- Median survival can be undefined when fewer than half have the event.
- C-index measures ranking only; check calibration (Brier score) before
  using predicted probabilities for decisions.
- Informative censoring (people leave the study because they are getting
  worse) breaks the standard assumptions.

## Related
- [[model-evaluation-and-metrics]] - classification metrics for fixed-horizon alternatives.
- [[gradient-boosting-tabular]] - XGBoost survival objectives.
- [[time-series-forecasting]] - aggregate forecasts rather than per-subject timing.
- [[anomaly-detection]] - early warning signals for failures.
- [[causal-inference-and-uplift]] - who to target with a retention offer.
- [[healthcare]], [[manufacturing-iot]], [[finance]] - main application areas.
- [[kaplan-meier-survival-from-scratch]] - Kaplan-Meier, Greenwood CI and log-rank test built by hand.

## References
- lifelines: https://lifelines.readthedocs.io/en/latest/
- scikit-survival: https://scikit-survival.readthedocs.io/en/stable/
