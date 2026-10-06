---
title: Online learning and concept drift
category: ml
tags: [online-learning, streaming, incremental-learning, concept-drift, data-drift, adwin, river, partial-fit, monitoring]
use_cases:
  - "update a model continuously from a stream of events"
  - "detect when a production model's accuracy or input data has shifted"
  - "train on data that does not fit in memory one batch at a time"
  - "keep a fraud or spam model current as behaviour changes"
status: draft
last_verified: 2026-10-03
sources:
  - https://riverml.xyz/latest/
  - https://scikit-learn.org/stable/computing/scaling_strategies.html
---

# Online learning and concept drift

## Summary
Online (incremental) learning updates a model one example or mini-batch at
a time instead of retraining on the full dataset. Concept drift is when the
relationship between inputs and the target changes over time, so a model
that was accurate decays. The two go together: drift detectors tell you
when to adapt, and online models make adapting cheap.

## Key concepts
- Data (covariate) drift: P(X) changes (new device types, new region).
  Concept drift: P(y | X) changes (fraudsters change tactics). Label drift: P(y) changes.
- Drift shapes: sudden, gradual, incremental, recurring (seasonal).
- Prequential (test-then-train) evaluation: predict each example, score it,
  then learn from it. The honest metric for streams.
- Detectors on the error stream: ADWIN, DDM, Page-Hinkley. On inputs without
  labels: population stability index (PSI), KS test, Jensen-Shannon distance per feature.
- Incremental models: SGD linear models, Naive Bayes, Hoeffding trees,
  adaptive random forests; scikit-learn `partial_fit` on `SGDClassifier`,
  `MultinomialNB`, `MiniBatchKMeans`.
- Adaptation strategies: sliding window retrain, weighting recent data,
  resetting or swapping the model when a detector fires.

## When to use / scenarios
- Fraud, spam, abuse: adversaries adapt within days.
- Ad click-through and recommendation: preferences and inventory change daily.
- IoT/manufacturing: sensors age, machines get recalibrated.
- Any production model: monitor drift even if you retrain in batch.
- NOT needed when data is stable and a weekly/monthly batch retrain keeps
  up; batch training of gradient boosting is usually more accurate than an
  online model. Do not go online just because data arrives as a stream.

## Setup & code
```bash
pip install river
```
```python
import random

from river import drift, linear_model, metrics, preprocessing, utils

# Stream where the relationship flips halfway: concept drift at t=5000
random.seed(0)
def stream(n=10_000):
    for t in range(n):
        x = {"x1": random.gauss(0, 1), "x2": random.gauss(0, 1)}
        score = x["x1"] + 0.5 * x["x2"]
        y = score > 0 if t < 5000 else score < 0
        yield t, x, y

model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
acc, window = metrics.Accuracy(), utils.Rolling(metrics.Accuracy, window_size=500)
detector = drift.ADWIN()

for t, x, y in stream():
    y_pred = model.predict_one(x)      # test-then-train (prequential)
    acc.update(y, y_pred)
    window.update(y, y_pred)
    detector.update(int(y_pred == y))   # feed the error signal
    if detector.drift_detected:
        print(f"drift detected at t={t}, rolling acc={window.get():.3f}")
    model.learn_one(x, y)
    if t in (4999, 5499, 9999):
        print(f"t={t} rolling acc={window.get():.3f}")
print("overall", acc)
```
Output (river 0.26.1): rolling accuracy is 1.000 at t=4999. ADWIN fires
at t=5023, 23 examples after the flip. Accuracy falls to 0.032 at t=5499
and is back to 0.998 by t=9999; overall accuracy is 92.52%. ADWIN also fired
at t=3455 while accuracy was *rising*: it flags change in either direction,
so check which way the metric moved before acting.

The slow recovery is the lesson: plain SGD has to unlearn large weights.
In production, reset or retrain the model on recent data when the detector
fires, or use an adaptive model (`forest.ARFClassifier`), which does this internally.

scikit-learn equivalent for out-of-core batches:
`SGDClassifier().partial_fit(X_batch, y_batch, classes=[0, 1])`.

## Choosing / trade-offs
- Batch retrain on a schedule + drift monitoring: default for most teams;
  simplest to test, version and roll back.
- True online learning: when labels arrive fast and the world changes
  faster than your retrain cycle (ads, fraud).
- Labels delayed (loans default months later): you cannot monitor accuracy
  in time, so monitor input drift and prediction distribution as proxies.
- Monitoring tools: Evidently, NannyML (estimates performance without
  labels), WhyLabs, or cloud model monitors.

## Gotchas
- Feedback loops: the model's own decisions shape future labels (blocked
  transactions never get a fraud label).
- Online models can be poisoned by bursts of bad or adversarial data; cap
  learning rate, validate updates, keep a fallback model.
- Drift on a feature that does not matter is noise; weight alerts by feature importance.
- Many per-feature tests on large samples will flag "significant" but tiny
  shifts; alert on effect size (PSI > 0.2 is a common rule of thumb), not p-values.
- Seasonality looks like drift; compare to the same period last cycle.
- Keep the scaler/encoder online too, or the pipeline silently uses stale statistics.

## Related
- [[mlops-lifecycle]] - monitoring and retraining in production.
- [[llm-observability]] - drift and quality monitoring for LLM apps.
- [[anomaly-detection]] - detecting unusual points rather than shifts.
- [[time-series-forecasting]] - forecasting series that also drift.
- [[finance]], [[security-defensive]] - adversarial drift.
- [[cusum-change-point-detection-from-scratch]] - CUSUM built by hand for detecting small mean shifts in a stream.

## References
- River: https://riverml.xyz/latest/
- scikit-learn incremental learning: https://scikit-learn.org/stable/computing/scaling_strategies.html
- Bifet & Gavalda, Learning from Time-Changing Data with Adaptive Windowing (ADWIN, SDM 2007); Gama et al., A Survey on Concept Drift Adaptation (ACM CSUR 2014).
