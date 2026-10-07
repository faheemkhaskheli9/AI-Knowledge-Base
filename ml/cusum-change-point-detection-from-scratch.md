---
title: CUSUM change detection from scratch (detecting small mean shifts in a stream)
category: ml
tags: [cusum, change-point-detection, statistical-process-control, drift-detection, monitoring, anomaly-detection, time-series, from-scratch]
use_cases:
  - "detect a small, sustained shift in a metric stream (latency, error rate, model score, sensor reading)"
  - "alert on gradual drift that a fixed 3-sigma threshold misses"
  - "trade detection delay against false-alarm rate with two parameters"
  - "explain CUSUM in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1093/biomet/41.1-2.100
  - https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm
---

# CUSUM change detection from scratch (detecting small mean shifts in a stream)

## Summary
CUSUM (cumulative sum) detects when the mean of a stream has shifted. It adds up standardized deviations from the expected mean, subtracts a small allowance `k` at each step so noise drains away, and alarms when the running sum passes a threshold `h`. A shift too small to stand out in any single point still adds up over time. The algorithm is a few lines with two running sums and no stored history. Below, CUSUM (`k = 0.5`, `h = 5`) and a 3-sigma threshold on single points are tuned to similar false-alarm rates: a median of about 300 vs 280 in-control points before a false alarm. On a 0.5 standard-deviation shift, CUSUM alarms after 37 points on average, while the 3-sigma rule takes 103 points and misses the shift entirely in 82 of 500 runs. On a 1 SD shift the delays are 9 vs 42 points.

## Key concepts
- **Recursion.** With `z_t = (x_t − μ₀) / σ`: `S⁺_t = max(0, S⁺_{t−1} + z_t − k)` and `S⁻_t = max(0, S⁻_{t−1} − z_t − k)`. Alarm when either exceeds `h`. `S⁺` catches upward shifts, `S⁻` downward ones.
- **Allowance `k`.** Each in-control step adds `z − k`, which is negative on average, so the sum keeps resetting to 0. A shift of `δ` SDs adds about `δ − k` per step. Set `k` to half the smallest shift you care about, in SD units (`k = 0.5` targets a 1 SD shift).
- **Threshold `h`.** Sets the trade-off between false alarms and delay. Larger `h` means fewer false alarms and slower detection. `h = 4` or `5` with `k = 0.5` is the textbook setting.
- **Average run length (ARL).** ARL₀ is the expected number of in-control points before a false alarm. ARL₁ is the expected delay after a real shift. Detectors are compared at equal ARL₀, as in the demo.
- **Why it beats a threshold.** A single-point rule only uses the current value. CUSUM is a sequential likelihood-ratio test (Page, 1954) that pools evidence across points, which is what small shifts need.

## When to use / scenarios
- Learning: the simplest sequential change detector, and the base of many drift detectors (Page-Hinkley is a close variant).
- Practice: monitoring ML model inputs and outputs for drift ([[online-learning-and-concept-drift]]), such as mean prediction score, feature means, or the share of a class. Service metrics such as latency or error rate after a deploy. Manufacturing and sensor process control, where it comes from.
- Interviews: "how would you detect a gradual drift in a metric", "CUSUM vs control chart".
- Not for: single large spikes (a threshold or [[anomaly-detection]] handles those, and CUSUM adds delay). Not for streams with strong seasonality or trend unless you run it on residuals from a forecast ([[time-series-forecasting]]). Not for finding several change points in a whole historical series offline, where PELT or binary segmentation (the `ruptures` library) fit better.

## Setup & code
NumPy only. Runs in under a second.

```python
import numpy as np

rng = np.random.default_rng(0)


def cusum(x, mu0, sigma, k=0.5, h=5.0):
    """Two-sided tabular CUSUM on standardized data. Returns the first alarm index or None."""
    hi = lo = 0.0
    for t, v in enumerate(x):
        z = (v - mu0) / sigma
        hi = max(0.0, hi + z - k)          # evidence the mean went up
        lo = max(0.0, lo - z - k)          # evidence the mean went down
        if hi > h or lo > h:
            return t
    return None


def shewhart(x, mu0, sigma, L=3.0):
    """Baseline: alarm on the first single point beyond mu0 +/- L sigma."""
    out = np.flatnonzero(np.abs(x - mu0) > L * sigma)
    return int(out[0]) if out.size else None


mu0, sigma, change_at = 10.0, 1.0, 300
print("shift   detector   mean delay   missed (of 500 runs, 600 points)")
for shift in (0.5, 1.0, 2.0):
    for name, det in (("CUSUM", cusum), ("3-sigma", shewhart)):
        delays, missed = [], 0
        for _ in range(500):
            x = rng.normal(mu0, sigma, 600)
            x[change_at:] += shift * sigma
            a = det(x[change_at:], mu0, sigma)   # start at the change to measure delay
            if a is None:
                missed += 1
            else:
                delays.append(a)
        print(f"{shift:4.1f} sd   {name:8s}   {np.mean(delays):9.1f}   {missed:4d}")

for name, det in (("CUSUM", cusum), ("3-sigma", shewhart)):
    runs = []
    for _ in range(300):
        x = rng.normal(mu0, sigma, 5000)
        a = det(x, mu0, sigma)
        runs.append(5000 if a is None else a + 1)
    print(f"in-control run length before a false alarm, {name}: median {np.median(runs):.0f}")
```

Output (Python 3.14, NumPy 2.5):
```
shift   detector   mean delay   missed (of 500 runs, 600 points)
 0.5 sd   CUSUM           37.1      0
 0.5 sd   3-sigma        102.8     82
 1.0 sd   CUSUM            9.0      0
 1.0 sd   3-sigma         41.7      0
 2.0 sd   CUSUM            3.1      0
 2.0 sd   3-sigma          5.3      0
in-control run length before a false alarm, CUSUM: median 298
in-control run length before a false alarm, 3-sigma: median 278
```

The two detectors raise false alarms at about the same rate, so the comparison is fair. For small shifts CUSUM is 3-5× faster, and the 3-sigma rule misses 16% of 0.5 SD shifts within 300 points (the mean delay for 3-sigma excludes those misses, so it flatters it). For a 2 SD shift the gap mostly closes, because single points already stand out. That is the rule of thumb: CUSUM for small persistent shifts, thresholds for big jumps, and both together in practice.

## Choosing / trade-offs
- **Picking `k` and `h`.** Choose `k` from the smallest shift worth detecting (`k = δ / 2`). Then raise `h` until the false-alarm rate on historical in-control data is acceptable. Simulate as in the demo rather than trusting tables when your data is not Gaussian.
- **CUSUM vs EWMA chart.** An exponentially weighted moving average with a control limit also detects small shifts and is easier to explain to stakeholders. Performance is similar; CUSUM is slightly better for the shift size it was tuned for.
- **CUSUM vs ML drift tests.** Kolmogorov-Smirnov or PSI on windows detect changes in the whole distribution, not just the mean, but need a window and react later. Run CUSUM on the summary statistics you care about most, and distribution tests on a schedule.
- **Known vs estimated baseline.** `μ₀` and `σ` come from a reference period. Estimate them from enough clean data (hundreds of points); a noisy `σ` estimate changes the false-alarm rate a lot.
- **After an alarm.** Reset both sums to 0. The time the sum last left zero estimates when the change started, which is useful for root cause analysis.

## Gotchas
- CUSUM assumes independent observations. Autocorrelated metrics (most time series) produce many false alarms. Run it on residuals from a forecast or AR model instead of raw values.
- Seasonality and trends look like shifts. Remove them first, or compare against a same-hour-last-week baseline.
- Count metrics such as errors per minute are not Gaussian at low counts. Use a Poisson or Bernoulli CUSUM, or aggregate to larger windows first.
- `σ` must be the per-point noise, not the spread of daily averages. Mixing the two scales makes `h` meaningless.
- The two-sided version has roughly half the ARL₀ of a one-sided one with the same `h`, since either side can alarm. If only increases matter (latency, error rate), use one side.
- An alarm marks a change, not its cause. Pair it with breakdowns by segment so the on-call engineer can find what moved.

## Related
- [[online-learning-and-concept-drift]] - drift detection for deployed models, where CUSUM is one tool.
- [[anomaly-detection]] - detecting single unusual points rather than sustained shifts.
- [[statistics-and-ab-testing]] - sequential testing and false-positive control behind ARL.
- [[holt-winters-exponential-smoothing-from-scratch]] - a forecast whose residuals CUSUM can monitor.
- [[kalman-filter-from-scratch]] - another sequential method for noisy streams.
- [[llm-observability]] - monitoring metrics for LLM apps where CUSUM-style alerts apply.

## References
- Page (1954), "Continuous Inspection Schemes", Biometrika: https://doi.org/10.1093/biomet/41.1-2.100
- NIST/SEMATECH e-Handbook of Statistical Methods, CUSUM control charts: https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm
