---
title: Dynamic time warping (DTW) from scratch
category: ml
tags: [dtw, dynamic-time-warping, time-series, alignment, distance, dynamic-programming, classification, knn, from-scratch]
use_cases:
  - "compare time series that have the same shape but different speed or phase"
  - "classify sensor, gesture or ECG sequences with a nearest-neighbour baseline"
  - "align two recordings of the same event point by point"
  - "explain DTW and the Sakoe-Chiba band in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1109/TASSP.1978.1163055
  - https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html
  - https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
---

# Dynamic time warping (DTW) from scratch

## Summary
Dynamic time warping measures the distance between two sequences after aligning them in time. Euclidean distance compares point `i` with point `i`, so two signals with the same shape but a small shift or speed change look far apart. DTW instead finds the cheapest monotonic alignment between the two, allowing one point to match several, by dynamic programming over an `n × m` cost table. This file implements DTW with backtracking and a Sakoe-Chiba window in NumPy. On a warped sine wave, the Euclidean distance is `3.46` and DTW is `0.65`. On a sine-vs-square classification task with random time warps, 1-nearest-neighbour accuracy rises from `0.87` with Euclidean distance to `1.00` with DTW.

## Key concepts
- **Recurrence.** `D[i, j] = cost(a_i, b_j) + min(D[i−1, j], D[i, j−1], D[i−1, j−1])`, with `D[0, 0] = 0` and the rest of the border at `∞`. The answer is `D[n, m]`. Cost is `O(n·m)` time and memory.
- **Warping path.** Backtracking from `(n, m)` to `(1, 1)` gives the alignment: a monotonic, continuous sequence of index pairs. Its length is between `max(n, m)` and `n + m − 1`.
- **Sakoe-Chiba band.** Restrict `|i − j| ≤ w`. It cuts the cost to `O(n·w)` and stops pathological alignments, where one point maps to half of the other series.
- **Not a metric.** DTW breaks the triangle inequality, so metric-tree indexes do not apply. Lower bounds (LB_Keogh) are used for fast search instead.
- **Sequences can differ in length.** Unlike Euclidean distance, DTW compares series of length `n` and `m` directly.

## When to use / scenarios
- Time-series classification baselines: 1-NN with DTW is a strong, hard-to-beat baseline on the UCR archive of benchmark datasets.
- Gesture and activity recognition from accelerometers, handwriting strokes, ECG beats, spoken-word templates: same pattern, different speed.
- Aligning two recordings (a reference and a test run of a machine cycle) to compare them step by step, or averaging aligned series (DBA).
- Clustering sequences with k-medoids or hierarchical clustering on a DTW distance matrix.
- Not for: forecasting (see [[time-series-forecasting]]), very long series without a window (quadratic cost), or cases where the timing itself is the signal, such as a delayed response that should count as different. Deep models (1D CNN, InceptionTime, ROCKET features) usually beat DTW when there is plenty of labelled data.

## Setup & code
NumPy only. The inner loop is pure Python for clarity, so the classification run takes about 10 seconds.

```python
import numpy as np

rng = np.random.default_rng(0)


def dtw(a, b, window=None):
    """Classic O(n*m) DTW with optional Sakoe-Chiba band. Returns cost and path."""
    n, m = len(a), len(b)
    w = max(window, abs(n - m)) if window is not None else max(n, m)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    path, i, j = [], n, m                      # backtrack the optimal alignment
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return np.sqrt(D[n, m]), path[::-1]


def make(kind, shift, n=60):
    t = np.linspace(0, 1, n)
    warped = np.clip(t + shift * np.sin(np.pi * t), 0, 1)   # nonlinear time warp
    y = np.sin(2 * np.pi * warped) if kind == 0 else np.sign(np.sin(2 * np.pi * warped))
    return y + rng.normal(0, 0.1, n)


a, b = make(0, 0.0), make(0, 0.15)
cost, path = dtw(a, b)
print(f"same shape, warped:  euclid {np.linalg.norm(a - b):.2f}   dtw {cost:.2f}   path len {len(path)}")
c = make(1, 0.0)
print(f"different shape:     euclid {np.linalg.norm(a - c):.2f}   dtw {dtw(a, c)[0]:.2f}")

# 1-NN classification: sine vs square waves with random warps.
train = [(make(k, s), k) for k in (0, 1) for s in rng.uniform(-0.3, 0.3, 5)]
test = [(make(k, s), k) for k in (0, 1) for s in rng.uniform(-0.3, 0.3, 50)]
for name, dist in [("euclid", lambda u, v: np.linalg.norm(u - v)),
                   ("dtw", lambda u, v: dtw(u, v)[0]),
                   ("dtw w=6", lambda u, v: dtw(u, v, window=6)[0])]:
    acc = np.mean([min(train, key=lambda tr: dist(x, tr[0]))[1] == y for x, y in test])
    print(f"1-NN {name:8s} accuracy {acc:.2f}")
```

Output (Python 3.14, NumPy 2.5):
```
same shape, warped:  euclid 3.46   dtw 0.65   path len 81
different shape:     euclid 3.81   dtw 3.24
1-NN euclid   accuracy 0.87
1-NN dtw      accuracy 1.00
1-NN dtw w=6  accuracy 1.00
```

For the warped sine, Euclidean distance (`3.46`) is nearly as large as the distance to a different shape (`3.81`), so a nearest-neighbour classifier confuses them. DTW separates them clearly (`0.65` vs `3.24`). The path has 81 steps for two 60-point series: 21 places where one point was matched to several. With 5 training examples per class, Euclidean 1-NN gets 87% and DTW gets every test series right. A 10% window (`w = 6`) keeps the same accuracy while filling only about a fifth of the table.

## Choosing / trade-offs
- **Window size.** 5-20% of the series length is typical. Tune it by cross-validation on the training set, as the UCR benchmark results do. Too small acts like Euclidean distance, too large allows unrealistic warps and costs more.
- **Normalisation.** Z-normalise each series (subtract its mean, divide by its standard deviation) before DTW unless the absolute level matters. Otherwise an offset dominates the distance.
- **Speed.** Use a compiled implementation: `tslearn.metrics.dtw`, `dtaidistance` (C), or `sktime`. For nearest-neighbour search over many series, prune with LB_Keogh lower bounds and early abandoning.
- **Multivariate series.** Use the vector distance between time steps (dependent DTW) or sum per-channel DTW (independent DTW). Which is better depends on whether the channels move together.
- **Differentiable alternatives.** Soft-DTW replaces `min` with a soft minimum so it can serve as a loss for neural networks.
- **DTW vs learned models.** DTW + 1-NN needs no training and works with a handful of labels. With thousands of labelled series, ROCKET or a 1D CNN is usually more accurate and faster at inference.

## Gotchas
- Without a window, DTW can map a single point of one series to most of the other, producing a small distance between quite different signals. Always consider a band.
- DTW is not a metric. Do not plug it into algorithms that assume the triangle inequality (some k-means variants, ball trees).
- Plain k-means with DTW distance has no well-defined mean. Use k-medoids or DBA averaging.
- The per-step cost choice (squared vs absolute difference) changes results. Libraries differ, so match it when comparing against published numbers.
- Pure-Python DTW (as here) is far too slow for real use. Use a compiled library for anything beyond a demo.
- Endpoints are forced to align. If series contain extra lead-in or trailing segments, trim them or use subsequence DTW.

## Related
- [[knn-from-scratch]] - the nearest-neighbour classifier DTW is usually paired with.
- [[distance-metrics-and-similarity]] - Euclidean, cosine and other distances to compare against.
- [[time-series-forecasting]] - the other main time-series task, which DTW does not address.
- [[hmm-from-scratch]] - Viterbi is the same kind of dynamic-programming alignment, with probabilities instead of costs.
- [[ctc-loss-from-scratch-numpy]] - another monotonic alignment computed by dynamic programming.
- [[hierarchical-clustering-from-scratch]] - clusters sequences from a precomputed DTW distance matrix.

## References
- Sakoe and Chiba (1978), "Dynamic programming algorithm optimization for spoken word recognition", IEEE TASSP: https://doi.org/10.1109/TASSP.1978.1163055
- tslearn DTW user guide: https://tslearn.readthedocs.io/en/stable/user_guide/dtw.html
- UCR Time Series Classification Archive: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
