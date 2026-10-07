---
title: DBSCAN from scratch (core points, cluster expansion, noise, choosing eps with the k-distance plot)
category: ml
tags: [dbscan, clustering, density-based, outliers, noise, eps, min-samples, k-distance, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement DBSCAN from scratch for learning or an interview"
  - "cluster data with non-convex shapes where k-means fails"
  - "find clusters without choosing the number of clusters, and flag outliers as noise"
  - "pick eps for DBSCAN from a k-distance plot"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/clustering.html#dbscan
  - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
  - https://dl.acm.org/doi/10.5555/3001460.3001507
---

# DBSCAN from scratch (core points, cluster expansion, noise, choosing eps with the k-distance plot)

## Summary
DBSCAN groups points that sit in dense regions and labels points in sparse regions as noise. A point with at least `min_samples` neighbours within radius `eps` is a core point; clusters are the connected components of core points plus the border points they reach. It needs no cluster count, finds arbitrary shapes such as two interleaved moons, and gives outliers for free. The 25-line version below produces exactly scikit-learn's core points, noise set and clusters.

## Key concepts
- **eps-neighbourhood.** All points within distance `eps`, including the point itself (scikit-learn counts it too).
- **Core, border, noise.** Core: neighbourhood size ≥ `min_samples`. Border: not core but inside a core point's neighbourhood. Noise: neither, labelled `-1`.
- **Expansion.** Start a cluster at an unvisited core point and flood-fill: every neighbour joins the cluster, and only core neighbours keep spreading it. It is a graph traversal (BFS/DFS) over the "within eps" graph.
- **No k.** The number of clusters falls out of `eps` and `min_samples`. That is the main win over k-means, and also the main tuning burden ([[k-means-from-scratch]]).
- **k-distance plot.** For each point, the distance to its `(min_samples − 1)`-th nearest other point, sorted. The knee where the curve shoots up is a good `eps`: below it points are in dense regions, above it they are outliers.
- **Border ambiguity.** A border point within `eps` of two clusters goes to whichever cluster reaches it first. Core points and noise are deterministic; border labels can depend on order.

## When to use / scenarios
- Spatial data: GPS pings into stops or hotspots, store or incident locations into areas, with haversine distance.
- Non-convex shapes and unknown cluster count, where k-means splits a curved cluster in half.
- Outlier flagging as a by-product: the noise points ([[anomaly-detection]]).
- Learning: a clean example of density thinking and graph traversal on a distance matrix.
- Not for: clusters of very different densities (one `eps` cannot fit both; use HDBSCAN), high-dimensional raw features where distances concentrate ([[curse-of-dimensionality]]), or very large `n` with a brute-force distance matrix.

## Setup & code
`pip install numpy scikit-learn`. Runs in about a second on CPU.

```python
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


def dbscan(X, eps=0.3, min_samples=5):
    d2 = (X ** 2).sum(1)[:, None] - 2 * X @ X.T + (X ** 2).sum(1)[None]
    nbrs = [np.flatnonzero(row <= eps ** 2) for row in d2]   # includes the point itself
    core = np.array([len(n) >= min_samples for n in nbrs])
    labels = np.full(len(X), -1)                              # -1 = noise
    c = 0
    for i in range(len(X)):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = c
        stack = [i]
        while stack:                                          # expand through core points
            j = stack.pop()
            for n in nbrs[j]:
                if labels[n] == -1:
                    labels[n] = c                             # border or core joins cluster
                    if core[n]:
                        stack.append(n)
        c += 1
    return labels, core


X, _ = make_moons(400, noise=0.08, random_state=0)
rng = np.random.default_rng(0)
X = np.vstack([X, rng.uniform(-1.5, 2.5, (20, 2))])          # 20 outliers
X = StandardScaler().fit_transform(X)

ours, core = dbscan(X, eps=0.3, min_samples=5)
sk = DBSCAN(eps=0.3, min_samples=5).fit(X)
print("clusters:", ours.max() + 1, "noise points:", (ours == -1).sum(), "core points:", core.sum())
print("same core points as sklearn:", np.array_equal(np.sort(sk.core_sample_indices_), np.flatnonzero(core)))
print("ARI vs sklearn labels:", adjusted_rand_score(sk.labels_, ours))
print("same noise set:", np.array_equal(ours == -1, sk.labels_ == -1))

# Choosing eps: distance to the k-th neighbour, sorted; look for the knee.
d = np.sqrt(np.maximum((X ** 2).sum(1)[:, None] - 2 * X @ X.T + (X ** 2).sum(1)[None], 0))
kdist = np.sort(np.sort(d, 1)[:, 4])                          # k = min_samples - 1 (column 0 is self)
print("k-distance percentiles 50/90/95/99:", np.round(np.percentile(kdist, [50, 90, 95, 99]), 3))
for eps in (0.1, 0.2, 0.3, 0.5, 1.0):
    lab, _ = dbscan(X, eps=eps)
    print(f"eps={eps}: clusters={lab.max() + 1}, noise={(lab == -1).sum()}")

km = KMeans(2, n_init=10, random_state=0).fit_predict(X[:400])
y_true = make_moons(400, noise=0.08, random_state=0)[1]
print("ARI on the moons: dbscan", round(adjusted_rand_score(y_true, ours[:400]), 3),
      "k-means", round(adjusted_rand_score(y_true, km), 3))
```

Output (numpy 2.5, scikit-learn 1.9):
```
clusters: 2 noise points: 14 core points: 400
same core points as sklearn: True
ARI vs sklearn labels: 1.0
same noise set: True
k-distance percentiles 50/90/95/99: [0.107 0.193 0.279 1.672]
eps=0.1: clusters=28, noise=137
eps=0.2: clusters=2, noise=19
eps=0.3: clusters=2, noise=14
eps=0.5: clusters=1, noise=12
eps=1.0: clusters=1, noise=8
ARI on the moons: dbscan 1.0 k-means 0.454
```

The from-scratch version finds the same 400 core points, the same 14 noise points and the same two clusters as scikit-learn. The k-distance curve stays below 0.28 for 95% of points and jumps to 1.67 at the 99th percentile, which is where the outliers live, so `eps` between 0.2 and 0.3 is the sensible range. The sweep confirms it: `eps = 0.1` shatters the moons into 28 fragments with a third of points as noise, and `eps ≥ 0.5` merges both moons into one cluster. On the moons themselves DBSCAN recovers the true labels exactly (ARI 1.0), while k-means, which can only draw a straight boundary between two centroids, scores 0.454. Six of the 20 uniform "outliers" landed close enough to a moon to join it; that is correct behaviour, not a bug.

## Choosing / trade-offs
- **DBSCAN vs k-means.** k-means is faster, scales to millions of points and suits round, similar-size clusters. DBSCAN handles shape and noise but needs a meaningful `eps` ([[clustering]]).
- **DBSCAN vs HDBSCAN.** HDBSCAN (`sklearn.cluster.HDBSCAN`) runs DBSCAN over all `eps` values and keeps the stable clusters, so it handles varying densities and drops the `eps` knob. Prefer it unless you need DBSCAN's simplicity or a fixed physical radius (e.g. "within 50 m").
- **min_samples.** Larger values make core status stricter: fewer, denser clusters and more noise. A common start is `2 × n_features`; raise it for noisy data.
- **Brute force vs index.** The full distance matrix is `O(n²)` memory. scikit-learn uses KD/ball trees for low dimensions; for large spatial data use a ball tree with haversine distance on radians.

## Gotchas
- Scale features first. `eps` is one radius across all dimensions, so an unscaled feature in metres next to one in kilometres makes the radius meaningless ([[feature-scaling-and-normalization]]).
- Compare against the squared radius (`d2 <= eps**2`) or take the square root consistently; mixing them silently changes `eps`.
- Clip the `‖a‖² − 2a·b + ‖b‖²` distances at 0 before `sqrt`; rounding can produce tiny negatives ([[knn-from-scratch]]).
- Border points can switch clusters when the data order changes. Do not treat cluster ids or border assignments as stable between runs; compare with ARI, not label equality.
- DBSCAN has no `predict` for new points. Assign a new point to the cluster of its nearest core point within `eps`, else noise, or refit.
- Cluster ids are arbitrary integers; `-1` is the only one with a meaning.

## Related
- [[clustering]] - DBSCAN, HDBSCAN, k-means and others in scikit-learn, and how to pick.
- [[k-means-from-scratch]] - the centroid-based alternative, built the same way.
- [[gaussian-mixture-models-and-em]] - soft, probabilistic clustering.
- [[anomaly-detection]] - noise points as outliers, and dedicated detectors.
- [[distance-metrics-and-similarity]] - choosing the distance, including haversine.
- [[knn-from-scratch]] - the same vectorised distance trick.

## References
- scikit-learn, Clustering (DBSCAN section): https://scikit-learn.org/stable/modules/clustering.html#dbscan
- scikit-learn `DBSCAN` API: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
- Ester, Kriegel, Sander, Xu (1996), "A density-based algorithm for discovering clusters in large spatial databases with noise", KDD: https://dl.acm.org/doi/10.5555/3001460.3001507
