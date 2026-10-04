---
title: Mean-shift clustering from scratch (kernel density modes, bandwidth, outliers)
category: ml
tags: [mean-shift, clustering, kernel-density-estimation, mode-seeking, bandwidth, gaussian-kernel, unsupervised-learning, numpy, from-scratch, ml-basics]
use_cases:
  - "implement mean-shift clustering with a Gaussian kernel in NumPy"
  - "cluster data without choosing the number of clusters in advance"
  - "see how the bandwidth controls the number of clusters and how outliers end up in tiny clusters"
  - "explain mean shift as gradient ascent on a kernel density estimate in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1109/34.1000236
  - https://doi.org/10.1109/34.400568
  - https://scikit-learn.org/stable/modules/clustering.html#mean-shift
---

# Mean-shift clustering from scratch (kernel density modes, bandwidth, outliers)

## Summary
Mean shift finds clusters as the peaks (modes) of a kernel density estimate. Every point repeatedly moves to the kernel-weighted mean of the data around it, which is a step uphill on the density, until it stops at a mode. Points that reach the same mode form one cluster. The number of clusters is not an input: it follows from the bandwidth, the one parameter that sets how smooth the density is. Below, three blobs plus 30 uniform outliers are clustered with a rule-of-thumb bandwidth of 0.99. Mean shift recovers the three blob centres to within 0.11, and the 8 points left in tiny clusters are all planted outliers. A bandwidth of 0.3 fragments the blobs into 33 clusters, and 4.0 merges everything into one.

## Key concepts
- **Kernel density estimate.** `p̂(x) ∝ Σᵢ K(‖x − xᵢ‖ / h)`. With a Gaussian kernel, `h` is the standard deviation of the bump placed on each point.
- **Mean-shift step.** `m(x) = Σᵢ K(x, xᵢ) xᵢ / Σᵢ K(x, xᵢ)`. The vector `m(x) − x` points along the gradient of the density estimate, so iterating `x ← m(x)` climbs to a local maximum. It is an adaptive step: large in sparse regions, small near a peak.
- **Basins of attraction.** Every starting point flows to one mode. A cluster is the set of points that share a mode, so clusters can have any shape a density basin can have.
- **Bandwidth.** The only real parameter. Small `h` gives a bumpy density with many modes. Large `h` smooths peaks together. Rules of thumb (Scott, Silverman) or `sklearn.cluster.estimate_bandwidth` give a starting value.
- **Mode merging.** Iteration stops within a tolerance, so points that reach the same peak end up a little apart. Merge modes closer than a fraction of the bandwidth.

## When to use / scenarios
- Learning: shows clustering as density estimation, and why "number of clusters" is really a smoothing choice.
- Interviews: "cluster without knowing k", "mean shift vs k-means vs DBSCAN", "what does the bandwidth do", "derive the mean-shift vector".
- Practice: image segmentation in joint colour and position space, colour quantisation, finding hotspots in location data (pickups, incidents), finding the dominant modes of a distribution, and the classic mean-shift / CamShift object tracker in video.
- Not for: large datasets (the naive version is O(n²) per iteration; use k-means or mini-batch k-means); high-dimensional data, where kernel density estimates become meaningless (reduce dimensions first); or clusters of very different density, where one bandwidth cannot fit all (try HDBSCAN).

## Setup & code
NumPy only. Runs in about a second.

```python
import numpy as np

rng = np.random.default_rng(0)

# 3 blobs of different sizes and spreads, plus 30 uniform outliers
centers = np.array([[0, 0], [6, 1], [2, 6]])
X = np.vstack([rng.normal(centers[0], 0.6, (200, 2)),
               rng.normal(centers[1], 1.0, (120, 2)),
               rng.normal(centers[2], 0.8, (80, 2)),
               rng.uniform(-4, 10, (30, 2))])
truth = np.repeat([0, 1, 2, -1], [200, 120, 80, 30])


def mean_shift(X, bandwidth, iters=300, tol=1e-5):
    """Move every point uphill on a Gaussian kernel density estimate until it stops, then merge the modes."""
    modes = X.copy()
    for it in range(iters):
        d2 = ((modes[:, None, :] - X[None, :, :]) ** 2).sum(-1)     # (n_modes, n_points)
        w = np.exp(-0.5 * d2 / bandwidth**2)
        new = w @ X / w.sum(1, keepdims=True)                      # weighted mean of the neighbourhood
        shift = np.abs(new - modes).max()
        modes = new
        if shift < tol:
            break
    # points whose modes are within bandwidth/2 belong to the same cluster
    centers, labels = [], np.empty(len(X), int)
    for i, m in enumerate(modes):
        for j, c in enumerate(centers):
            if np.linalg.norm(m - c) < bandwidth / 2:
                labels[i] = j
                break
        else:
            centers.append(m)
            labels[i] = len(centers) - 1
    return np.array(centers), labels, it + 1


def silverman_bandwidth(X):
    """Rule-of-thumb bandwidth for a d-dim Gaussian KDE (Silverman/Scott-style)."""
    n, d = X.shape
    return X.std(0).mean() * (4 / ((d + 2) * n)) ** (1 / (d + 4))


def purity(labels, truth):
    inl = truth >= 0
    return sum(np.bincount(truth[inl][labels[inl] == c]).max() for c in np.unique(labels[inl])) / inl.sum()


h0 = silverman_bandwidth(X)
print(f"n={len(X)}, rule-of-thumb bandwidth {h0:.2f}")
print("\nbandwidth  clusters  big clusters (>=20 pts)  purity  iters")
for h in [0.3, h0, 2.0, 4.0]:
    C, lab, its = mean_shift(X, h)
    sizes = np.bincount(lab)
    print(f"{h:>9.2f}  {len(C):>8}  {np.sum(sizes >= 20):>23}  {purity(lab, truth):>6.3f}  {its:>5}")

C, lab, _ = mean_shift(X, h0)
big = np.bincount(lab) >= 20
print(f"\ncenters of the big clusters at h={h0:.2f}:", np.round(C[big], 2).tolist())
print("cluster sizes:", sorted(np.bincount(lab)[big].tolist(), reverse=True),
      f"| points in tiny clusters: {np.sum(~big[lab])}, of which planted outliers: {np.sum(~big[lab] & (truth == -1))}")
```

Output (Python 3.14, NumPy 2.5):
```
n=430, rule-of-thumb bandwidth 0.99

bandwidth  clusters  big clusters (>=20 pts)  purity  iters
     0.30        33                        5   1.000    188
     0.99         6                        3   0.998     26
     2.00         3                        3   0.988     42
     4.00         1                        1   0.500     17

centers of the big clusters at h=0.99: [[-0.05, -0.0], [6.02, 1.11], [2.01, 5.98]]
cluster sizes: [206, 127, 89] | points in tiny clusters: 8, of which planted outliers: 8
```

Purity is the share of blob points whose cluster is dominated by their own blob. At the rule-of-thumb bandwidth, mean shift finds the three planted centres (0, 0), (6, 1) and (2, 6) without being told there are three. Eight isolated outliers sit in their own density bumps and form tiny clusters, so cluster size doubles as a simple outlier flag. The other 22 outliers lie close enough to a blob to be absorbed, which is why the big clusters hold 206, 127 and 89 points rather than 200, 120 and 80. The bandwidth sweep is the whole story. At 0.3 the estimate is so bumpy that the blobs split into 5 big clusters plus 28 crumbs, and convergence takes 188 iterations because the slopes are flat. At 2.0 every outlier is pulled into a blob. At 4.0 a single peak remains, and purity falls to 0.5.

## Choosing / trade-offs
- **Mean shift vs k-means.** k-means needs k and assumes compact, similar-sized clusters, and is very fast. Mean shift needs a bandwidth instead of k and finds the modes of any density. Use k-means when you know k or have millions of points.
- **Mean shift vs DBSCAN/HDBSCAN.** DBSCAN also finds arbitrary shapes and marks noise explicitly, and it scales better with a spatial index. HDBSCAN handles clusters of varying density, which a single mean-shift bandwidth cannot.
- **Kernel.** The Gaussian kernel gives smooth convergence. A flat (uniform) kernel, which scikit-learn uses, is faster with a neighbour search and converges in a finite number of steps.
- **Speed.** Naive mean shift is O(n² · iterations). Seed from a grid or a random subset (`bin_seeding` in scikit-learn), use a ball tree for neighbours, and stop points that have already converged.
- **Adaptive bandwidth.** A variable bandwidth (for example based on the distance to the k-th neighbour) helps when density varies across the space, at the cost of another parameter.

## Gotchas
- Scale features first. One Euclidean bandwidth over features in metres and in dollars means one feature decides everything.
- The rule-of-thumb bandwidth assumes roughly Gaussian data and tends to oversmooth multimodal data. Treat it as a starting point and check how the cluster count changes around it.
- The mode-merge threshold is a second, hidden parameter. Too small and one peak splits into many clusters; too large and nearby peaks merge.
- Without a minimum cluster size you get many one-point clusters from outliers. Decide whether to drop them, label them as noise, or attach them to the nearest big cluster.
- The pairwise distance matrix needs `n × n` memory per iteration. At 50,000 points that is 2.5 billion entries, so use a neighbour structure or a subset of seeds.
- Plateaus (uniform density regions) give near-zero mean-shift vectors and very slow convergence. Cap the iterations and report how many points converged.

## Related
- [[clustering]] - clustering methods in scikit-learn and how to choose between them.
- [[kernel-methods-and-density-estimation]] - kernel density estimation, the density mean shift climbs.
- [[k-means-from-scratch]] - the fixed-k, centroid-based alternative.
- [[dbscan-from-scratch]] - another density-based method that labels noise explicitly.
- [[gmm-em-from-scratch]] - parametric density clustering with a chosen number of components.
- [[anomaly-detection]] - using low density or tiny clusters as an outlier signal.

## References
- Comaniciu and Meer (2002), "Mean shift: a robust approach toward feature space analysis", IEEE TPAMI: https://doi.org/10.1109/34.1000236
- Cheng (1995), "Mean shift, mode seeking, and clustering", IEEE TPAMI: https://doi.org/10.1109/34.400568
- scikit-learn user guide, mean shift: https://scikit-learn.org/stable/modules/clustering.html#mean-shift
