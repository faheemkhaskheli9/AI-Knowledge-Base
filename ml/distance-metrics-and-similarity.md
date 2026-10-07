---
title: Distance metrics and similarity measures
category: ml
tags: [distance, similarity, euclidean, cosine, manhattan, mahalanobis, jaccard, knn, scaling, curse-of-dimensionality, scipy, scikit-learn]
use_cases:
  - "pick the right distance for kNN, clustering or nearest-neighbour search"
  - "compare text or embedding vectors (cosine vs dot product vs Euclidean)"
  - "find similar customers/products from mixed or binary features"
  - "flag outliers that are far from the data's correlation structure (Mahalanobis)"
  - "understand why kNN accuracy collapses on unscaled or high-dimensional data"
status: stable
last_verified: 2026-10-04
sources:
  - https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
  - https://scikit-learn.org/stable/modules/neighbors.html
  - https://scikit-learn.org/stable/modules/metrics.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html
---

# Distance metrics and similarity measures

## Summary
Every distance-based method (kNN, k-means, DBSCAN, hierarchical clustering, RBF kernels, vector search, anomaly scores) silently assumes a definition of "close". The metric matters as much as the algorithm: the same kNN on the same data scores 0.69 with raw Euclidean distance and 0.95 after scaling. This file covers the common metrics, which data each one suits, and the two traps that break them: unscaled features and high dimensions.

## Key concepts
- **Minkowski family (Lp).** `d = (Σ|xᵢ-yᵢ|^p)^(1/p)`. p=2 is Euclidean (straight line), p=1 is Manhattan / cityblock (sum of absolute gaps, less sensitive to one large gap), p=∞ is Chebyshev (largest single gap).
- **Cosine.** `1 - x·y / (‖x‖‖y‖)` compares direction and ignores length. The default for text (TF-IDF) and embeddings. On L2-normalized vectors, cosine distance = ‖x-y‖²/2, so Euclidean and cosine rank neighbours identically.
- **Dot product.** Not a metric (no triangle inequality), but it is what many retrieval models are trained with. Use it when the model card says so; vector length then carries meaning (e.g. popularity).
- **Mahalanobis.** `sqrt((x-y)ᵀ Σ⁻¹ (x-y))` measures distance in units of the data's covariance. It removes scale and correlation, so it is the right "how unusual is this row" score for roughly Gaussian data.
- **Set / binary.** Jaccard = 1 - |A∩B| / |A∪B| (ignores shared zeros, good for baskets, tags, sparse binary). Hamming = fraction of positions that differ (fixed-length codes, hashes).
- **Mixed types.** Gower distance averages per-feature distances (range-scaled numeric, match/mismatch categorical). Not in scikit-learn; use the `gower` package or one-hot + scaling.
- **Sequences and distributions.** Edit (Levenshtein) distance for strings, DTW for time series of different speed, KL / Jensen-Shannon / Wasserstein for distributions ([[information-theory-for-ml]]).
- **Similarity ↔ distance.** Similarity s in [0, 1] becomes distance 1 - s; an RBF kernel `exp(-γ d²)` turns a distance into a similarity ([[kernel-methods-and-density-estimation]]).

## When to use / scenarios
- **Euclidean (after scaling):** dense continuous features of comparable meaning: sensor readings, physical measurements, PCA outputs, k-means.
- **Manhattan:** many features, some with outliers or heavy tails; grid-like movement costs.
- **Cosine:** TF-IDF documents, sentence/image embeddings, user-item vectors where magnitude is mostly "activity level" ([[embeddings]]).
- **Mahalanobis:** fraud and quality-control outlier scores on correlated numeric features; multivariate process monitoring ([[anomaly-detection]]).
- **Jaccard:** market baskets, tag sets, near-duplicate detection with MinHash ([[association-rules-market-basket]]).
- **Learned metric:** when no hand-picked metric matches "similar" for your task (faces, products, signatures), train one ([[metric-learning-and-few-shot]]).
- NOT a fix for bad features: if kNN is poor with every metric, the features do not carry the signal. Try a tree model ([[gradient-boosting-tabular]]).

## Setup & code
```bash
pip install "scikit-learn>=1.4" scipy numpy
```
Metrics on toy vectors, the effect of scaling on kNN, Mahalanobis neighbours and distance concentration in high dimensions:
```python
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

a, b = np.array([[1.0, 0.0, 2.0]]), np.array([[2.0, 1.0, 0.0]])
for m in ("euclidean", "cityblock", "chebyshev", "cosine"):
    print(f"{m:10s} {cdist(a, b, metric=m)[0, 0]:.3f}")
print(f"jaccard    {cdist([[1, 1, 0, 1]], [[1, 0, 0, 1]], metric='jaccard')[0, 0]:.3f}")

X, y = load_wine(return_X_y=True)
for name, model in [
    ("raw euclidean", KNeighborsClassifier(5)),
    ("scaled euclidean", make_pipeline(StandardScaler(), KNeighborsClassifier(5))),
    ("scaled manhattan", make_pipeline(StandardScaler(), KNeighborsClassifier(5, metric="manhattan"))),
    ("scaled cosine", make_pipeline(StandardScaler(), KNeighborsClassifier(5, metric="cosine"))),
]:
    print(f"{name:17s} acc={cross_val_score(model, X, y, cv=5).mean():.3f}")

VI = np.linalg.inv(np.cov(X, rowvar=False))
d = cdist(X[:1], X, metric="mahalanobis", VI=VI)[0]
print(f"mahalanobis nearest to row 0: {np.argsort(d)[1:4].tolist()}")

rng = np.random.default_rng(0)
for dim in (2, 100, 1000):
    P = rng.random((500, dim))
    D = cdist(P[:1], P[1:])[0]
    print(f"dim={dim:4d} (max-min)/min distance = {(D.max() - D.min()) / D.min():.2f}")
```
Output (scikit-learn 1.9.0, SciPy 1.18.0):
```
euclidean  2.449
cityblock  4.000
chebyshev  2.000
cosine     0.600
jaccard    0.333
raw euclidean     acc=0.691
scaled euclidean  acc=0.949
scaled manhattan  acc=0.950
scaled cosine     acc=0.944
mahalanobis nearest to row 0: [20, 22, 40]
dim=   2 (max-min)/min distance = 66.13
dim= 100 (max-min)/min distance = 0.41
dim=1000 (max-min)/min distance = 0.12
```
Scaling lifts kNN from 0.69 to 0.95: on raw wine data the `proline` column (values in the hundreds) decides every distance. Once scaled, the choice between Euclidean, Manhattan and cosine barely matters. The last block shows **distance concentration**: in 1000 uniform dimensions the farthest point is only 12% farther than the nearest, so "nearest" carries little information.

For many vectors use `sklearn.metrics.pairwise_distances` (parallel, sparse-aware) or a nearest-neighbour index (`NearestNeighbors`, FAISS, a vector DB, [[vector-databases]]) instead of a full `cdist` matrix.

## Choosing / trade-offs
- **Scale first, then pick.** The scaler is part of the metric: `StandardScaler` (Gaussian-ish features), `RobustScaler` (outliers), `MinMaxScaler` (bounded). Weight a feature by multiplying its column after scaling.
- **Cosine vs Euclidean vs dot product for embeddings.** Use what the embedding model was trained with. If vectors are L2-normalized, all three rank neighbours the same and dot product is the cheapest.
- **Mahalanobis needs a good covariance.** With more features than ~n/10 rows the inverse covariance is unstable. Use `sklearn.covariance.LedoitWolf` or `MinCovDet` (robust to the outliers you are trying to find).
- **Exact vs approximate search.** Exact kNN is O(n) per query. Above ~1M vectors, or at low latency, use approximate indexes (HNSW, IVF) and accept a small recall loss.
- **Reduce dimension before distance.** PCA / UMAP to tens of dimensions often makes distances more meaningful and search faster ([[dimensionality-reduction]]).

## Gotchas
- One-hot columns plus scaled numeric columns: each categorical gets weight from its number of levels. Check whether a 50-level category dominates the distance.
- Cosine distance on vectors with negative or zero entries can behave oddly: a zero vector has undefined cosine (scikit-learn returns distance 1 for it). Drop or handle empty documents.
- `metric="cosine"` in `KNeighborsClassifier` forces brute-force search (no KD-tree/ball-tree). Normalize the rows and use Euclidean if speed matters.
- Euclidean distance on raw time series punishes small time shifts. Use DTW, or features (FFT, statistics), for shape similarity.
- Haversine for latitude/longitude, not Euclidean on degrees. scikit-learn's `haversine` expects **radians** in (lat, lon) order.
- Distances leak scale from the training set: fit the scaler on train only, inside the pipeline ([[data-leakage-and-validation-splits]]).
- High-dimensional sparse data (bag-of-words) makes Euclidean meaningless; cosine or Jaccard work far better.

## Related
- [[svm-knn-naive-bayes]] - kNN and RBF-SVM are where the metric choice shows up first.
- [[clustering]] - k-means assumes Euclidean; DBSCAN/agglomerative accept any metric.
- [[feature-engineering]] - scaling and encoding decide what the distance sees.
- [[metric-learning-and-few-shot]] - learn the metric when hand-picked ones fail.
- [[embeddings]] - cosine/dot-product search over learned vectors.
- [[anomaly-detection]] - distance and Mahalanobis-based outlier scores.
- [[dimensionality-reduction]] - fight distance concentration before measuring.

## References
- SciPy `scipy.spatial.distance` (all metric definitions): https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
- scikit-learn, Nearest neighbors: https://scikit-learn.org/stable/modules/neighbors.html
- scikit-learn, Pairwise metrics and kernels: https://scikit-learn.org/stable/modules/metrics.html
- Aggarwal, Hinneburg & Keim, "On the Surprising Behavior of Distance Metrics in High Dimensional Space", ICDT 2001.
- Beyer et al., "When Is 'Nearest Neighbor' Meaningful?", ICDT 1999.
