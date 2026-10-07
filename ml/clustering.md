---
title: Clustering (k-means, DBSCAN, HDBSCAN, Gaussian mixtures)
category: ml
tags: [clustering, unsupervised, k-means, dbscan, hdbscan, gaussian-mixture, segmentation, silhouette, scikit-learn]
use_cases:
  - "segment customers into groups for targeted marketing without labels"
  - "group similar support tickets, documents or embeddings into topics"
  - "find natural groups of stores, products or sensors with similar behaviour"
  - "pick the number of clusters and check whether the clusters are real"
status: stable
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/clustering.html
  - https://scikit-learn.org/stable/modules/mixture.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
---

# Clustering (k-means, DBSCAN, HDBSCAN, Gaussian mixtures)

## Summary
Clustering groups rows that are similar to each other without any labels. k-means splits data into k round groups around centroids and is the fast default; DBSCAN and HDBSCAN find dense regions of any shape and mark the rest as noise; Gaussian mixtures give soft, probabilistic membership. There is no ground truth, so the result is only as good as the feature space and distance you choose, and it has to be checked against what the business can use.

## Key concepts
- **Distance defines the result.** Clusters are groups that are close under some metric (Euclidean, cosine). Scale features first, or the feature with the largest units decides everything.
- **k-means.** Assign each point to the nearest of k centroids, move centroids to the mean, repeat. Assumes roughly spherical, similar-sized clusters; you must choose k. `MiniBatchKMeans` for millions of rows.
- **DBSCAN.** A cluster is a dense region: points with at least `min_samples` neighbours within `eps`. Finds any shape, labels outliers `-1`, needs no k, but one `eps` must fit all densities.
- **HDBSCAN.** Hierarchical DBSCAN that handles clusters of different densities and only needs `min_cluster_size`; usually the better density-based default (`sklearn.cluster.HDBSCAN`).
- **Gaussian mixture (GMM).** Each cluster is a Gaussian with its own shape; gives membership probabilities. Pick the number of components with BIC.
- **Agglomerative (hierarchical).** Merge closest groups bottom-up; produces a dendrogram you can cut at any level.
- **Internal metrics.** Silhouette (−1 to 1, higher = tighter, better separated), Davies-Bouldin (lower better), Calinski-Harabasz (higher better). They score geometry, not usefulness.
- **High dimensions.** Distances concentrate in many dimensions; reduce with PCA or cluster on embeddings, see [[dimensionality-reduction]] and [[embeddings]].

## When to use / scenarios
- Customer segmentation for marketing (RFM features: recency, frequency, monetary): k-means with 3-8 clusters, then name each segment from its centroid.
- Topic discovery over support tickets or documents: embed with a sentence model, reduce, then HDBSCAN; label clusters with an LLM.
- Store/product grouping for planning, sensor or machine behaviour profiles in [[manufacturing-iot]].
- Deduplication or near-duplicate detection (tight DBSCAN on embeddings).
- Not when: you actually have labels for the groups you care about (train a classifier); you want outliers rather than groups (use [[anomaly-detection]]); or "segments" will be defined by a business rule anyway (write the rule).

## Setup & code
```bash
pip install scikit-learn
```
```python
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# k-means on blob-shaped data: choose k by silhouette.
X, _ = make_blobs(n_samples=1500, centers=4, cluster_std=1.0, random_state=0)
X = StandardScaler().fit_transform(X)
for k in range(2, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    print(f"k={k}: silhouette={silhouette_score(X, labels):.3f}")

# Non-spherical shapes: k-means cuts the moons in half, HDBSCAN follows them.
Xm, ym = make_moons(n_samples=600, noise=0.06, random_state=0)
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(Xm)
hd = HDBSCAN(min_cluster_size=20).fit_predict(Xm)
agree = lambda a, b: max((a == b).mean(), (a != b).mean())   # label-swap safe
print("k-means agreement with true moons:", round(agree(km, ym), 2))
print("HDBSCAN clusters:", len(set(hd) - {-1}), "noise points:", int((hd == -1).sum()))
```
Silhouette peaks at k=4 for four blobs; on the moons, HDBSCAN recovers the two shapes that k-means cannot.

## Choosing / trade-offs
- **k-means** when clusters are roughly round, you can name k, and data is large; fastest and easiest to explain (each segment has a centroid profile).
- **HDBSCAN** when shapes are irregular, densities differ, noise should be left out, or k is unknown (common for embeddings).
- **GMM** when clusters overlap and you need soft membership ("60% bargain hunter, 40% loyal").
- **Agglomerative** for small data where a hierarchy (segments within segments) is useful.
- **Choosing k.** Silhouette, elbow (inertia) and BIC give candidates; the final k is whatever gives segments that are stable across re-runs and that the team can act on.

## Gotchas
- Unscaled features: a `revenue` column in rupees swamps an `age` column, and you have clustered on revenue alone.
- k-means always returns k clusters, even on structureless data; check that the silhouette is meaningfully above what random data gives.
- k-means results depend on initialisation; use `n_init` ≥ 10 and a fixed `random_state`, and check stability with different seeds or bootstrap samples.
- Cluster IDs are arbitrary and change between runs; never hard-code "cluster 3 = VIP" without matching on centroids.
- Categorical features do not fit Euclidean distance; one-hot with care, or use k-prototypes / Gower distance (outside scikit-learn).
- t-SNE/UMAP plots show apparent clusters that may not exist in the original space; cluster on the data or PCA, use t-SNE/UMAP only to look.

## Related
- [[dimensionality-reduction]] - PCA/UMAP before clustering high-dimensional data, and for plotting clusters.
- [[embeddings]] - turning text or images into vectors you can cluster.
- [[anomaly-detection]] - when the interesting points are the ones outside every cluster.
- [[ml-fundamentals]] - where unsupervised learning fits among problem types.
- [[feature-engineering]] - scaling and encoding that decide what "similar" means.

## References
- scikit-learn clustering guide (includes the method comparison chart): https://scikit-learn.org/stable/modules/clustering.html
- Gaussian mixture models: https://scikit-learn.org/stable/modules/mixture.html
- HDBSCAN API: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
