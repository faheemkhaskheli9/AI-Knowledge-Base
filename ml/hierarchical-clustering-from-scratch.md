---
title: Hierarchical (agglomerative) clustering from scratch (linkages, Lance-Williams update, dendrogram cuts)
category: ml
tags: [hierarchical-clustering, agglomerative-clustering, linkage, single-linkage, complete-linkage, average-linkage, ward, dendrogram, lance-williams, scipy, scikit-learn, numpy, from-scratch, ml-basics]
use_cases:
  - "implement agglomerative clustering from scratch for learning or an interview"
  - "understand single, complete, average and Ward linkage and when each works"
  - "build a dendrogram and choose the number of clusters from merge heights"
  - "cluster a few thousand items when a full hierarchy (taxonomy) is useful"
status: stable
last_verified: 2026-10-04
sources:
  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
  - https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
  - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
  - https://doi.org/10.1093/comjnl/9.4.373
---

# Hierarchical (agglomerative) clustering from scratch (linkages, Lance-Williams update, dendrogram cuts)

## Summary
Agglomerative clustering starts with every point as its own cluster and repeatedly merges the two closest clusters until one is left. The merge history is a tree (a dendrogram). Cutting the tree at any height gives a flat clustering, so one run yields every `k`. What "closest clusters" means is the **linkage**: single (nearest pair), complete (farthest pair), average (mean pair) or Ward (smallest increase in within-cluster variance). The 40-line NumPy version below reproduces SciPy's merge heights exactly for all four linkages.

## Key concepts
- **Algorithm.** Compute the pairwise distance matrix; repeat `n − 1` times: find the smallest entry, merge those two clusters, update the merged cluster's distance to every other cluster, and record `[a, b, height, size]`. That record is SciPy's linkage matrix `Z`; new clusters get ids `n, n+1, ...`.
- **Lance-Williams update.** All four linkages update distances from the two old rows alone, without revisiting the points: single `min(dᵢ, dⱼ)`, complete `max(dᵢ, dⱼ)`, average `(nᵢdᵢ + nⱼdⱼ)/(nᵢ + nⱼ)`, Ward `((nᵢ+nₖ)dᵢ + (nⱼ+nₖ)dⱼ − nₖdᵢⱼ)/(nᵢ+nⱼ+nₖ)` on squared distances.
- **Linkage behaviour.** Single linkage follows chains, so it finds curved shapes but also bridges clusters through a few noisy points. Complete and average prefer compact clusters. Ward behaves like k-means: round clusters of similar size.
- **Dendrogram cut.** Undo the last `k − 1` merges to get `k` clusters, or cut at a height. A large jump between consecutive merge heights means two well-separated groups were forced together, so stop just before it.
- **Cost.** The naive loop is `O(n³)` time and `O(n²)` memory. Libraries use `O(n²)` algorithms (nearest-neighbour chain, MST for single linkage), but the `O(n²)` distance matrix still limits practical `n` to tens of thousands.

## When to use / scenarios
- You want a hierarchy, not just a partition: product or document taxonomies, gene expression heatmaps, customer segments at several granularities.
- You do not know `k` and want to see the structure first; the dendrogram shows candidate cuts.
- Small to medium data (up to roughly 10-20k points) with any distance, including precomputed ones such as edit distance or 1 − correlation ([[distance-metrics-and-similarity]]).
- Learning: a clean example of greedy merging and of how the choice of linkage changes the result.
- Not for: large `n` (use k-means, MiniBatchKMeans or HDBSCAN), or when clusters need to be revised after a bad early merge, since merges are never undone ([[clustering]]).

## Setup & code
`pip install numpy scipy scikit-learn`. Runs in a few seconds on CPU.

```python
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score


def agglomerative(X, method="average"):
    """Naive O(n^3) agglomerative clustering. Returns a SciPy-style linkage matrix (n-1, 4)."""
    n = len(X)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    if method == "ward":
        D = D ** 2                                   # Lance-Williams for Ward works on squared distances
    np.fill_diagonal(D, np.inf)
    size = {i: 1 for i in range(n)}                  # active cluster id -> size
    ids = list(range(n))                             # row/col of D -> cluster id
    Z = []
    for step in range(n - 1):
        i, j = np.unravel_index(np.argmin(D), D.shape)
        i, j = min(i, j), max(i, j)
        a, b = ids[i], ids[j]
        na, nb = size.pop(a), size.pop(b)
        h = np.sqrt(D[i, j]) if method == "ward" else D[i, j]
        Z.append([min(a, b), max(a, b), h, na + nb])
        # Lance-Williams: distance from the merged cluster to every other cluster k
        di, dj = D[i], D[j]
        if method == "single":
            new = np.minimum(di, dj)
        elif method == "complete":
            new = np.maximum(di, dj)
        elif method == "average":
            new = (na * di + nb * dj) / (na + nb)
        elif method == "ward":
            nk = np.array([size.get(c, 0) for c in ids], float)
            new = ((na + nk) * di + (nb + nk) * dj - nk * D[i, j]) / (na + nb + nk)
        D[i], D[:, i] = new, new                     # row i becomes the merged cluster
        D[i, i] = np.inf
        D = np.delete(np.delete(D, j, 0), j, 1)
        ids[i] = n + step
        del ids[j]
        size[n + step] = na + nb
    return np.array(Z)


def cut(Z, k):
    """Labels for k clusters: undo the last k-1 merges."""
    n = len(Z) + 1
    parent = list(range(2 * n - 1))
    for step, (a, b, _, _) in enumerate(Z[: n - k]):
        parent[int(a)] = parent[int(b)] = n + step
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    _, labels = np.unique([root(i) for i in range(n)], return_inverse=True)
    return labels


Xb, yb = make_blobs(150, centers=3, cluster_std=[0.6, 1.0, 1.4], random_state=0)
for m in ("single", "complete", "average", "ward"):
    Z = agglomerative(Xb, m)
    ref = linkage(Xb, m)
    print(f"{m:8s} merge heights match scipy: {np.allclose(Z[:, 2], ref[:, 2])}, "
          f"ARI(k=3) vs scipy: {adjusted_rand_score(fcluster(ref, 3, 'maxclust'), cut(Z, 3)):.3f}, "
          f"vs truth: {adjusted_rand_score(yb, cut(Z, 3)):.3f}")

Z = agglomerative(Xb, "ward")
sk = AgglomerativeClustering(3, linkage="ward").fit_predict(Xb)
print("ward ARI vs sklearn AgglomerativeClustering:", adjusted_rand_score(sk, cut(Z, 3)))
print("last 5 ward merge heights:", np.round(Z[-5:, 2], 2))
print("k from the biggest jump in merge height:", len(Z) - np.argmax(np.diff(Z[:, 2])))

Xm, ym = make_moons(200, noise=0.05, random_state=0)
for m in ("single", "average", "ward"):
    print(f"moons {m:8s} ARI: {adjusted_rand_score(ym, cut(agglomerative(Xm, m), 2)):.3f}")
print(f"moons k-means  ARI: {adjusted_rand_score(ym, KMeans(2, n_init=10, random_state=0).fit_predict(Xm)):.3f}")
```

Output (numpy 2.5, scipy 1.18, scikit-learn 1.9):
```
single   merge heights match scipy: True, ARI(k=3) vs scipy: 1.000, vs truth: 0.002
complete merge heights match scipy: True, ARI(k=3) vs scipy: 1.000, vs truth: 0.357
average  merge heights match scipy: True, ARI(k=3) vs scipy: 1.000, vs truth: 0.433
ward     merge heights match scipy: True, ARI(k=3) vs scipy: 1.000, vs truth: 0.787
ward ARI vs sklearn AgglomerativeClustering: 1.0
last 5 ward merge heights: [ 7.71  7.87  8.43 23.12 29.89]
k from the biggest jump in merge height: 3
moons single   ARI: 1.000
moons average  ARI: 0.433
moons ward     ARI: 0.433
moons k-means  ARI: 0.256
```

All 149 merge heights match SciPy for every linkage, and the 3-cluster cuts agree with SciPy and scikit-learn exactly (ARI 1.0). Linkage choice dominates the result. On three Gaussian blobs of different spread, Ward recovers the groups best (ARI 0.79), while single linkage chains 146 of 150 points into one cluster and makes the other two out of 3 and 1 stray points (ARI 0.002). The Ward merge heights jump from 8.4 to 23.1 at the last-but-one merge, which correctly suggests `k = 3`. On two interleaved moons the ranking flips: single linkage follows each curve and is perfect (ARI 1.0), while average, Ward and k-means all cut straight across.

## Choosing / trade-offs
- **Which linkage.** Ward is the default for numeric features with compact clusters (Euclidean only). Average is a robust choice for arbitrary distance matrices. Single only when clusters are well separated chains with little noise. Complete when you need every cluster to have a bounded diameter.
- **Hierarchical vs k-means.** k-means scales to millions of points but needs `k` and gives one partition. Hierarchical gives all `k` at once and works with any distance, at `O(n²)` memory ([[k-means-from-scratch]]).
- **Hierarchical vs DBSCAN/HDBSCAN.** HDBSCAN is itself a single-linkage-style hierarchy made robust to noise; prefer it for arbitrary shapes with outliers ([[dbscan-from-scratch]]).
- **Connectivity constraints.** scikit-learn's `AgglomerativeClustering(connectivity=...)` only merges neighbours in a given graph (e.g. adjacent pixels or regions), which speeds it up and gives spatially contiguous clusters.

## Gotchas
- Scale features first; every linkage uses distances directly ([[feature-scaling-and-normalization]]).
- Ward's Lance-Williams formula works on **squared** Euclidean distances. Run it on plain distances and the merge order changes silently. SciPy and the code above report heights back on the plain-distance scale.
- `scipy.cluster.hierarchy.linkage` takes raw observations or a **condensed** distance vector (`scipy.spatial.distance.pdist`), not a square matrix. A square matrix is treated as observations and gives wrong results; convert it with `squareform`.
- Ward and centroid linkage are only meaningful for Euclidean distance. Use average or complete with cosine or custom distances.
- Ties in the distance matrix make the merge order, and so the cut, depend on index order.
- There is no `predict` for new points. Assign them to the cluster of the nearest member or centroid, or refit.

## Related
- [[clustering]] - agglomerative clustering, k-means, DBSCAN and others in scikit-learn, and how to pick.
- [[k-means-from-scratch]] - the flat, centroid-based alternative; Ward optimises a similar objective greedily.
- [[dbscan-from-scratch]] - density-based clustering, related to single linkage.
- [[gmm-em-from-scratch]] - soft, probabilistic clustering.
- [[distance-metrics-and-similarity]] - choosing the distance the linkage runs on.
- [[pca-from-scratch]] - reduce dimensions before clustering high-dimensional data.

## References
- SciPy `linkage` (linkage matrix format and methods): https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
- scikit-learn, Hierarchical clustering: https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
- scikit-learn `AgglomerativeClustering`: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
- Lance and Williams (1967), "A General Theory of Classificatory Sorting Strategies", The Computer Journal: https://doi.org/10.1093/comjnl/9.4.373
