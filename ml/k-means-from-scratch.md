---
title: k-means from scratch (Lloyd's algorithm, k-means++ seeding, inertia and the elbow)
category: ml
tags: [k-means, kmeans-plus-plus, lloyds-algorithm, clustering, inertia, elbow-method, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement k-means from scratch for learning or an interview"
  - "understand why k-means needs good seeding and several restarts"
  - "write k-means++ initialisation and match scikit-learn's KMeans"
  - "pick k with the elbow of inertia and know its limits"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/clustering.html#k-means
  - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
  - https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
---

# k-means from scratch (Lloyd's algorithm, k-means++ seeding, inertia and the elbow)

## Summary
k-means splits data into `k` groups by alternating two steps until nothing changes: assign each point to its nearest centre, then move each centre to the mean of its points. This is Lloyd's algorithm. It only finds a local minimum of the within-cluster squared distance (inertia), so the result depends on the starting centres. k-means++ seeding and several restarts fix most of that. About 30 lines of NumPy reach the same inertia as scikit-learn's `KMeans`.

## Key concepts
- **Objective.** Inertia = `Σᵢ ‖xᵢ − c_{label(i)}‖²`, the total squared distance from each point to its centre. Lower is better for a fixed `k`.
- **Assignment step.** Give each point the label of its nearest centre (an `(n, k)` distance matrix and `argmin`). This can only lower inertia.
- **Update step.** Move each centre to the mean of its assigned points. The mean minimises the summed squared distance, so this also can only lower inertia. Both steps non-increasing means the loop always converges, usually in tens of iterations.
- **Local minima.** Convergence is to a local optimum that depends on the initial centres. Two centres can start in one true cluster and never leave.
- **k-means++.** Pick the first centre at random, then each next centre with probability proportional to `D(x)²`, the squared distance to the nearest centre already chosen. Spread-out seeds give an expected `O(log k)` approximation to the optimum and faster convergence.
- **Restarts (`n_init`).** Run several seeds and keep the lowest-inertia result.
- **Empty clusters.** If no point picks a centre, re-seed it (a random point, or the point farthest from its centre) instead of computing the mean of nothing.
- **Choosing k.** Inertia always falls as `k` grows, so "lowest inertia" always picks `k = n`. Look for the elbow where the drop flattens, or use silhouette ([[clustering]]).

## When to use / scenarios
- Learning: the cleanest example of an alternating-minimisation (EM-like) algorithm. [[gaussian-mixture-models-and-em]] is its soft, probabilistic version.
- Interviews: writing Lloyd's loop, explaining k-means++ and why k-means fails on non-convex shapes.
- Vector quantisation: colour palettes, codebooks, a cheap first pass before an index ([[distance-metrics-and-similarity]]).
- Production: use `sklearn.cluster.KMeans` (or `MiniBatchKMeans` for millions of rows, or FAISS for GPU / very large `n`).
- Not for: elongated, ring or moon shapes, very different cluster sizes or densities, or data with many outliers. Use HDBSCAN, spectral clustering or a GMM instead ([[clustering]]).

## Setup & code
`pip install numpy scikit-learn`. Runs on CPU in a few seconds.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


def kmeans_pp(X, k, rng):
    """k-means++ seeding: next centre drawn with prob. proportional to D(x)^2."""
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = ((X[:, None] - np.array(C)[None]) ** 2).sum(-1).min(1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(C)


def kmeans(X, k, n_init=10, max_iter=300, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    best = (np.inf, None, None)
    for _ in range(n_init):                       # keep the lowest-inertia run
        C = kmeans_pp(X, k, rng)
        for _ in range(max_iter):
            d2 = ((X[:, None] - C[None]) ** 2).sum(-1)   # (n, k)
            labels = d2.argmin(1)                        # assignment step
            newC = np.array([X[labels == j].mean(0) if (labels == j).any()
                             else X[rng.integers(len(X))] for j in range(k)])
            shift = ((newC - C) ** 2).sum()
            C = newC                                     # update step
            if shift < tol:
                break
        inertia = ((X - C[labels]) ** 2).sum()
        if inertia < best[0]:
            best = (inertia, C, labels)
    return best


X, y = make_blobs(n_samples=1500, centers=4, cluster_std=1.2, random_state=0)
inertia, C, labels = kmeans(X, 4)
sk = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
print(f"ours inertia={inertia:.1f}  sklearn inertia={sk.inertia_:.1f}")
print("ARI ours vs sklearn:", round(adjusted_rand_score(labels, sk.labels_), 3))
print("ARI ours vs truth:  ", round(adjusted_rand_score(labels, y), 3))

# Elbow: inertia always falls with k, look for the bend.
for k in range(2, 7):
    print(f"k={k}: inertia={kmeans(X, k)[0]:.0f}")
```

Output (numpy 2.5, scikit-learn 1.9):
```
ours inertia=3348.6  sklearn inertia=3348.6
ARI ours vs sklearn: 0.998
ARI ours vs truth:   0.709
k=2: inertia=8179
k=3: inertia=4950
k=4: inertia=3349
k=5: inertia=2917
k=6: inertia=2553
```

The from-scratch version reaches the same inertia as scikit-learn and the same partition up to a point or two on a boundary (ARI 0.998). Agreement with the generating labels is only 0.709 because two of the four blobs overlap at `cluster_std=1.2`, and k-means correctly reports the minimum-inertia split, not the "true" one. The elbow is at k=4: the drop is 3,229 and 1,601 before it, then only 432 and 364 after.

## Choosing / trade-offs
- **Seeding.** k-means++ costs `k` passes over the data but almost always pays for itself. Random seeding needs many more restarts. For huge `n`, scikit-learn samples candidates (greedy k-means++) and k-means|| parallelises it.
- **Lloyd vs Elkan vs mini-batch.** Lloyd is simple. Elkan's triangle-inequality pruning skips distance computations on low-dimensional, well-separated data. Mini-batch k-means updates centres from small samples: much faster, slightly higher inertia.
- **Elbow vs silhouette vs a downstream metric.** The elbow is often ambiguous. Silhouette compares within- vs between-cluster distance. If the clusters feed a later task (segments for a campaign, codebook for retrieval), choose `k` by that task's metric.
- **k-means vs GMM.** k-means gives hard labels and assumes equal, round clusters. A GMM gives probabilities and allows elliptical, unequal clusters at a higher cost ([[gaussian-mixture-models-and-em]]).

## Gotchas
- Scale features first. k-means uses Euclidean distance, so a feature in metres dominates one in kilometres ([[feature-scaling-and-normalization]]).
- The `(n, k, d)` broadcast in the code above needs `n·k·d` floats of memory. For large data compute `‖x‖² − 2x·c + ‖c‖²` with a matrix product, or process in chunks.
- Cluster ids are arbitrary: run twice and label 0 can become label 2. Compare partitions with ARI or a matching, never with `labels_a == labels_b`.
- The mean of an empty cluster is NaN and silently poisons the next iteration. Handle empty clusters explicitly.
- Categorical or one-hot data breaks the "mean" step. Use k-modes / k-prototypes or a different distance ([[categorical-encoding]]).
- k-means always returns `k` clusters, even on uniform noise. Check that the structure is real (silhouette, stability across seeds) before reporting segments.
- In high dimensions all distances look alike and k-means degrades. Reduce dimensions (PCA, embeddings) first ([[curse-of-dimensionality]], [[dimensionality-reduction]]).

## Related
- [[clustering]] - choosing between k-means, HDBSCAN, agglomerative and spectral clustering.
- [[gaussian-mixture-models-and-em]] - the soft, probabilistic generalisation of k-means.
- [[distance-metrics-and-similarity]] - the distance k-means assumes and alternatives.
- [[feature-scaling-and-normalization]] - required before any distance-based method.
- [[curse-of-dimensionality]] - why k-means struggles on raw high-dimensional data.
- [[linear-regression-from-scratch]] - the other "closed-form step in a loop" from-scratch model.

## References
- scikit-learn, K-means (Lloyd, Elkan, mini-batch, k-means++): https://scikit-learn.org/stable/modules/clustering.html#k-means
- scikit-learn KMeans API: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- Arthur and Vassilvitskii, k-means++: The Advantages of Careful Seeding (SODA 2007): https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
