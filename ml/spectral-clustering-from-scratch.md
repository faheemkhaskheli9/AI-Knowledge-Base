---
title: Spectral clustering from scratch (kNN graph, normalised Laplacian, eigengap)
category: ml
tags: [spectral-clustering, graph-laplacian, eigenvectors, eigengap, affinity-matrix, clustering, unsupervised-learning, k-means, numpy, from-scratch, ml-basics]
use_cases:
  - "implement Ng-Jordan-Weiss spectral clustering in NumPy"
  - "cluster non-convex shapes (rings, moons) where k-means fails"
  - "pick the number of clusters with the eigengap heuristic"
  - "explain the graph Laplacian, eigenvectors and the neighbour parameter in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/0711.0189
  - https://papers.nips.cc/paper/2092-on-spectral-clustering-analysis-and-an-algorithm
  - https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering
---

# Spectral clustering from scratch (kNN graph, normalised Laplacian, eigengap)

## Summary
Spectral clustering treats the data as a graph: points are nodes, and nearby points are joined by weighted edges. It embeds each point using the bottom eigenvectors of the graph Laplacian, then runs k-means in that embedding. Clusters that are connected but not round, such as two concentric rings, become well-separated blobs there. On noisy rings the NumPy version below scores a perfect adjusted Rand index (ARI 1.0) where k-means on the raw coordinates scores 0. The catch is the graph. With 10 neighbours it is perfect; with 5 the graph splits into extra pieces (ARI 0.73); with 50 or more, edges jump between the rings and the method fails completely (ARI 0). On three Gaussian blobs, the largest gap in the eigenvalues correctly picks k = 3.

## Key concepts
- **Affinity graph.** `W_ij = exp(−‖x_i − x_j‖² / σ_i σ_j)` for each point's k nearest neighbours, symmetrised. Local scales `σ_i` (distance to the k-th neighbour, Zelnik-Manor and Perona) adapt to varying density.
- **Degree.** `d_i = Σ_j W_ij`, `D = diag(d)`.
- **Normalised Laplacian.** `L_sym = I − D^{−1/2} W D^{−1/2}`. Its eigenvalues lie in [0, 2].
- **Connected components.** The number of zero eigenvalues equals the number of connected components. If the clusters are separate components, the bottom k eigenvectors are indicators of them.
- **Embedding.** Take the k eigenvectors with the smallest eigenvalues as columns, normalise each row to unit length (Ng-Jordan-Weiss), then run k-means on the rows.
- **Graph cut view.** This relaxes the NP-hard normalised-cut problem: cut few edges while keeping the parts balanced.
- **Eigengap.** If the first k eigenvalues are near 0 and the (k+1)-th is much larger, there are k well-separated clusters.

## When to use / scenarios
- Learning: why "cluster" depends on the similarity you choose; eigenvectors of a graph as features; the link to GNNs ([[gcn-from-scratch-numpy]]).
- Interviews: "why does k-means fail on rings", "what do the Laplacian eigenvectors mean", "how do you pick k", "how does spectral clustering scale".
- Non-convex clusters in low-to-moderate dimension: shapes in 2-D/3-D point clouds, image segmentation (pixels as nodes), community detection when you already have a graph.
- Not for: large n without approximation (dense eigendecomposition is O(n³)), clusters that differ mainly in density ([[dbscan-from-scratch]] or HDBSCAN), or high-dimensional raw data where Euclidean neighbours are meaningless (embed first).

## Setup & code
`pip install numpy`. Runs in under a second on CPU.

```python
import numpy as np


def kmeans(X, k, rng, iters=100, restarts=10):
    best = None
    for _ in range(restarts):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C_new = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(C_new, C):
                break
            C = C_new
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return best[1]


def spectral_clustering(X, k, n_neighbors, rng):
    """Ng-Jordan-Weiss: kNN affinity -> normalised Laplacian -> k smallest eigenvectors -> k-means."""
    d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    idx = np.argsort(d2, 1)[:, 1:n_neighbors + 1]
    sigma = np.sqrt(d2[np.arange(len(X))[:, None], idx][:, -1])            # local scale: distance to k-th neighbour
    W = np.zeros_like(d2)
    rows = np.repeat(np.arange(len(X)), n_neighbors)
    W[rows, idx.ravel()] = np.exp(-d2[rows, idx.ravel()] / (sigma[rows] * sigma[idx.ravel()]))
    W = np.maximum(W, W.T)                                                  # symmetrise the kNN graph
    d = W.sum(1)
    L_sym = np.eye(len(X)) - W / np.sqrt(np.outer(d, d))                    # I - D^-1/2 W D^-1/2
    evals, evecs = np.linalg.eigh(L_sym)
    U = evecs[:, :k]
    U = U / np.linalg.norm(U, axis=1, keepdims=True)                        # row-normalise
    return kmeans(U, k, rng), evals[:k + 2]


def ari(a, b):
    """Adjusted Rand index (1 = identical partitions, ~0 = random)."""
    comb = lambda n: n * (n - 1) / 2
    ct = np.array([[np.sum((a == i) & (b == j)) for j in np.unique(b)] for i in np.unique(a)])
    s_ij, s_a, s_b = comb(ct).sum(), comb(ct.sum(1)).sum(), comb(ct.sum(0)).sum()
    exp = s_a * s_b / comb(len(a))
    return (s_ij - exp) / (0.5 * (s_a + s_b) - exp)


rng = np.random.default_rng(0)
# two concentric rings: k-means fails, spectral succeeds
n = 300
t = rng.uniform(0, 2 * np.pi, n)
r = np.where(np.arange(n) < n // 2, 1.0, 3.0)
X = np.c_[r * np.cos(t), r * np.sin(t)] + rng.normal(0, 0.1, (n, 2))
y = (np.arange(n) >= n // 2).astype(int)

print(f"rings  | k-means ARI {ari(y, kmeans(X, 2, rng)):.3f}")
for nn in [5, 10, 50, 150]:
    lab, ev = spectral_clustering(X, 2, nn, rng)
    print(f"rings  | spectral (n_neighbors={nn:3d}) ARI {ari(y, lab):.3f} | smallest eigenvalues {np.round(ev, 4)}")

# eigengap picks k: three blobs
Xb = np.vstack([rng.normal(c, 0.4, (100, 2)) for c in [(0, 0), (4, 0), (2, 3.5)]])
_, ev = spectral_clustering(Xb, 6, 10, rng)
ev = ev[:6]
print(f"blobs  | first eigenvalues {np.round(ev, 4)} | largest gap after k={int(np.argmax(np.diff(ev))) + 1}")
```

Output (numpy 2.5):
```
rings  | k-means ARI -0.003
rings  | spectral (n_neighbors=  5) ARI 0.727 | smallest eigenvalues [-0.     -0.      0.      0.0008]
rings  | spectral (n_neighbors= 10) ARI 1.000 | smallest eigenvalues [0.     0.     0.0013 0.0057]
rings  | spectral (n_neighbors= 50) ARI 0.013 | smallest eigenvalues [-0.      0.0778  0.1083  0.134 ]
rings  | spectral (n_neighbors=150) ARI 0.000 | smallest eigenvalues [-0.      0.3901  0.4708  0.7255]
blobs  | first eigenvalues [-0.     -0.      0.      0.041   0.0424  0.0572] | largest gap after k=3
```

K-means cuts the rings in half with a straight line (ARI 0) because it can only make convex, roughly round clusters. Spectral clustering with 10 neighbours separates them perfectly. The eigenvalues show why: exactly two are 0, meaning the kNN graph has two connected components, one per ring, and the next is well above. With 5 neighbours there are three zero eigenvalues: the sparse graph has broken one ring into two pieces, and asking for k = 2 merges the wrong ones (ARI 0.73). With 50 or 150 neighbours only one eigenvalue is 0. Edges now cross from ring to ring, the graph is a single component, and the method degrades to a k-means-like split (ARI 0). On three blobs, three eigenvalues sit at 0 and the fourth jumps to 0.04, so the largest gap says k = 3. The number of near-zero eigenvalues is the main diagnostic for both the graph and k.

## Choosing / trade-offs
- **Graph construction matters more than the algorithm.** kNN graphs (5–20 neighbours, or about log n) are robust to scale; a fully connected Gaussian kernel needs a well-chosen `σ`; ε-neighbourhood graphs break with varying density. Check the zero eigenvalues before trusting a result.
- **Laplacian variant.** Use the normalised `L_sym` with row normalisation (Ng-Jordan-Weiss) or the random-walk `L_rw = I − D⁻¹W` (Shi-Malik). Both beat the unnormalised `L = D − W`, which favours cutting off tiny groups.
- **Assigning labels.** k-means on the embedding is the default; `discretize` and `cluster_qr` in scikit-learn are deterministic alternatives without restarts.
- **Scale.** Use sparse affinity matrices and `scipy.sparse.linalg.eigsh` (or AMG/LOBPCG) for the few bottom eigenvectors; Nyström approximation for very large n. Dense `eigh` as above is O(n³) time and O(n²) memory, fine to a few thousand points.
- **Alternatives.** DBSCAN/HDBSCAN find arbitrary shapes and noise without fixing k; GMMs give soft round clusters; on an existing graph use Louvain/Leiden community detection.

## Gotchas
- An isolated point has degree 0, and `D^{−1/2}` divides by zero. kNN graphs avoid this; with ε-graphs, drop or connect isolated points.
- Symmetrise the kNN graph (`max` = "either is a neighbour" or `min` = "mutual neighbours"). An asymmetric `W` gives complex eigenvalues and a wrong Laplacian.
- Use `eigh`, not `eig`: the Laplacian is symmetric, `eigh` returns sorted real eigenvalues, and `eig` can return them in any order.
- Eigenvectors for repeated (zero) eigenvalues are any rotation of the indicators, so they are not individually meaningful. That is why the k-means step is needed.
- Feature scaling changes the neighbour graph. Standardise features as you would for k-means.
- ARI and other label-agreement scores need ground truth. Without it, use the eigengap, silhouette scores in the embedding, or stability across neighbour settings.
- Labels are arbitrary integers; compare partitions with ARI, never with `==`.

## Related
- [[clustering]] - library usage and how to pick between k-means, DBSCAN, GMM, hierarchical and spectral.
- [[k-means-from-scratch]] - the final step inside spectral clustering, and the baseline it beats.
- [[dbscan-from-scratch]] - the other way to find non-convex clusters, density-based.
- [[gcn-from-scratch-numpy]] - the same normalised adjacency `D^{−1/2} W D^{−1/2}` as a neural network layer.
- [[pca-from-scratch]] - eigenvectors of a covariance matrix rather than of a graph.
- [[dimensionality-reduction]] - Laplacian eigenmaps, the embedding step on its own.

## References
- von Luxburg (2007), "A Tutorial on Spectral Clustering", Statistics and Computing 17: https://arxiv.org/abs/0711.0189
- Ng, Jordan and Weiss (2001), "On Spectral Clustering: Analysis and an Algorithm", NeurIPS: https://papers.nips.cc/paper/2092-on-spectral-clustering-analysis-and-an-algorithm
- scikit-learn user guide, Spectral clustering: https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering
