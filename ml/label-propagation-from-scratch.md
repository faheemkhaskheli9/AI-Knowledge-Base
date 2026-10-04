---
title: Label propagation / label spreading from scratch (graph semi-supervised learning)
category: ml
tags: [label-propagation, label-spreading, semi-supervised-learning, graph-based-learning, knn-graph, transductive, two-moons, numpy, from-scratch]
use_cases:
  - "implement label spreading (Zhou et al. 2004) on a kNN graph in NumPy"
  - "classify a dataset where only a handful of points are labelled but the clusters follow a manifold"
  - "compare graph-based semi-supervised learning with a classifier trained on the labelled points only"
  - "explain the cluster/manifold assumption, alpha and the normalised graph Laplacian in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://papers.nips.cc/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html
  - http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf
  - https://scikit-learn.org/stable/modules/semi_supervised.html
---

# Label propagation / label spreading from scratch (graph semi-supervised learning)

## Summary
Label propagation turns a few labels into many by letting them flow along a similarity graph. Build a k-nearest-neighbour graph over all points, labelled and unlabelled, put each known label on its node, and repeatedly let every node take the weighted average of its neighbours' label scores. Labels travel along dense regions and stop at gaps between clusters. The method assumes points connected through high-density paths share a label (the cluster / manifold assumption). This file implements label spreading (Zhou et al., 2004), the variant with a normalised graph and a pull back toward the seed labels, on two interleaved half-moons of 600 points. With 2 labels in total, one per class, it labels the other 598 points with 99.5% accuracy on average over 10 seeds, against 78.1% for 1-nearest-neighbour on the same two labels. With 20 labels the gap closes (100% vs 99.0%).

## Key concepts
- **Graph.** `W_ij = exp(−||x_i − x_j||² / σ²)` for the k nearest neighbours, made symmetric. Sparse kNN graphs work better than dense Gaussian graphs, which connect across the gap between clusters.
- **Normalisation.** `S = D^{−1/2} W D^{−1/2}`, where `D` is the diagonal degree matrix. It stops high-degree hubs from dominating.
- **Iteration.** `F ← α S F + (1 − α) Y`, where `Y` holds one-hot seed labels (zero rows for unlabelled points) and `α ∈ (0, 1)` trades smoothness against fitting the seeds. The prediction is `argmax F`.
- **Closed form.** The fixed point is `F* = (1 − α)(I − αS)^{−1} Y`, which minimises a smoothness term `Σ W_ij ||F_i/√d_i − F_j/√d_j||²` plus a fit-to-seeds term. The loop is a cheap way to approximate the inverse.
- **Propagation vs spreading.** Label propagation (Zhu & Ghahramani, 2002) uses the row-normalised `D^{−1}W` and clamps labelled nodes to their labels each step. Spreading uses the symmetric `S` and lets seed labels change, which tolerates a wrong label.
- **Transductive.** The output is labels for the points in the graph. A new point needs its edges to the graph (or retraining an inductive model on the propagated labels).

## When to use / scenarios
- Few labels, many unlabelled points, and data whose classes form connected clusters: document or image embeddings, user graphs, sensor clusters.
- Pseudo-labelling: propagate on embedding space, keep the confident labels, train a normal classifier on them.
- Label cleanup: spreading with `α` close to 1 can flag seed labels that disagree with their neighbourhood (see [[label-noise-and-data-cleaning]]).
- Interviews: "what assumption does semi-supervised learning rely on", "propagation vs spreading", "what does α do", "how does this relate to the graph Laplacian and spectral clustering".
- Not for: classes that overlap in feature space (the graph connects them and labels leak), or huge datasets without approximate neighbour search.

## Setup & code
`pip install numpy`. Runs in about 10 seconds (dense 600 × 600 matrices, 40 problems).

```python
import numpy as np


def two_moons(n, noise, rng):
    t = rng.uniform(0, np.pi, n)
    half = n // 2
    X = np.r_[np.c_[np.cos(t[:half]), np.sin(t[:half])],
              np.c_[1 - np.cos(t[half:]), 0.5 - np.sin(t[half:])]]
    y = np.r_[np.zeros(half, int), np.ones(n - half, int)]
    return X + rng.normal(0, noise, X.shape), y


def label_spreading(X, y_partial, k=10, alpha=0.99, iters=200):
    """Zhou et al. 2004. y_partial: class id for labelled points, -1 for unlabelled."""
    n, C = len(X), y_partial.max() + 1
    D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    # kNN graph with a Gaussian weight; sigma = mean distance to the k-th neighbour
    nn = np.argsort(D2, 1)[:, 1:k + 1]
    sigma2 = D2[np.arange(n)[:, None], nn].mean()
    W = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k)
    W[rows, nn.ravel()] = np.exp(-D2[rows, nn.ravel()] / sigma2)
    W = np.maximum(W, W.T)                                   # symmetrise
    dinv = 1 / np.sqrt(W.sum(1))
    S = dinv[:, None] * W * dinv[None]                       # D^-1/2 W D^-1/2
    Y0 = np.zeros((n, C))
    lab = y_partial >= 0
    Y0[lab, y_partial[lab]] = 1
    F = Y0.copy()
    for _ in range(iters):
        F = alpha * S @ F + (1 - alpha) * Y0                 # spread, then pull back to the seeds
    return F.argmax(1)


def knn_predict(Xtr, ytr, X, k=1):
    d = ((X[:, None] - Xtr[None]) ** 2).sum(-1)
    return np.array([np.bincount(ytr[r], minlength=2).argmax() for r in np.argsort(d, 1)[:, :k]])


for n_lab in [2, 4, 10, 20]:
    accs_lp, accs_nn = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        X, y = two_moons(600, 0.08, rng)
        # n_lab labels per class
        idx = np.r_[rng.choice(np.where(y == 0)[0], n_lab // 2, replace=False),
                    rng.choice(np.where(y == 1)[0], n_lab // 2, replace=False)]
        yp = -np.ones(len(y), int)
        yp[idx] = y[idx]
        un = yp < 0
        accs_lp.append((label_spreading(X, yp)[un] == y[un]).mean())
        accs_nn.append((knn_predict(X[idx], y[idx], X[un]) == y[un]).mean())
    print(f"{n_lab:2d} labels: label spreading {np.mean(accs_lp):.3f} (min {np.min(accs_lp):.3f}) | "
          f"1-NN on labels only {np.mean(accs_nn):.3f}")
```

Output (numpy 2.5, 10 random draws of data and labels per row; accuracy on the unlabelled points):
```
 2 labels: label spreading 0.995 (min 0.970) | 1-NN on labels only 0.781
 4 labels: label spreading 1.000 (min 0.998) | 1-NN on labels only 0.840
10 labels: label spreading 1.000 (min 1.000) | 1-NN on labels only 0.945
20 labels: label spreading 1.000 (min 1.000) | 1-NN on labels only 0.990
```

A supervised model that sees only the labelled points draws its boundary halfway between them, which cuts through both moons: 1-NN with one label per class gets 78% right. Label spreading uses the 598 unlabelled points to discover that each moon is one connected strip, so a single seed is enough to label the whole strip (99.5% on average; the worst of 10 draws is 97%, where noise put a few bridging edges between the moons). As labels are added the supervised baseline catches up, because enough labels outline the moons by themselves. Semi-supervised learning pays off most when labels are scarce and the cluster assumption holds.

## Choosing / trade-offs
- **k and σ.** Small k (5 to 15) keeps clusters separate but can leave the graph disconnected (isolated components never get a label). Large k or a dense RBF graph bridges gaps. Check the number of connected components first.
- **α.** Close to 1 (0.9 to 0.99) trusts the graph; smaller values keep predictions close to the seeds. Spreading with `α < 1` tolerates noisy seeds; clamped propagation does not.
- **Iterations vs closed form.** The inverse is `O(n³)`; the iteration is `O(iters × nnz)` on a sparse graph and scales to millions of nodes.
- **Scaling up.** Use approximate kNN (FAISS, HNSW, or [[locality-sensitive-hashing-from-scratch]]) and sparse matrices. `sklearn.semi_supervised.LabelSpreading(kernel="knn")` does exactly this.
- **Alternatives.** Self-training / pseudo-labelling with a strong model, consistency-regularisation methods (FixMatch) for images, or graph neural networks ([[gcn-from-scratch-numpy]]), which propagate learned features instead of labels.

## Gotchas
- The cluster assumption is the whole method. If two classes share a dense region, labels leak through it and accuracy can fall below a supervised baseline.
- Scale features first. The graph is built from Euclidean distances, so one large-range feature decides every neighbour (see [[feature-scaling-and-normalization]]).
- Class imbalance in the seeds biases the result: the class with more seeds floods shared regions. Normalise each column of `F` by its seed count (class mass normalisation) when seeds are unbalanced.
- Evaluate only on unlabelled points; counting the seeds inflates accuracy.
- A disconnected component with no seed gets an all-zero row and `argmax` silently returns class 0. Detect and report those points.
- Dense `n × n` distance matrices stop fitting in memory around tens of thousands of points.

## Related
- [[semi-supervised-and-active-learning]] - the wider family: pseudo-labelling, consistency training, active learning.
- [[spectral-clustering-from-scratch]] - the same normalised graph, used without labels.
- [[gcn-from-scratch-numpy]] - neural message passing on a graph.
- [[knn-from-scratch]] - the supervised baseline and the graph-building step.
- [[label-noise-and-data-cleaning]] - using neighbourhood agreement to find bad labels.
- [[locality-sensitive-hashing-from-scratch]] - approximate neighbours for building the graph at scale.

## References
- Zhou, Bousquet, Lal, Weston and Schölkopf (2004), "Learning with Local and Global Consistency", NeurIPS 2003: https://papers.nips.cc/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html
- Zhu and Ghahramani (2002), "Learning from Labeled and Unlabeled Data with Label Propagation", CMU-CALD-02-107: http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf
- scikit-learn user guide, "Semi-supervised learning": https://scikit-learn.org/stable/modules/semi_supervised.html
