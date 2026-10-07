---
title: t-SNE from scratch (perplexity calibration, Student-t kernel, KL gradient, early exaggeration)
category: ml
tags: [t-sne, tsne, dimensionality-reduction, visualization, manifold-learning, perplexity, kl-divergence, embedding, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement exact t-SNE in NumPy and compare it with scikit-learn"
  - "understand what perplexity controls and how sigma is found per point"
  - "visualise high-dimensional embeddings or features in 2-D"
  - "explain why t-SNE cluster sizes and gaps should not be read literally"
status: stable
last_verified: 2026-10-04
sources:
  - https://www.jmlr.org/papers/v9/vandermaaten08a.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
  - https://distill.pub/2016/misread-tsne/
---

# t-SNE from scratch (perplexity calibration, Student-t kernel, KL gradient, early exaggeration)

## Summary
t-SNE (t-distributed stochastic neighbour embedding) places high-dimensional points in 2-D so that points that are neighbours in the data stay neighbours on the map. It turns distances into neighbour probabilities `P` in the input space, with a Gaussian whose width is tuned per point to a target perplexity. It then moves 2-D points so that their probabilities `Q`, under a heavy-tailed Student-t kernel, match `P` by minimising `KL(P‖Q)`. The NumPy version below has the same neighbourhood preservation as scikit-learn on 800 handwritten digits (trustworthiness 0.993 vs 0.994; PCA scores 0.800). It also shows that cluster sizes on the map do not follow the spread of each class in the data.

## Key concepts
- **Conditional probabilities** `p_{j|i} ∝ exp(−‖x_i − x_j‖² · β_i)`. Each point gets its own precision `β_i`, found by binary search so the row's entropy equals `log(perplexity)`. Perplexity is roughly "the number of effective neighbours", usually 5-50.
- **Symmetrise:** `p_ij = (p_{j|i} + p_{i|j}) / 2n`, so every point contributes and outliers still get a pull.
- **Student-t kernel in 2-D:** `q_ij ∝ (1 + ‖y_i − y_j‖²)⁻¹`. Its heavy tail lets moderately distant points sit far apart on the map, which fixes the "crowding problem" of SNE's Gaussian map.
- **Gradient:** `∂KL/∂y_i = 4 Σ_j (p_ij − q_ij)(1 + ‖y_i − y_j‖²)⁻¹ (y_i − y_j)`. Attractive where `p > q`, repulsive where `q > p`.
- **Early exaggeration.** Multiply `P` by about 12 for the first 250 iterations so clusters form tight, well-separated groups before the fine layout settles.
- **Optimisation tricks.** Momentum 0.5 then 0.8, per-parameter adaptive gains, tiny random initialisation (or PCA initialisation).
- **Cost.** Exact t-SNE is O(n²) in time and memory. Barnes-Hut (scikit-learn's default) is O(n log n); FFT-based methods (openTSNE) scale to millions.

## When to use / scenarios
- Learning: a non-linear embedding built from three pieces you already know: entropy, KL divergence and gradient descent ([[information-theory-for-ml]]).
- Interviews: "what does perplexity do", "why Student-t", "can you read distances on a t-SNE plot".
- Exploring embeddings (sentence vectors, CNN features, single-cell expression) to check that labels form groups, spot mislabelled points, or find sub-clusters.
- Not for: features fed into a downstream model (no `transform` for new points; use PCA or UMAP); reading global structure, cluster sizes or between-cluster distances; very large n with exact t-SNE (use Barnes-Hut, openTSNE or UMAP, see [[dimensionality-reduction]]).

## Setup & code
`pip install numpy scikit-learn`. Exact O(n²) version: about 1.5 minutes on CPU for 800 points, including scikit-learn's run.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def sq_dists(X):
    s = (X ** 2).sum(1)
    return np.maximum(s[:, None] + s[None] - 2 * X @ X.T, 0)


def conditional_p(D, perplexity, tol=1e-5, steps=60):
    """Binary-search a Gaussian precision per point so that each row's entropy matches log(perplexity)."""
    n, target = len(D), np.log(perplexity)
    P = np.zeros((n, n))
    for i in range(n):
        d = np.delete(D[i], i)
        lo, hi, beta = 0.0, np.inf, 1.0
        for _ in range(steps):
            p = np.exp(-(d - d.min()) * beta); p /= p.sum()       # shift by min for stability
            H = -(p * np.log(p + 1e-12)).sum()
            if abs(H - target) < tol:
                break
            if H > target:                                         # too flat -> narrower Gaussian
                lo, beta = beta, beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi, beta = beta, (beta + lo) / 2
        P[i, np.arange(n) != i] = p
    return P


def tsne(X, dims=2, perplexity=30, iters=1000, lr=200.0, exaggeration=12.0, seed=0):
    n = len(X)
    P = conditional_p(sq_dists(X), perplexity)
    P = (P + P.T) / (2 * n)                                        # symmetric joint p_ij, sums to 1
    P = np.maximum(P, 1e-12)
    Y = np.random.default_rng(seed).normal(scale=1e-4, size=(n, dims))
    vel, gains = np.zeros_like(Y), np.ones_like(Y)
    kl = []
    for it in range(iters):
        early = it < 250
        num = 1 / (1 + sq_dists(Y)); np.fill_diagonal(num, 0)     # Student-t (1 dof) kernel
        Q = np.maximum(num / num.sum(), 1e-12)
        PQ = (exaggeration if early else 1) * P - Q
        grad = 4 * ((PQ * num)[:, :, None] * (Y[:, None] - Y[None])).sum(1)
        gains = np.where(np.sign(grad) != np.sign(vel), gains + 0.2, gains * 0.8).clip(0.01)
        vel = (0.5 if early else 0.8) * vel - lr * gains * grad
        Y += vel
        Y -= Y.mean(0)
        if it % 250 == 249 or it == 0:
            kl.append((it + 1, (P * np.log(P / Q)).sum()))
    return Y, kl


X, y = load_digits(return_X_y=True)
X, y = X[:800], y[:800]
X = PCA(30, random_state=0).fit_transform(X)                       # usual pre-step: PCA to ~30-50 dims

Y, kl = tsne(X)
print("KL(P||Q) at iteration:", ", ".join(f"{i}: {v:.3f}" for i, v in kl))
Y_sk = TSNE(perplexity=30, init="random", random_state=0).fit_transform(X)
Y_pca = PCA(2).fit_transform(X)
for name, E in [("t-SNE ours", Y), ("t-SNE sklearn", Y_sk), ("PCA 2-D", Y_pca)]:
    acc = cross_val_score(KNeighborsClassifier(5), E, y, cv=5).mean()
    print(f"{name:14s} trustworthiness(k=10) {trustworthiness(X, E, n_neighbors=10):.3f} | 5-NN accuracy in 2-D {acc:.3f}")

# cluster sizes in the map do not track how spread out each class is in the data
spread = lambda E: np.array([np.linalg.norm(E[y == k] - E[y == k].mean(0), axis=1).mean() for k in range(10)])
s_in, s_out = spread(X), spread(Y)
print(f"cluster spread max/min ratio: input space {s_in.max() / s_in.min():.2f} | t-SNE map {s_out.max() / s_out.min():.2f}")
```

Output (numpy 2.5, scikit-learn 1.9):
```
KL(P||Q) at iteration: 1: 3.212, 250: 2.632, 500: 0.457, 750: 0.443, 1000: 0.441
t-SNE ours     trustworthiness(k=10) 0.993 | 5-NN accuracy in 2-D 0.912
t-SNE sklearn  trustworthiness(k=10) 0.994 | 5-NN accuracy in 2-D 0.924
PCA 2-D        trustworthiness(k=10) 0.800 | 5-NN accuracy in 2-D 0.512
cluster spread max/min ratio: input space 1.63 | t-SNE map 4.17
```

The KL divergence falls slowly during early exaggeration (the map is optimising a scaled `P`) and then drops to 0.44 once the real `P` is used. Trustworthiness measures how many of each point's 10 map neighbours are true neighbours in the 30-D input. The from-scratch map scores 0.993 against scikit-learn's 0.994, and a 5-NN classifier on the 2-D coordinates gets 91% vs 92%. Linear PCA to 2-D keeps only 0.800 and 51%, because the digit classes are not linearly separable in two directions. In the data, the widest digit class is 1.63× as spread out as the tightest; on the map the ratio is 4.17. Map cluster sizes come from the optimisation, not from the data's variance, so do not read them literally.

## Choosing / trade-offs
- **Perplexity.** Low values (5-10) show local detail and can split real clusters; high values (50-100) show more global arrangement and cost more. Try a few values; the defaults (30) suit a few hundred to a few thousand points.
- **t-SNE vs UMAP.** UMAP is faster, keeps somewhat more global structure, and can `transform` new points. t-SNE often gives cleaner local clusters. Both distort distances and densities.
- **t-SNE vs PCA.** PCA is linear, deterministic, invertible and keeps global variance; use it for preprocessing and as the init for t-SNE (`init="pca"`, scikit-learn's default) ([[pca-from-scratch]]).
- **Exact vs Barnes-Hut vs FFT.** Exact for n under about 2,000 or for teaching; Barnes-Hut (`method="barnes_hut"`) up to around 100k; openTSNE's FFT method beyond that.

## Gotchas
- Distances between clusters and cluster sizes are not meaningful (1.63 vs 4.17 above). Only "these points are neighbours" is reliable.
- Random noise can look like clusters at low perplexity. Check that a structure survives several perplexities and seeds before believing it.
- t-SNE has no `transform`. Rerunning on new data gives a different map; for a fixed projection of new points use PCA, UMAP or a parametric model.
- Run PCA to 30-50 dimensions first: it removes noise and makes the O(n²) distance step cheap.
- Too few iterations leave a half-formed map. Check the KL curve has flattened (here it moves only 0.016 after iteration 500).
- Scale features before t-SNE when they have different units; the Gaussian uses plain Euclidean distance ([[feature-scaling-and-normalization]]).

## Related
- [[dimensionality-reduction]] - PCA, UMAP, t-SNE and friends with library code.
- [[pca-from-scratch]] - the linear baseline and the usual pre-step.
- [[information-theory-for-ml]] - entropy, perplexity and KL divergence used here.
- [[distance-metrics-and-similarity]] - the input-space distances t-SNE starts from.
- [[word2vec-skip-gram-from-scratch-numpy]] - embeddings that are commonly visualised with t-SNE.
- [[curse-of-dimensionality]] - why distances in high dimensions need care.

## References
- van der Maaten and Hinton (2008), "Visualizing Data using t-SNE", JMLR 9: https://www.jmlr.org/papers/v9/vandermaaten08a.html
- scikit-learn `TSNE`: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
- Wattenberg, Viégas and Johnson (2016), "How to Use t-SNE Effectively", Distill: https://distill.pub/2016/misread-tsne/
