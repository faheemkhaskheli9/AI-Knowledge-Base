---
title: PCA from scratch (centring, SVD vs covariance eigendecomposition, explained variance, reconstruction)
category: ml
tags: [pca, principal-component-analysis, svd, eigendecomposition, explained-variance, dimensionality-reduction, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement PCA from scratch for learning or an interview"
  - "understand how PCA relates to the SVD and the covariance matrix"
  - "pick the number of components from explained variance"
  - "project, reconstruct and measure reconstruction error with PCA"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/decomposition.html#pca
  - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
  - https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
---

# PCA from scratch (centring, SVD vs covariance eigendecomposition, explained variance, reconstruction)

## Summary
Principal component analysis finds orthogonal directions along which centred data varies most, then projects onto the top `k` of them. In practice it is one SVD: centre `X`, compute `X_c = U S Vᵀ`, and the rows of `Vᵀ` are the components with variances `S² / (n − 1)`. Ten lines of NumPy match scikit-learn's `PCA` exactly, up to the sign of each component.

## Key concepts
- **Centring.** Subtract the column means first. Without it the first component points at the mean, not along the spread.
- **Covariance view.** Components are the eigenvectors of the covariance `Σ = X_cᵀ X_c / (n − 1)`, sorted by eigenvalue. Each eigenvalue is the variance along its component.
- **SVD view.** With `X_c = U S Vᵀ`, `Σ = V (S² / (n − 1)) Vᵀ`. So the right singular vectors are the components and `S² / (n − 1)` are the eigenvalues. SVD never forms `Σ`, which is more accurate numerically (forming `XᵀX` squares the condition number).
- **Projection and reconstruction.** Scores `Z = X_c Wᵀ` (shape `(n, k)`), reconstruction `X̂ = Z W + μ`. PCA gives the lowest squared reconstruction error of any rank-`k` linear projection (Eckart-Young).
- **Explained variance ratio.** `λᵢ / Σλ`. The cumulative sum tells you how much variance `k` components keep.
- **Sign ambiguity.** `v` and `−v` are equally valid components. Libraries pick a sign convention (scikit-learn flips each so its largest-magnitude loading is positive). Do the same if you compare runs.
- **Whitening.** Dividing scores by `√λ` gives unit-variance, uncorrelated features. Useful before some models, but it amplifies the noise in low-variance directions.

## When to use / scenarios
- Learning: the cleanest link between linear algebra (SVD, eigenvectors) and an ML method ([[math-for-machine-learning]]).
- Interviews: deriving PCA from the covariance, explaining SVD vs `eigh`, choosing `k`.
- Preprocessing: compress correlated features, denoise, or speed up distance-based methods ([[curse-of-dimensionality]], [[k-means-from-scratch]]).
- Visualisation: a 2-D PCA plot is a fast, deterministic first look before t-SNE/UMAP ([[dimensionality-reduction]]).
- Production: use `sklearn.decomposition.PCA` (`svd_solver="randomized"` for large data, `IncrementalPCA` for data that does not fit in memory).
- Not for: non-linear structure (manifolds, clusters on curves). Use kernel PCA, UMAP or an autoencoder ([[autoencoders-and-self-supervised-learning]]).

## Setup & code
`pip install numpy scikit-learn`. Runs in under a second on CPU.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


def pca_fit(X, k):
    mu = X.mean(0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)   # Xc = U S Vt
    signs = np.sign(Vt[np.arange(k), np.abs(Vt[:k]).argmax(1)])
    comps = Vt[:k] * signs[:, None]                      # deterministic sign
    var = S ** 2 / (len(X) - 1)                          # eigenvalues of the covariance
    return mu, comps, var[:k], var[:k] / var.sum()


def pca_eig(X, k):
    """Same thing via the covariance eigendecomposition (fine for small d)."""
    Xc = X - X.mean(0)
    vals, vecs = np.linalg.eigh(Xc.T @ Xc / (len(X) - 1))  # ascending order
    return vals[::-1][:k], vecs[:, ::-1][:, :k].T


X = load_digits().data                                   # (1797, 64)
mu, W, var, ratio = pca_fit(X, 10)
Z = (X - mu) @ W.T                                       # project: (n, 10)
X_hat = Z @ W + mu                                       # reconstruct

sk = PCA(n_components=10).fit(X)
print("explained var ratio (ours):", ratio[:3].round(4))
print("explained var ratio (sk):  ", sk.explained_variance_ratio_[:3].round(4))
print("max |component diff| up to sign:",
      np.abs(np.abs(W) - np.abs(sk.components_)).max().round(8))
vals, _ = pca_eig(X, 10)
print("eigh eigenvalues == SVD S^2/(n-1):", np.allclose(vals, var))
print(f"10 comps keep {ratio.sum():.1%} of variance, "
      f"recon MSE={((X - X_hat) ** 2).mean():.3f}")

_, _, _, full = pca_fit(X, 64)
k95 = int(np.searchsorted(np.cumsum(full), 0.95) + 1)
print("components for 95% variance:", k95)
```

Output (numpy 2.5, scikit-learn 1.9):
```
explained var ratio (ours): [0.1489 0.1362 0.1179]
explained var ratio (sk):   [0.1489 0.1362 0.1179]
max |component diff| up to sign: 0.0
eigh eigenvalues == SVD S^2/(n-1): True
10 comps keep 73.8% of variance, recon MSE=4.914
components for 95% variance: 29
```

The SVD and covariance routes give the same eigenvalues, and both match scikit-learn. On the 64-pixel digits, 10 components keep 73.8% of the variance and 29 keep 95%: less than half the dimensions for almost all of the signal.

## Choosing / trade-offs
- **SVD vs `eigh` of the covariance.** SVD is more stable and works directly on `X`. `eigh` on the `d × d` covariance is cheaper when `n ≫ d` and `d` is small, and is fine in float64.
- **Full vs randomized vs incremental.** Full SVD costs `O(n d min(n, d))`. Randomized SVD gets the top `k` much faster when `k ≪ d`. `IncrementalPCA` streams mini-batches for data larger than memory.
- **Choosing k.** Use a cumulative explained-variance threshold (90-99%), the elbow of the scree plot, or best of all the downstream metric (classifier accuracy, retrieval recall) under cross-validation ([[model-selection-and-comparison]]).
- **Standardise or not.** On features with different units, standardise first (PCA on the correlation matrix). Otherwise the feature with the largest raw variance owns the first component. On homogeneous features (pixels, same-unit sensors), centring alone is usually right ([[feature-scaling-and-normalization]]).

## Gotchas
- Fit the mean and components on the training split only, then apply them to validation/test. Fitting PCA on all data is leakage ([[data-leakage-and-validation-splits]]).
- The sign of each component can flip between library versions, solvers or data subsets. Never compare raw loadings or scores across runs without aligning signs.
- `np.linalg.eigh` returns eigenvalues in ascending order; `np.linalg.svd` returns singular values in descending order. Reversing one and not the other is a common bug.
- Variance is not importance for the label. A low-variance direction can be the one that separates classes. PCA is unsupervised; LDA or a supervised method may be better for classification.
- PCA is sensitive to outliers because it maximises squared deviations. Clip or use robust PCA when a few points dominate.
- Use `ddof` consistently: `S² / (n − 1)` matches `np.cov` and scikit-learn's `explained_variance_`. Ratios are unaffected.

## Related
- [[dimensionality-reduction]] - PCA vs t-SNE, UMAP and other reducers, and when to use each.
- [[math-for-machine-learning]] - eigenvectors, SVD and the linear algebra behind PCA.
- [[linear-regression-from-scratch]] - the other SVD/least-squares from-scratch model.
- [[k-means-from-scratch]] - often run on PCA-reduced features.
- [[curse-of-dimensionality]] - why reducing dimensions first helps distance-based methods.
- [[autoencoders-and-self-supervised-learning]] - the non-linear generalisation (a linear autoencoder learns the PCA subspace).

## References
- scikit-learn, Principal component analysis: https://scikit-learn.org/stable/modules/decomposition.html#pca
- scikit-learn PCA API: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- NumPy `linalg.svd`: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
