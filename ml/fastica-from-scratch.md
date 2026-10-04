---
title: FastICA from scratch (blind source separation, whitening, non-Gaussianity)
category: ml
tags: [ica, fastica, independent-component-analysis, blind-source-separation, whitening, non-gaussianity, kurtosis, cocktail-party, numpy, from-scratch, ml-basics]
use_cases:
  - "implement FastICA in NumPy to unmix linearly mixed signals"
  - "solve the cocktail-party problem: recover sources from several microphone mixtures"
  - "see why PCA decorrelates but does not separate sources, and why Gaussian sources cannot be separated"
  - "explain whitening, non-Gaussianity and ICA ambiguities in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1016/S0893-6080(00)00026-5
  - https://scikit-learn.org/stable/modules/decomposition.html#independent-component-analysis-ica
  - https://www.cs.helsinki.fi/u/ahyvarin/papers/fastica.shtml
---

# FastICA from scratch (blind source separation, whitening, non-Gaussianity)

## Summary
Independent component analysis (ICA) assumes the observed signals are linear mixtures `X = A S` of statistically independent, non-Gaussian sources, and recovers both the sources and the unmixing matrix from the mixtures alone. FastICA does it in two steps. It whitens the data, so the remaining unknown is a rotation. It then finds the rotation that makes each output as non-Gaussian as possible, using a fixed-point iteration. Below, three signals (sine, square wave, sawtooth) are mixed by a random 3×3 matrix. In the raw mixtures, each source's best match has a correlation of 0.62 to 0.80. PCA reaches 0.63 to 0.82, because decorrelating is not separating. FastICA recovers all three at 0.998 or better, in 4 iterations, up to order and sign. With Gaussian sources the same code fails (0.72 to 0.84). Non-Gaussianity is the signal ICA relies on.

## Key concepts
- **Model.** `X = A S`, with `A` unknown and square (or reduce `X` with PCA to the number of sources first). Goal: `W ≈ A⁻¹`, so `S ≈ W X`.
- **Whitening.** Center and transform so `cov(Z) = I` (via the eigendecomposition of the covariance, as in PCA). After whitening, the unmixing matrix is orthogonal, which leaves only a rotation to find.
- **Why non-Gaussianity.** By the central limit theorem a mixture of independent signals is more Gaussian than the signals themselves. Making an output maximally non-Gaussian therefore isolates one source.
- **Contrast function.** FastICA approximates negentropy with `G(u) = log cosh(u)` (derivative `g = tanh`). The fixed-point update per component is `w ← E[z g(wᵀz)] − E[g'(wᵀz)] w`, followed by normalisation.
- **Decorrelation.** Without it, all rows converge to the same source. Symmetric FastICA orthogonalises all rows at once with `W ← (W Wᵀ)^(−1/2) W` (here via SVD). Deflation extracts sources one by one with Gram-Schmidt.
- **Ambiguities.** Sources come back in any order, with any sign and scale. `W A` is a scaled permutation matrix, not the identity.

## When to use / scenarios
- Learning: the gap between "uncorrelated" (PCA) and "independent" (ICA); higher-order statistics as a learning signal.
- Interviews: "ICA vs PCA", "why can't ICA separate Gaussian sources", "what does whitening buy you", "what are ICA's ambiguities".
- Signal processing: removing eye-blink and heartbeat artefacts from EEG/MEG (MNE-Python's standard workflow), fMRI component analysis, separating sensor signals that physically add, and feature extraction from image patches.
- Not for: real reverberant audio rooms, where sound arrives with delays and echoes (a convolutive mixture; use frequency-domain ICA or neural source separation), more sources than sensors, or Gaussian-like sources.

## Setup & code
`pip install numpy`. Runs in under a second on CPU.

```python
import numpy as np


def whiten(X):
    """Center and whiten: rows of X are signals. Returns Z with cov(Z) = I and the whitening matrix."""
    X = X - X.mean(1, keepdims=True)
    d, E = np.linalg.eigh(np.cov(X))
    K = E @ np.diag(d ** -0.5) @ E.T
    return K @ X, K


def fastica(X, n_iter=200, tol=1e-8, seed=0):
    """Symmetric FastICA with the log-cosh contrast (g = tanh). Returns sources and unmixing matrix."""
    Z, K = whiten(X)
    n, m = Z.shape
    W = np.random.default_rng(seed).normal(size=(n, n))
    for it in range(n_iter):
        G = np.tanh(W @ Z)
        W_new = G @ Z.T / m - np.diag((1 - G**2).mean(1)) @ W     # fixed-point step, one row per component
        u, s, vt = np.linalg.svd(W_new)
        W_new = u @ vt                                            # symmetric decorrelation: (W W^T)^-1/2 W
        converged = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1)) < tol
        W = W_new
        if converged:
            break
    return W @ Z, W @ K, it + 1


def match(S_true, S_est):
    """|correlation| of each true source with its best-matching estimate (ICA order and sign are arbitrary)."""
    C = np.abs(np.corrcoef(S_true, S_est)[:3, 3:])
    return C.max(1)


rng = np.random.default_rng(0)
t = np.linspace(0, 8, 4000)
S = np.vstack([np.sin(2 * t),                         # sine
               np.sign(np.sin(3 * t)),                # square wave
               2 * ((t * 1.5) % 1) - 1])              # sawtooth
S += 0.02 * rng.normal(size=S.shape)
A = rng.uniform(0.5, 2.0, (3, 3))                     # unknown mixing: 3 microphones
X = A @ S

S_ica, W, iters = fastica(X)
d, E = np.linalg.eigh(np.cov(X))
S_pca = E.T @ (X - X.mean(1, keepdims=True))          # PCA: decorrelated, not independent
print(f"FastICA converged in {iters} iterations")
print(f"best |corr| with true sources, raw mixtures: {np.round(match(S, X), 3).tolist()}")
print(f"best |corr| with true sources, PCA:          {np.round(match(S, S_pca), 3).tolist()}")
print(f"best |corr| with true sources, FastICA:      {np.round(match(S, S_ica), 3).tolist()}")
P = W @ A                                             # should be a scaled permutation matrix
print("W @ A (normalised rows), ~ permutation:\n", np.round(P / np.abs(P).max(1, keepdims=True), 2))

# Gaussian sources: every rotation of white Gaussian data looks the same, so ICA cannot identify them
Sg = rng.normal(size=(3, 4000))
Sg_ica, _, it_g = fastica(A @ Sg)
print(f"Gaussian sources: FastICA |corr| {np.round(match(Sg, Sg_ica), 3).tolist()} after {it_g} iterations")
kurt = lambda s: ((s - s.mean()) ** 4).mean() / s.var() ** 2 - 3
print(f"excess kurtosis: sine {kurt(S[0]):.2f}, square {kurt(S[1]):.2f}, saw {kurt(S[2]):.2f}, gaussian {kurt(Sg[0]):.2f}")
```

Output (numpy 2.5):
```
FastICA converged in 4 iterations
best |corr| with true sources, raw mixtures: [0.624, 0.802, 0.645]
best |corr| with true sources, PCA:          [0.632, 0.752, 0.817]
best |corr| with true sources, FastICA:      [0.998, 1.0, 1.0]
W @ A (normalised rows), ~ permutation:
 [[ 1.   -0.04  0.01]
 [ 0.04 -1.    0.  ]
 [ 0.    0.    1.  ]]
Gaussian sources: FastICA |corr| [0.782, 0.837, 0.722] after 21 iterations
excess kurtosis: sine -1.36, square -1.99, saw -1.20, gaussian 0.01
```

Each mixture is a blend, so its best correlation with any one source is 0.62 to 0.80. PCA rotates the data onto uncorrelated axes of maximal variance, which are still blends (0.63 to 0.82): any further rotation of whitened data is also uncorrelated, so second-order statistics cannot single out the sources. FastICA chooses among those rotations by non-Gaussianity and recovers every source at 0.998 or better in 4 iterations. The product `W A` is close to a signed permutation; the −1 in the middle row is the sign ambiguity, and the small off-diagonal 0.04 is leakage from the added noise. All three sources have strongly negative excess kurtosis (sub-Gaussian), which the log-cosh contrast handles as well as super-Gaussian (spiky) sources. Mixtures of Gaussian sources are themselves Gaussian, and white Gaussian data looks the same under every rotation, so FastICA returns an arbitrary rotation (0.72 to 0.84) even though it converges.

## Choosing / trade-offs
- **ICA vs PCA.** PCA finds orthogonal directions of maximum variance and is the right tool for compression. ICA finds statistically independent sources and is the right tool for separating additive generators. PCA is usually the whitening and dimension-reduction step before ICA.
- **Contrast.** `logcosh` is a robust general-purpose default; `exp(−u²/2)` suits highly super-Gaussian sources; kurtosis (`u³`) is fast but sensitive to outliers.
- **Symmetric vs deflation.** Symmetric estimates all components in parallel and avoids error accumulation; deflation lets you stop after the first k components.
- **Library.** `sklearn.decomposition.FastICA` for general use; MNE-Python's ICA (FastICA, Infomax or Picard) for EEG/MEG; Picard converges faster and more reliably on real data.
- **Number of components.** It cannot exceed the number of sensors. Reduce with PCA first when there are more sensors than sources, keeping enough variance (MNE recommends dropping near-zero-variance dimensions).

## Gotchas
- Always center and whiten first; the fixed-point update assumes white data.
- Do not compare recovered and true sources by position or sign. Match by absolute correlation (`match` above).
- Results depend on the random initial `W`; for real data run several seeds and check that the components are stable (ICASSO-style).
- Near-singular covariance (duplicated or rank-deficient channels) makes whitening blow up. Drop or reduce those dimensions with PCA.
- ICA assumes instantaneous linear mixing with a fixed `A`. Delays, echoes or moving sources break the model.
- Time order is ignored: ICA treats samples as i.i.d. Methods such as SOBI use temporal correlation instead and can separate Gaussian sources with different spectra.
- Recovered scale is arbitrary (unit variance). Rescale using the columns of `A_est = W⁻¹` when amplitudes matter.

## Related
- [[pca-from-scratch]] - the eigendecomposition used for whitening, and what decorrelation alone gives.
- [[dimensionality-reduction]] - PCA, ICA, NMF and manifold methods with scikit-learn.
- [[information-theory-for-ml]] - entropy and negentropy, the quantities FastICA approximates.
- [[tsne-from-scratch]] - another non-linear way to look for structure that PCA misses.
- [[autoencoders-and-self-supervised-learning]] - learned, non-linear representations when the linear mixing model does not fit.

## References
- Hyvärinen and Oja (2000), "Independent component analysis: algorithms and applications", Neural Networks 13: https://doi.org/10.1016/S0893-6080(00)00026-5
- scikit-learn user guide, Independent component analysis: https://scikit-learn.org/stable/modules/decomposition.html#independent-component-analysis-ica
- FastICA papers and original package, University of Helsinki: https://www.cs.helsinki.fi/u/ahyvarin/papers/fastica.shtml
