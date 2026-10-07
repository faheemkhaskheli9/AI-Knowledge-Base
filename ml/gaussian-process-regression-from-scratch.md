---
title: Gaussian process regression from scratch (RBF kernel, Cholesky, marginal likelihood)
category: ml
tags: [gaussian-process, gp-regression, kernel, rbf-kernel, marginal-likelihood, uncertainty, bayesian, cholesky, numpy, from-scratch, ml-basics]
use_cases:
  - "implement exact Gaussian process regression with an RBF kernel in NumPy"
  - "get predictions with honest error bars that widen away from the data"
  - "choose kernel hyperparameters by maximising the log marginal likelihood"
  - "explain GP posterior mean/variance, kernels and O(n^3) cost in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://gaussianprocess.org/gpml/chapters/RW2.pdf
  - https://scikit-learn.org/stable/modules/gaussian_process.html
  - https://distill.pub/2019/visual-exploration-gaussian-processes/
---

# Gaussian process regression from scratch (RBF kernel, Cholesky, marginal likelihood)

## Summary
A Gaussian process (GP) puts a prior over functions. A kernel says how strongly the function values at two inputs are correlated. Conditioning that prior on noisy observations gives a closed-form posterior: a mean prediction and a variance at every test point. The NumPy version below fits 16 noisy points of `sin(3x) + 0.5x` that have a deliberate gap. It picks the kernel length scale, signal variance and noise by maximising the log marginal likelihood, and recovers the true noise level (0.04) exactly. Inside the data its error bar is about 0.17; in the gap and beyond the data it reverts to the prior mean with an error bar of about 1.4. That widening is the main reason to use a GP. Exact inference costs O(n³), so plain GPs stop at a few thousand points.

## Key concepts
- **Prior.** `f ~ GP(0, k(x, x'))`. Any finite set of function values is jointly Gaussian with covariance `K_ij = k(x_i, x_j)`.
- **RBF kernel.** `k(x, x') = σ² exp(−‖x − x'‖² / 2ℓ²)`. Length scale `ℓ` sets how far correlation reaches (wiggliness); `σ²` sets the amplitude.
- **Noisy observations.** `y = f(x) + ε`, `ε ~ N(0, σ_n²)`, so the training covariance is `K + σ_n² I`.
- **Posterior.** `mean = K*ᵀ (K + σ_n² I)⁻¹ y`, `var = k** − K*ᵀ (K + σ_n² I)⁻¹ K*`. The variance does not depend on `y`, only on where the data are.
- **Cholesky.** Factor `K + σ_n² I = L Lᵀ` once (O(n³)); then each solve is two triangular solves. Never form the inverse.
- **Log marginal likelihood.** `log p(y) = −½ yᵀ α − Σ log L_ii − (n/2) log 2π`, with `α = (K + σ_n² I)⁻¹ y`. It trades data fit against complexity, so maximising it picks hyperparameters without a validation split.
- **Predictive vs latent variance.** The band for a new noisy observation adds `σ_n²` to the latent function variance.

## When to use / scenarios
- Learning: Bayesian inference with a closed-form posterior; the kernel trick from [[svm-from-scratch]] seen as a covariance.
- Interviews: "what does the length scale do", "why does GP uncertainty grow away from data", "how do you fit hyperparameters", "why O(n³)".
- Small-data regression where error bars matter: calibration curves, sensor interpolation, geostatistics (kriging), physical experiments.
- Bayesian optimisation: the GP is the surrogate that drives hyperparameter search and experiment design ([[hyperparameter-tuning]]).
- Not for: more than ~10k points without approximations, high-dimensional raw inputs (images, text), or heavy-tailed noise. Use gradient boosting or a neural net there, with [[conformal-prediction-and-uncertainty]] for intervals.

## Setup & code
`pip install numpy`. Runs in under a second on CPU.

```python
import numpy as np


def rbf(A, B, length, var):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return var * np.exp(-0.5 * d2 / length**2)


def gp_fit_predict(X, y, Xs, length, var, noise):
    """Exact GP regression via Cholesky. Returns posterior mean, sd and log marginal likelihood."""
    K = rbf(X, X, length, var) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))     # K^-1 y without forming the inverse
    Ks = rbf(X, Xs, length, var)
    mean = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var_s = var - (v**2).sum(0)                              # diag of Kss - Ks^T K^-1 Ks
    lml = -0.5 * y @ alpha - np.log(np.diag(L)).sum() - 0.5 * len(X) * np.log(2 * np.pi)
    return mean, np.sqrt(np.maximum(var_s, 1e-12)), lml


rng = np.random.default_rng(0)
f = lambda x: np.sin(3 * x) + 0.5 * x
X = np.sort(rng.uniform(-3, 3, 25))[:, None]
X = X[(X[:, 0] < 0) | (X[:, 0] > 1.5)]                      # leave a gap in the data
y = f(X[:, 0]) + rng.normal(0, 0.2, len(X))
Xs = np.linspace(-4, 4, 401)[:, None]
noise_true = 0.2**2

# hyperparameters by maximising the log marginal likelihood over a small grid
best = max(((gp_fit_predict(X, y, Xs, l, v, n)[2], l, v, n)
            for l in [0.1, 0.3, 0.5, 1.0, 2.0] for v in [0.5, 1.0, 2.0, 4.0] for n in [0.01, 0.04, 0.1]))
lml, l, v, n = best
print(f"n train {len(X)} | best by marginal likelihood: length {l}, var {v}, noise {n} (true noise {noise_true}) | LML {lml:.2f}")

for name, (l_, v_, n_) in {"fitted": (l, v, n), "length too short 0.1": (0.1, v, n), "length too long 2.0": (2.0, v, n)}.items():
    mean, sd, lml_ = gp_fit_predict(X, y, Xs, l_, v_, n_)
    inside = (Xs[:, 0] > -3) & (Xs[:, 0] < 3)
    rmse = np.sqrt(np.mean((mean[inside] - f(Xs[inside, 0])) ** 2))
    print(f"{name:21s}: LML {lml_:7.2f} | RMSE vs true f on [-3,3] {rmse:.3f}")

mean, sd, _ = gp_fit_predict(X, y, Xs, l, v, n)
for x0 in [-2.0, 0.75, 3.8]:
    i = np.argmin(np.abs(Xs[:, 0] - x0))
    print(f"x={x0:5.2f}: mean {mean[i]:6.3f} true {f(x0):6.3f} sd {sd[i]:.3f}")

# calibration: share of fresh noisy test points inside the 95% predictive band (latent sd + noise)
Xt = rng.uniform(-3, 3, 2000)[:, None]
yt = f(Xt[:, 0]) + rng.normal(0, 0.2, 2000)
mt, st, _ = gp_fit_predict(X, y, Xt, l, v, n)
print(f"share of test points inside 95% predictive band: {np.mean(np.abs(yt - mt) < 1.96 * np.sqrt(st**2 + n)):.3f}")
```

Output (numpy 2.5):
```
n train 16 | best by marginal likelihood: length 0.5, var 2.0, noise 0.04 (true noise 0.04000000000000001) | LML -11.14
fitted               : LML  -11.14 | RMSE vs true f on [-3,3] 0.547
length too short 0.1 : LML  -20.61 | RMSE vs true f on [-3,3] 0.797
length too long 2.0  : LML  -32.88 | RMSE vs true f on [-3,3] 0.997
x=-2.00: mean -0.818 true -0.721 sd 0.169
x= 0.75: mean -0.073 true  1.153 sd 1.400
x= 3.80: mean -0.045 true  0.981 sd 1.407
share of test points inside 95% predictive band: 0.990
```

The marginal likelihood picks length 0.5 and noise 0.04, which is the true noise variance, from 16 points and no validation set. Its ranking agrees with the true error: the fitted kernel has the highest LML (−11.1) and the lowest RMSE (0.55). A length scale of 0.1 overfits (each point is its own bump, RMSE 0.80) and 2.0 underfits (a smooth line through a sine, RMSE 1.00). The point-wise rows show the property that matters. At x = −2, inside the data, the prediction is off by 0.1 with sd 0.17. At x = 0.75, in the gap, and x = 3.8, past the data, the mean falls back to the prior mean 0 and is off by about 1. The sd rises to 1.4, so the model says it does not know. The 95% predictive band covers 99% of fresh test points. That is slightly conservative, mostly because of the wide gap region.

## Choosing / trade-offs
- **Kernel.** RBF for smooth functions; Matérn 3/2 or 5/2 for rougher, more realistic ones (the scikit-learn and BoTorch default is often Matérn 5/2); periodic for seasonality; add or multiply kernels to combine trends. One length scale per input (ARD) also ranks feature relevance.
- **Hyperparameter fitting.** Use gradient-based LML maximisation with restarts (scikit-learn `GaussianProcessRegressor(n_restarts_optimizer=...)`, GPyTorch). A grid like the one above is only for teaching. Full Bayesian treatment (MCMC over hyperparameters) helps with very few points.
- **Scale.** Exact GPs fit up to a few thousand points. Beyond that use sparse/inducing-point GPs (SVGP), structured kernels (KISS-GP), or GPyTorch's conjugate-gradient solvers on GPU.
- **Non-Gaussian likelihoods.** Classification and counts need approximate inference (Laplace, EP, variational); the closed form above is regression-only.
- **Library.** `scikit-learn` for small problems, `GPyTorch` for scale and GPUs, `BoTorch` for Bayesian optimisation, `GPflow` in the TensorFlow world.

## Gotchas
- Always add a noise or jitter term (`1e-6` at least) to the diagonal. Without it Cholesky fails on near-duplicate inputs.
- Standardise inputs and targets. The zero-mean prior means far-from-data predictions return to 0, which is wrong if `y` has a large offset; subtract the mean (or model a linear trend) first.
- The LML surface is multi-modal. One local optimum explains everything as noise (long length scale, large `σ_n`); restart from several initialisations.
- Report the right band: latent `f` variance for "where is the function", latent plus `σ_n²` for "where will the next measurement land".
- Do not invert `K` with `np.linalg.inv`. It is slower and less stable than Cholesky solves.
- The O(n²) memory of `K` bites before the O(n³) time does: 50k points is 20 GB in float64.
- Uncertainty is only as good as the kernel. A stationary RBF cannot see a sudden jump or a change in noise level and will be confidently wrong there.

## Related
- [[bayesian-and-gaussian-processes]] - library usage, Bayesian optimisation and GP classification.
- [[kernel-methods-and-density-estimation]] - the same kernels in SVMs, kernel ridge and KDE.
- [[svm-from-scratch]] - the kernel trick from the margin side.
- [[kalman-filter-from-scratch]] - Gaussian conditioning applied step by step over time.
- [[conformal-prediction-and-uncertainty]] - distribution-free intervals for models without a built-in posterior.
- [[hyperparameter-tuning]] - GP surrogates inside Bayesian optimisation.

## References
- Rasmussen and Williams, Gaussian Processes for Machine Learning (MIT Press, 2006), Chapter 2 and Algorithm 2.1: https://gaussianprocess.org/gpml/chapters/RW2.pdf
- scikit-learn user guide, Gaussian Processes: https://scikit-learn.org/stable/modules/gaussian_process.html
- Görtler, Kehlbeck and Deussen, "A Visual Exploration of Gaussian Processes", Distill (2019): https://distill.pub/2019/visual-exploration-gaussian-processes/
