---
title: Gaussian mixture model EM from scratch (multivariate, log-space E-step, full covariance, BIC, checked against scikit-learn)
category: ml
tags: [gaussian-mixture, gmm, expectation-maximization, em-algorithm, log-sum-exp, cholesky, soft-clustering, bic, numpy, scikit-learn, from-scratch, ml-basics]
use_cases:
  - "implement a multivariate Gaussian mixture model with EM from scratch"
  - "write a numerically stable E-step with log-sum-exp and Cholesky"
  - "check a hand-written GMM against sklearn GaussianMixture"
  - "compute BIC by hand to choose the number of components"
status: stable
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/mixture.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
  - https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/
  - https://www.jstor.org/stable/2984875
---

# Gaussian mixture model EM from scratch (multivariate, log-space E-step, full covariance, BIC, checked against scikit-learn)

## Summary
A Gaussian mixture model explains the data as `K` Gaussians with weights `π`, means `μ` and covariances `Σ`, and EM fits it by alternating two steps: the E-step computes each point's responsibility `rᵢₖ = P(component k | xᵢ)`, and the M-step re-estimates `π, μ, Σ` as responsibility-weighted averages. The NumPy version below handles any dimension and full covariances, works in log space so it never underflows, and matches scikit-learn's log-likelihood to `1e-7` and its BIC for every `K`. [[gaussian-mixture-models-and-em]] covers when to use GMMs and a 1-D toy EM; this file is the complete, production-shaped version.

## Key concepts
- **Log-likelihood.** `log p(X) = Σᵢ log Σₖ πₖ N(xᵢ | μₖ, Σₖ)`. EM never decreases it, which makes it the convergence check and a free bug detector.
- **Log-space E-step.** Compute `log πₖ + log N(xᵢ | k)` for every pair, then `rᵢₖ = exp(that − logsumexp over k)`. Multiplying raw densities underflows to 0 in more than a few dimensions or for far-away points, giving `0/0`.
- **Cholesky log-density.** With `Σ = LLᵀ`, the Mahalanobis term is `‖L⁻¹(x − μ)‖²` and `log|Σ| = 2 Σ log Lᵢᵢ`. This avoids an explicit inverse and determinant and fails loudly if `Σ` is not positive definite.
- **M-step.** `Nₖ = Σᵢ rᵢₖ`, `πₖ = Nₖ/n`, `μₖ = Σᵢ rᵢₖxᵢ/Nₖ`, `Σₖ = Σᵢ rᵢₖ(xᵢ − μₖ)(xᵢ − μₖ)ᵀ/Nₖ + εI`.
- **Covariance regularisation.** The `εI` term (scikit-learn's `reg_covar`, default `1e-6`) stops a component from collapsing onto one point, where its variance goes to 0 and the likelihood to infinity.
- **Initialisation.** Start from k-means hard assignments as the first responsibilities, as scikit-learn does by default. EM only finds a local optimum, so restarts (`n_init`) matter on hard data.
- **BIC.** `−2 log p(X) + p log n`, with `p = Kd + K·d(d+1)/2 + K − 1` free parameters for full covariances. Lower is better.

## When to use / scenarios
- Learning: EM is the general recipe for latent-variable models, and the GMM is its cleanest worked example.
- Interviews: "derive the E and M steps", "why work in log space", "how is k-means a special case".
- Custom variants scikit-learn does not ship: weighted samples, fixed means, missing values, or a mixture of non-Gaussian components. Start from this loop and change the M-step.
- Production: use `sklearn.mixture.GaussianMixture` for soft clustering and density-based anomaly scores ([[gaussian-mixture-models-and-em]], [[anomaly-detection]]).
- Not for: high-dimensional raw data, where full covariances have `O(d²)` parameters per component. Reduce dimensions first or use `diag` covariances ([[pca-from-scratch]]).

## Setup & code
`pip install numpy scipy scikit-learn`. Runs in about a second on CPU.

```python
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture


def log_gauss(X, mu, cov):
    """log N(x | mu, cov) for every row of X, via a Cholesky factor (no explicit inverse)."""
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, (X - mu).T)                     # L^-1 (x - mu)
    return -0.5 * (sol ** 2).sum(0) - np.log(np.diag(L)).sum() - 0.5 * X.shape[1] * np.log(2 * np.pi)


def gmm_em(X, K, n_iter=200, tol=1e-6, reg=1e-6, seed=0):
    n, d = X.shape
    r = np.eye(K)[KMeans(K, n_init=1, random_state=seed).fit_predict(X)]   # k-means init, hard
    lls = []
    for _ in range(n_iter):
        # M-step: weighted counts, means, covariances
        nk = r.sum(0) + 1e-10
        pi = nk / n
        mu = r.T @ X / nk[:, None]
        cov = np.array([(r[:, k, None] * (X - mu[k])).T @ (X - mu[k]) / nk[k] + reg * np.eye(d)
                        for k in range(K)])
        # E-step in log space: log pi_k + log N(x | k), normalised with log-sum-exp
        logp = np.stack([np.log(pi[k]) + log_gauss(X, mu[k], cov[k]) for k in range(K)], 1)
        ll = logsumexp(logp, 1)
        r = np.exp(logp - ll[:, None])
        lls.append(ll.mean())
        if len(lls) > 1 and lls[-1] - lls[-2] < tol:
            break
    return pi, mu, cov, r, np.array(lls)


rng = np.random.default_rng(0)
X = np.vstack([
    rng.multivariate_normal([0, 0], [[1.0, 0.8], [0.8, 1.0]], 300),   # elongated
    rng.multivariate_normal([4, 0], [[0.3, 0.0], [0.0, 0.3]], 200),   # small, tight
    rng.multivariate_normal([2, 4], [[1.5, 0.0], [0.0, 0.5]], 250),
])
y = np.repeat([0, 1, 2], [300, 200, 250])

pi, mu, cov, r, lls = gmm_em(X, 3)
print(f"iterations: {len(lls)}, avg log-likelihood: {lls[0]:.4f} -> {lls[-1]:.4f}")
print("log-likelihood never decreased:", bool(np.all(np.diff(lls) >= -1e-12)))
order = np.argsort(mu[:, 0] + 10 * mu[:, 1])                          # stable component order
print("weights:", pi[order].round(3), "\nmeans:\n", mu[order].round(2))
print("cov of elongated component:\n", cov[order][0].round(2))

sk = GaussianMixture(3, tol=1e-6, max_iter=200, init_params="kmeans", random_state=0).fit(X)
print(f"sklearn avg log-likelihood: {sk.score(X):.4f}, |ours - sklearn| = {abs(sk.score(X) - lls[-1]):.1e}")
print("ARI ours vs sklearn:", round(adjusted_rand_score(sk.predict(X), r.argmax(1)), 3),
      "| ours vs truth:", round(adjusted_rand_score(y, r.argmax(1)), 3),
      "| k-means vs truth:", round(adjusted_rand_score(y, KMeans(3, n_init=10, random_state=0).fit_predict(X)), 3))

# BIC = -2 * total log-likelihood + n_params * log(n); full covariance in d dims
n, d = X.shape
for K in range(1, 6):
    ll = gmm_em(X, K)[4][-1] * n
    p = K * d + K * d * (d + 1) / 2 + K - 1
    print(f"K={K}: BIC ours {-2 * ll + p * np.log(n):8.1f}  sklearn "
          f"{GaussianMixture(K, tol=1e-6, max_iter=200, random_state=0).fit(X).bic(X):8.1f}")

uncertain = (r.max(1) < 0.9).sum()
print("points with max responsibility < 0.9:", uncertain)
```

Output (numpy 2.5, scipy 1.18, scikit-learn 1.9):
```
iterations: 8, avg log-likelihood: -3.3016 -> -3.2927
log-likelihood never decreased: True
weights: [0.399 0.267 0.335] 
means:
 [[ 0.06  0.08]
 [ 3.95 -0.04]
 [ 2.1   4.02]]
cov of elongated component:
 [[0.97 0.76]
 [0.76 0.94]]
sklearn avg log-likelihood: -3.2927, |ours - sklearn| = 6.5e-08
ARI ours vs sklearn: 1.0 | ours vs truth: 0.974 | k-means vs truth: 0.947
K=1: BIC ours   6261.5  sklearn   6261.5
K=2: BIC ours   5695.8  sklearn   5695.8
K=3: BIC ours   5051.7  sklearn   5051.7
K=4: BIC ours   5082.5  sklearn   5082.5
K=5: BIC ours   5114.0  sklearn   5114.0
points with max responsibility < 0.9: 16
```

Starting from k-means, EM converges in 8 iterations and the log-likelihood rises at every step. It recovers the generating weights (300, 200 and 250 of 750 points = 0.40, 0.27, 0.33), the means and the correlated covariance of the elongated blob (true off-diagonal 0.8, estimated 0.76). The final log-likelihood matches scikit-learn's to `7e-8` and the hard labels are identical. Because each component has its own shape, the GMM labels the true groups slightly better than k-means (ARI 0.974 vs 0.947). BIC, computed by hand, equals scikit-learn's `bic()` for every `K` and is lowest at `K = 3`, the true number. The 16 points with no responsibility above 0.9 sit where blobs overlap; that uncertainty is what soft clustering gives over k-means.

## Choosing / trade-offs
- **Covariance type.** `full` fits any ellipse but costs `d(d+1)/2` parameters per component. `diag` (axis-aligned) is the usual choice past ~10 dimensions. `tied` shares one shape across components. `spherical` is soft k-means. In this code only the `cov` line changes.
- **GMM vs k-means.** k-means is the limit of EM with equal spherical covariances and hard assignments. Use a GMM when clusters differ in shape or size, or when you need probabilities or a density ([[k-means-from-scratch]]).
- **BIC vs a Bayesian GMM.** BIC needs a fit per `K`. `BayesianGaussianMixture` with a Dirichlet-process prior fits once and shrinks unneeded components toward zero weight.
- **EM vs gradient descent.** EM needs no learning rate and is monotone. Gradient-based fitting (e.g. in PyTorch) is useful only when the mixture is part of a larger differentiable model.

## Gotchas
- Never compute responsibilities from raw densities in more than a couple of dimensions; use log densities and `logsumexp`.
- Add `reg_covar` to every covariance. Without it, a component that captures one point or a set of duplicate points gets a singular `Σ` and the Cholesky fails.
- A component can die (`Nₖ → 0`). Guard the division with a small constant, or re-initialise that component.
- Compare against scikit-learn with the same initialisation and `tol`; different starts can reach different local optima with different likelihoods.
- scikit-learn's `score()` is the **mean** log-likelihood per sample, while `bic()` uses the total. Multiply by `n` before comparing.
- Component order is arbitrary. Compare labels with ARI, and sort components (e.g. by mean) before comparing parameters.

## Related
- [[gaussian-mixture-models-and-em]] - GMM usage in scikit-learn, covariance types, BIC and anomaly detection.
- [[k-means-from-scratch]] - the hard-assignment special case.
- [[hierarchical-clustering-from-scratch]] - a deterministic, tree-shaped alternative.
- [[maximum-likelihood-and-map-estimation]] - the objective EM maximises.
- [[hidden-markov-models-and-kalman-filters]] - EM over sequences (Baum-Welch).
- [[anomaly-detection]] - low log-density as an anomaly score.

## References
- scikit-learn, Gaussian mixture models: https://scikit-learn.org/stable/modules/mixture.html
- scikit-learn `GaussianMixture` (`reg_covar`, `init_params`, `score`, `bic`): https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
- Bishop (2006), Pattern Recognition and Machine Learning, chapter 9: https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/
- Dempster, Laird, Rubin (1977), "Maximum Likelihood from Incomplete Data via the EM Algorithm": https://www.jstor.org/stable/2984875
