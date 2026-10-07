---
title: Bayesian linear regression from scratch (posterior, predictive uncertainty, evidence maximisation)
category: ml
tags: [bayesian-linear-regression, bayesian-inference, posterior, predictive-distribution, uncertainty, marginal-likelihood, evidence, empirical-bayes, model-selection, ridge-regression, numpy, from-scratch, ml-basics]
use_cases:
  - "implement Bayesian linear regression in NumPy and get a prediction with an error bar"
  - "tune the regularisation strength and noise level from the training data alone, without a validation split"
  - "pick a model (polynomial degree, basis) by marginal likelihood instead of cross-validation"
  - "explain how ridge regression, MAP estimation and the Bayesian posterior relate in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/
  - https://doi.org/10.1162/neco.1992.4.3.415
  - https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression
---

# Bayesian linear regression from scratch (posterior, predictive uncertainty, evidence maximisation)

## Summary
Bayesian linear regression puts a Gaussian prior on the weights. With Gaussian noise the posterior over the weights is also Gaussian, in closed form, so every prediction comes with a variance and no sampling is needed. The prior precision `α` and the noise precision `β` can be set by maximising the marginal likelihood (the evidence), which also scores competing models on the training data alone. In the example below, 18 noisy points from `sin(2πx)` have a gap in the middle. Plain least squares with a degree-9 polynomial overfits (test RMSE 1.14). With α and β learned from the evidence, the same degree-9 model gets 0.26, and the noise std comes out at 0.17 to 0.19 against a true 0.2. The evidence ranks degree 5 highest. The example also shows the method's weak spot: a polynomial basis is confidently wrong inside the gap and badly wrong outside the data (mean +9.0 ± 2.0 at x = 1.2, where the truth is +0.95).

## Key concepts
- **Model.** `y = φ(x)ᵀw + ε`, `ε ~ N(0, β⁻¹)`, prior `w ~ N(0, α⁻¹ I)`. The basis `φ` (here polynomial) carries all the non-linearity; the model stays linear in `w`.
- **Posterior.** `S⁻¹ = α I + β ΦᵀΦ`, `m = β S Φᵀ y`. The mean `m` is exactly the ridge-regression solution with `λ = α/β`: ridge is the MAP estimate, and the Bayesian version adds the covariance `S`.
- **Predictive distribution.** `p(y* | x*) = N(φ*ᵀm, 1/β + φ*ᵀ S φ*)`. The first term is irreducible noise. The second is parameter uncertainty, which shrinks as data accumulates near `x*`.
- **Evidence.** `log p(y | α, β)` integrates the weights out. It trades data fit against model complexity (the `log det S` term, an automatic Occam's razor), so it can compare models without a validation set.
- **Evidence maximisation (empirical Bayes, type-II maximum likelihood).** MacKay's fixed point: with `λᵢ` the eigenvalues of `β ΦᵀΦ`, `γ = Σ λᵢ / (λᵢ + α)`, then `α = γ / mᵀm` and `β = (N − γ) / ‖y − Φm‖²`. `γ` is the effective number of parameters the data pins down.
- **Sequential learning.** The posterior after one batch is the prior for the next. Updating in batches gives exactly the same answer as fitting all the data at once.

## When to use / scenarios
- Learning: the simplest model where the full Bayesian machinery (prior, posterior, predictive, evidence) is exact and fits in 40 lines.
- Interviews: "how does ridge relate to a Gaussian prior", "where do error bars on a regression come from", "what is the marginal likelihood and why does it penalise complexity".
- Practice: small-data regression where you need an uncertainty estimate (calibration curves, sensor models, dose-response, A/B uplift with few points); tuning regularisation with no data to spare for validation; online updating of a linear model as new batches arrive.
- `sklearn.linear_model.BayesianRidge` implements the same evidence updates; `ARDRegression` gives each weight its own `α` for automatic feature selection.
- Not for: non-linear structure you cannot capture with a fixed basis (use a Gaussian process or a neural network with an uncertainty method), heavy-tailed noise or outliers (use a Student-t likelihood), or a classification target (the posterior is no longer Gaussian; use a Laplace approximation or MCMC).

## Setup & code
`pip install numpy`. Runs instantly on CPU.

```python
import numpy as np


def features(x, degree):
    """Polynomial design matrix [1, x, x^2, ..., x^degree]."""
    return np.vander(x, degree + 1, increasing=True)


def posterior(Phi, y, alpha, beta):
    """Gaussian prior w ~ N(0, alpha^-1 I), noise precision beta. Returns posterior mean and covariance."""
    S_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi
    S = np.linalg.inv(S_inv)
    m = beta * S @ Phi.T @ y
    return m, S


def predict(Phi_new, m, S, beta):
    """Predictive mean and std: noise variance 1/beta plus parameter uncertainty phi^T S phi."""
    mean = Phi_new @ m
    var = 1 / beta + np.einsum("ij,jk,ik->i", Phi_new, S, Phi_new)
    return mean, np.sqrt(var)


def log_evidence(Phi, y, alpha, beta):
    """log p(y | alpha, beta): the marginal likelihood, with the weights integrated out."""
    N, M = Phi.shape
    m, S = posterior(Phi, y, alpha, beta)
    E = beta / 2 * np.sum((y - Phi @ m) ** 2) + alpha / 2 * m @ m
    _, logdet_S = np.linalg.slogdet(S)
    return M / 2 * np.log(alpha) + N / 2 * np.log(beta) - E + logdet_S / 2 - N / 2 * np.log(2 * np.pi)


def fit_evidence(Phi, y, alpha=1e-2, beta=100.0, n_iter=500):
    """Empirical Bayes: MacKay fixed-point updates of alpha and beta that maximise the evidence."""
    N = len(y)
    eig = np.linalg.eigvalsh(Phi.T @ Phi)
    for _ in range(n_iter):
        m, S = posterior(Phi, y, alpha, beta)
        lam = beta * eig
        gamma = np.sum(lam / (lam + alpha))               # effective number of well-determined parameters
        alpha = gamma / (m @ m)
        beta = (N - gamma) / np.sum((y - Phi @ m) ** 2)
    return alpha, beta, gamma


rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)
x = rng.uniform(0, 1, 25)
x = x[(x < 0.35) | (x > 0.65)]                            # a gap in the middle of the data
y = f(x) + rng.normal(0, 0.2, len(x))                     # true noise std 0.2
x_test = np.linspace(0, 1, 500)
y_test = f(x_test) + rng.normal(0, 0.2, 500)

print(f"{len(x)} training points, gap in (0.35, 0.65)")
print("degree  log-evidence  test-RMSE  least-squares test-RMSE  alpha     noise-std  gamma")
for degree in [1, 3, 5, 7, 9]:
    Phi = features(x, degree)
    alpha, beta, gamma = fit_evidence(Phi, y)
    m, S = posterior(Phi, y, alpha, beta)
    mu, _ = predict(features(x_test, degree), m, S, beta)
    w_ls = np.linalg.lstsq(Phi, y, rcond=None)[0]
    te = np.sqrt(np.mean((mu - y_test) ** 2))
    te_ls = np.sqrt(np.mean((features(x_test, degree) @ w_ls - y_test) ** 2))
    print(f"{degree:>6}  {log_evidence(Phi, y, alpha, beta):>12.2f}  {te:>9.3f}  {te_ls:>24.3f}  "
          f"{alpha:<8.3g}  {1/np.sqrt(beta):>9.3f}  {gamma:.1f}")

Phi = features(x, 5)
alpha, beta, _ = fit_evidence(Phi, y)
m, S = posterior(Phi, y, alpha, beta)
for xq in [0.1, 0.5, 0.9, 1.2]:
    mu, sd = predict(features(np.array([xq]), 5), m, S, beta)
    print(f"x={xq:.1f}: mean {mu[0]:+.2f}  std {sd[0]:.2f}  (true f {f(xq):+.2f})")
mu, sd = predict(features(x_test, 5), m, S, beta)
cover = np.mean(np.abs(y_test - mu) < 1.96 * sd)
print(f"95% predictive interval covers {cover:.1%} of test points")

# sequential learning: the posterior after one batch is the prior for the next
Phi = features(x, 3)
m0, S0 = posterior(Phi[:10], y[:10], 2.0, 25.0)
S_inv = np.linalg.inv(S0) + 25.0 * Phi[10:].T @ Phi[10:]
m_seq = np.linalg.solve(S_inv, np.linalg.inv(S0) @ m0 + 25.0 * Phi[10:].T @ y[10:])
m_all, _ = posterior(Phi, y, 2.0, 25.0)
print("sequential == batch posterior mean:", np.allclose(m_seq, m_all))
```

Output (numpy 2.5):
```
18 training points, gap in (0.35, 0.65)
degree  log-evidence  test-RMSE  least-squares test-RMSE  alpha     noise-std  gamma
     1        -18.54      0.551                     0.534  1.02          0.572  1.8
     3        -12.55      0.228                     0.233  0.00184       0.185  4.0
     5        -11.16      0.300                     0.239  0.00951       0.190  4.2
     7        -11.98      0.291                     0.652  0.0101        0.189  4.6
     9        -11.83      0.259                     1.136  0.00908       0.173  4.9
x=0.1: mean +0.63  std 0.20  (true f +0.59)
x=0.5: mean +0.31  std 0.23  (true f +0.00)
x=0.9: mean -0.61  std 0.21  (true f -0.59)
x=1.2: mean +9.03  std 1.99  (true f +0.95)
95% predictive interval covers 88.2% of test points
sequential == batch posterior mean: True
```

Least squares gets steadily worse as the degree grows (test RMSE 0.23 at degree 3, 1.14 at degree 9) because 10 coefficients chase 18 noisy points. With `α` and `β` set by the evidence, the degree-9 model stays at 0.26. The prior keeps the extra coefficients small, and `γ = 4.9` says only about 5 of the 10 parameters are determined by the data. Degree 1 cannot bend, so the evidence fit explains most of the signal as noise (noise std 0.57). From degree 3 up, the learned noise std is 0.17 to 0.19, close to the true 0.2. The evidence peaks at degree 5 (−11.16), while degree 3 has the best test RMSE (0.228). The evidence is a good model-selection signal, but it does not directly minimise test error.

The predictive std is about 0.20 near the data, which is mostly the noise term. In the gap at `x = 0.5` it rises only to 0.23, while the mean is off by 0.31. Outside the data at `x = 1.2` the std grows to 1.99, but the mean of +9.03 is 4 standard deviations from the truth. A polynomial basis is global, so uncertainty in the gap is pinned by the data on both sides, and outside the data the polynomial runs off with confidence. The 95% predictive interval covers 88.2% of test points rather than 95%. The last line checks that two sequential batch updates give the same posterior mean as one fit on all the data.

## Choosing / trade-offs
- **Evidence vs cross-validation.** Evidence uses all the data, is smooth in `α` and `β`, and costs one fit per step. Cross-validation optimises predictive error directly and makes no assumption about the prior. If the model is badly misspecified, trust CV.
- **Full Bayes vs empirical Bayes.** Putting Gamma priors on `α` and `β` and integrating them out (or sampling them, e.g. in PyMC) gives more honest uncertainty with very little data. Point estimates by evidence are almost always good enough once N is well above the number of parameters.
- **Basis choice decides the uncertainty shape.** Polynomials give wild extrapolation. Localised bases (RBFs, splines) make uncertainty grow away from the data, and a Gaussian process is the limit of infinitely many such bases.
- **ARD.** One `α` per weight prunes irrelevant features (their `α → ∞`), but it can be unstable with correlated features.
- **Cost.** Inverting the `M × M` matrix `S` costs O(M³). Cheap for tens to thousands of features. A GP costs O(N³) in the number of points instead.

## Gotchas
- Rescale inputs before building a polynomial basis. A Vandermonde matrix on unscaled `x` is badly conditioned, and `np.linalg.inv` quietly returns garbage. Prefer `solve` or Cholesky for large `M`.
- The MacKay updates can settle in a poor optimum that explains everything as noise. Starting from `β = 1` here gave a noise std of 0.55 for every degree. Start `β` near `1 / (0.1 · var(y))` and `α` small, and check the result against a least-squares fit.
- The predictive variance is only as good as the model. It does not grow in regions the basis cannot express (the gap above), and it is overconfident when the noise is not Gaussian.
- Do not regularise the bias. With a shared `α` on every weight, including the intercept, centre `y` first, or exclude the intercept from the prior.
- Report the predictive std (`1/β + φᵀSφ`), not just the parameter part (`φᵀSφ`), when you claim an interval for new observations.
- The evidence compares models with the same data and likelihood. It is not comparable across different targets or transformations of `y`.

## Related
- [[linear-regression-from-scratch]] - the least-squares and gradient-descent fit that the posterior mean generalises.
- [[ridge-and-lasso-from-scratch]] - ridge is the MAP estimate of this model with `λ = α/β`.
- [[gaussian-process-regression-from-scratch]] - the kernel version, whose uncertainty does grow away from the data.
- [[bayesian-and-gaussian-processes]] - library-level Bayesian methods and when to reach for them.
- [[bayesian-deep-learning-and-uncertainty]] - uncertainty estimates once the model is a neural network.

## References
- Bishop (2006), Pattern Recognition and Machine Learning, chapter 3 (Bayesian linear regression, evidence approximation): https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/
- MacKay (1992), "Bayesian interpolation", Neural Computation 4(3): https://doi.org/10.1162/neco.1992.4.3.415
- scikit-learn user guide, Bayesian regression (BayesianRidge, ARDRegression): https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression
