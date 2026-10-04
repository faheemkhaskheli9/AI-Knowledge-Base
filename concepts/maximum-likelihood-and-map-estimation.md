---
title: Maximum likelihood and MAP estimation
category: concepts
tags: [maximum-likelihood, mle, map, bayesian, prior, negative-log-likelihood, regularization, aic, scipy, statsmodels]
use_cases:
  - "understand why cross-entropy and MSE are the losses we train with"
  - "fit a probability distribution (Normal, Poisson, Weibull) to observed data"
  - "write a custom loss for a model that predicts a distribution"
  - "explain L2/L1 regularization as a prior on the weights"
  - "estimate a rate from very few observations without getting 0% or 100%"
status: draft
last_verified: 2026-10-04
sources:
  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html
  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
  - https://www.statsmodels.org/stable/examples/notebooks/generated/generic_mle.html
  - https://www.deeplearningbook.org/contents/ml.html
---

# Maximum likelihood and MAP estimation

## Summary
Maximum likelihood estimation (MLE) picks the parameters under which the observed data is most probable. Almost every training loss is an MLE in disguise: mean squared error is the Gaussian negative log-likelihood, cross-entropy is the categorical one. Maximum a posteriori (MAP) estimation adds a prior over the parameters, and that prior is exactly what L2 and L1 regularization are. Knowing this lets you pick the right loss for a target, fit distributions to data, and reason about regularization instead of tuning it blindly.

## Key concepts
- **Likelihood.** L(θ) = p(data | θ), viewed as a function of θ with the data fixed. It is not a probability distribution over θ.
- **Log-likelihood and NLL.** For i.i.d. data, log L(θ) = Σ log p(xᵢ | θ). Products become sums (numerically stable), and optimizers minimize the **negative log-likelihood** (NLL).
- **Loss ↔ noise model.**
  - Gaussian noise with fixed σ → NLL ∝ MSE.
  - Laplace noise → MAE (median regression).
  - Bernoulli / categorical → binary / categorical cross-entropy ([[loss-functions]]).
  - Poisson → Poisson deviance for counts ([[generalized-linear-models]]).
- **MAP.** argmax p(θ | data) = argmax [log p(data | θ) + log p(θ)]. A Gaussian prior N(0, τ²) on weights adds λ‖w‖² with λ = σ²/τ² (ridge / weight decay). A Laplace prior adds λ‖w‖₁ (lasso). A flat prior gives back MLE.
- **Conjugate priors.** When prior and posterior share a family the update is a formula: a Beta(a, b) prior on a coin's p with h heads in n flips gives Beta(a+h, b+n-h). MAP = (h+a-1)/(n+a+b-2). Additive (Laplace) smoothing in Naive Bayes is this ([[svm-knn-naive-bayes]]).
- **Properties of MLE.** Consistent and asymptotically efficient with enough data and a correct model. Can be biased in small samples: the MLE of variance divides by n, not n-1.
- **Model comparison.** AIC = 2k - 2 log L, BIC = k log n - 2 log L. Lower is better; they penalize the number of parameters k.
- **MAP is not full Bayes.** MAP returns one point (the posterior mode). Full Bayesian inference keeps the whole posterior for uncertainty ([[bayesian-and-gaussian-processes]], [[bayesian-deep-learning-and-uncertainty]]).

## When to use / scenarios
- Choosing a loss: counts (calls per hour, claims) → Poisson NLL; positive skewed values (durations, amounts) → Gamma / log-normal NLL; heavy outliers → Laplace or Student-t NLL instead of MSE.
- Fitting a distribution to data: time-to-failure (Weibull), arrival counts (Poisson vs negative binomial), latencies (log-normal), then using its quantiles for SLAs or capacity.
- Small data: click-through rate from 3 impressions, defect rate from a short run. A Beta prior (MAP or posterior mean) avoids 0% and 100%.
- Probabilistic outputs in deep learning: a network that predicts mean and variance trained on Gaussian NLL ([[quantile-regression]] is the non-parametric alternative).
- NOT enough on its own when you need calibrated uncertainty in θ: use bootstrap, Fisher-information standard errors (statsmodels reports them), or full Bayesian inference.

## Setup & code
```bash
pip install "scikit-learn>=1.4" scipy numpy
```
MLE in closed form and numerically, MLE vs MAP on a coin, MAP = ridge, and AIC to choose between Poisson and negative binomial:
```python
import numpy as np
from scipy import optimize, stats
from sklearn.linear_model import Ridge

rng = np.random.default_rng(0)

# 1) Closed form vs numeric MLE for a Normal
x = rng.normal(loc=5.0, scale=2.0, size=200)
nll = lambda p: -stats.norm.logpdf(x, loc=p[0], scale=np.exp(p[1])).sum()   # log-sigma keeps sigma > 0
res = optimize.minimize(nll, x0=[0.0, 0.0])
print(f"closed form: mu={x.mean():.3f} sigma={x.std(ddof=0):.3f}")
print(f"numeric    : mu={res.x[0]:.3f} sigma={np.exp(res.x[1]):.3f}")
print(f"scipy fit  : mu={stats.norm.fit(x)[0]:.3f} sigma={stats.norm.fit(x)[1]:.3f}")

# 2) Coin flips: MLE vs MAP with a Beta(a, b) prior
heads, n = 3, 3
a, b = 2, 2
print(f"coin MLE p={heads / n:.3f}  MAP(Beta(2,2)) p={(heads + a - 1) / (n + a + b - 2):.3f}  "
      f"posterior mean={(heads + a) / (n + a + b):.3f}")

# 3) MAP with a Gaussian prior on weights == L2 (ridge)
X = rng.normal(size=(30, 5))
w_true = np.array([2.0, -1.0, 0.0, 0.0, 0.5])
y = X @ w_true + rng.normal(scale=1.0, size=30)
sigma2, tau2 = 1.0, 0.5            # noise variance, prior variance
lam = sigma2 / tau2
w_mle = np.linalg.solve(X.T @ X, X.T @ y)
w_map = np.linalg.solve(X.T @ X + lam * np.eye(5), X.T @ y)
w_ridge = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_
print("w_mle  ", np.round(w_mle, 3))
print("w_map  ", np.round(w_map, 3))
print("ridge  ", np.round(w_ridge, 3), "same as MAP:", np.allclose(w_map, w_ridge))

# 4) Poisson vs negative binomial for counts, compared by AIC
counts = rng.poisson(3.2, size=500)
lam_hat = counts.mean()                                  # Poisson MLE is the sample mean
ll_pois = stats.poisson.logpmf(counts, lam_hat).sum()
r, p = 5.0, 5.0 / (5.0 + lam_hat)
nb_nll = lambda q: -stats.nbinom.logpmf(counts, np.exp(q[0]), 1 / (1 + np.exp(-q[1]))).sum()
nb = optimize.minimize(nb_nll, x0=[np.log(r), np.log(p / (1 - p))])
print(f"poisson lambda={lam_hat:.3f} AIC={2 * 1 - 2 * ll_pois:.1f}  negbin AIC={2 * 2 + 2 * nb.fun:.1f}")
```
Output (SciPy 1.18.0, scikit-learn 1.9.0):
```
closed form: mu=5.031 sigma=1.922
numeric    : mu=5.031 sigma=1.922
scipy fit  : mu=5.031 sigma=1.922
coin MLE p=1.000  MAP(Beta(2,2)) p=0.800  posterior mean=0.714
w_mle   [ 1.966 -1.209  0.159 -0.052  0.571]
w_map   [ 1.706 -1.043  0.158 -0.137  0.552]
ridge   [ 1.706 -1.043  0.158 -0.137  0.552] same as MAP: True
poisson lambda=3.160 AIC=1939.0  negbin AIC=1941.1
```
Three heads in three flips give an MLE of p=1, which would predict tails are impossible; the Beta(2,2) prior pulls it to 0.8. The ridge coefficients match the MAP solution exactly, with `alpha = σ²/τ²`. On truly Poisson data the extra dispersion parameter of the negative binomial does not pay for itself, so AIC prefers Poisson.

For regression-style custom likelihoods with standard errors, subclass `statsmodels.base.model.GenericLikelihoodModel`. In PyTorch, `torch.distributions.<Dist>(...).log_prob(y)` gives the NLL for any distribution head, and `nn.GaussianNLLLoss` / `nn.PoissonNLLLoss` cover the common ones.

## Choosing / trade-offs
- **Closed form vs numeric.** Normal, Poisson, Bernoulli, exponential have closed-form MLEs (means and proportions). Weibull, Gamma, negative binomial, mixtures need an optimizer (`scipy.stats.<dist>.fit`, `optimize.minimize`) or EM ([[gaussian-mixture-models-and-em]]).
- **MLE vs MAP.** Many rows per parameter: they agree, use MLE. Few rows, many parameters, or rare events: MAP (or a full posterior) is far more stable. The prior strength is a hyperparameter; set it from domain knowledge or by cross-validation.
- **MAP vs full Bayes.** MAP is as cheap as MLE. Full Bayes (PyMC, NumPyro, conjugate formulas) is needed when decisions depend on uncertainty in θ, e.g. A/B tests with few conversions ([[statistics-and-ab-testing]]).
- **Which distribution.** Compare candidate families by AIC/BIC and check the fit visually (QQ plot, histogram vs fitted pdf). A good AIC on the wrong family still gives wrong tails.

## Gotchas
- Optimize unconstrained parameters: fit log σ, logit p, log rate, then transform. Raw σ can go negative and crash `logpdf`.
- Sum log-probabilities, never multiply probabilities: 1000 rows of p≈0.01 underflow to 0.
- `scipy.stats.<dist>.fit` also fits `loc` (and `scale`) by default. For a Weibull or Gamma starting at 0, pass `floc=0`, or the fit can wander to a nonsensical shifted solution.
- MLE overfits like any estimator: a Gaussian mixture can drive one component's variance to 0 on a single point (infinite likelihood). Regularize the covariance (`reg_covar`) or use a prior.
- Comparing log-likelihoods or AIC across models only works on the same data and the same target scale. A model of log(y) and a model of y are not comparable without the Jacobian term.
- "Weight decay = Gaussian prior" holds for plain SGD. With Adam, L2-in-the-loss and decoupled weight decay (AdamW) differ ([[optimizers]]).
- The MLE of variance (`ddof=0`) is biased low on small samples; use `ddof=1` when reporting a population estimate.

## Related
- [[loss-functions]] - each standard loss is a negative log-likelihood.
- [[math-for-machine-learning]] - the probability and calculus behind it.
- [[information-theory-for-ml]] - minimizing NLL = minimizing cross-entropy / KL to the data.
- [[generalized-linear-models]] - MLE for Poisson, Gamma and binomial targets.
- [[regularization-in-deep-learning]] - weight decay as a prior.
- [[bayesian-and-gaussian-processes]] - going from the posterior mode to the full posterior.
- [[gaussian-mixture-models-and-em]] - MLE when there are latent variables.

## References
- SciPy, `rv_continuous.fit`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html
- SciPy, `optimize.minimize`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
- statsmodels, Maximum likelihood estimation (generic models): https://www.statsmodels.org/stable/examples/notebooks/generated/generic_mle.html
- Goodfellow, Bengio & Courville, Deep Learning, ch. 5.5-5.6 (MLE, Bayesian statistics): https://www.deeplearningbook.org/contents/ml.html
- Murphy, Probabilistic Machine Learning: An Introduction, ch. 4 (Statistics), MIT Press 2022.
