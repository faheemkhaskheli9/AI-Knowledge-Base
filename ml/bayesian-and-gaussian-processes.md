---
title: Bayesian ML and Gaussian processes
category: ml
tags: [bayesian, gaussian-process, uncertainty, probabilistic-programming, pymc, bayesian-optimization, calibration]
use_cases:
  - "predict a value with an honest confidence interval, not just a point estimate"
  - "fit a model on a small dataset (tens to a few thousand rows) and know where it is unsure"
  - "pool estimates across many small groups (stores, clinics, products) with a hierarchical model"
  - "build a surrogate model for expensive experiments or simulations"
  - "decide what to measure next (active learning, Bayesian optimization)"
status: draft
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/gaussian_process.html
  - https://www.pymc.io/projects/docs/en/stable/learn.html
  - https://gaussianprocess.org/gpml/
---

# Bayesian ML and Gaussian processes

## Summary
Bayesian methods treat model parameters (or the function itself) as uncertain
and return a distribution over predictions instead of a single number: prior
belief + data -> posterior. Use them when you need calibrated uncertainty, have
little data, or want to share strength across related groups. Gaussian
processes (GPs) are the workhorse for small-data regression with error bars;
probabilistic programming (PyMC, NumPyro, Stan) handles custom hierarchical
models.

## Key concepts
- Bayes' rule: posterior is proportional to likelihood x prior. The prior is
  a modelling choice and must be stated, not hidden.
- Posterior predictive: average predictions over parameter uncertainty; gives
  credible intervals.
- Epistemic (model, shrinks with data) vs aleatoric (noise, does not) uncertainty.
- Gaussian process: a prior over functions defined by a kernel (RBF, Matern,
  periodic, linear; kernels add and multiply). Prediction is closed-form;
  the variance grows away from training data.
- Kernel hyperparameters (length scale, noise) are fitted by maximising the
  marginal likelihood; read them back to check they make sense.
- Hierarchical (multilevel) models: group parameters drawn from a shared
  distribution, so small groups shrink toward the global mean ("partial pooling").
- Inference: exact (GP regression, conjugate models), MCMC (NUTS, accurate,
  slow), variational inference (fast, approximate), Laplace.
- Cheap Bayesian-flavoured uncertainty for deep nets: deep ensembles, MC
  dropout, conformal prediction for distribution-free intervals.

## When to use / scenarios
- Engineering/science: surrogate models for expensive simulations or lab
  experiments; Bayesian optimization picks the next experiment
  ([[hyperparameter-tuning]] uses the same idea).
- Retail/marketing: hierarchical models for many stores or SKUs with sparse
  sales; marketing-mix models (PyMC-Marketing, Meridian).
- Healthcare/finance: risk estimates that must carry an interval for a human
  decision.
- Sensors/geostatistics: interpolation over space or time (kriging is a GP).
- NOT for: large tabular data where accuracy is all that matters (use
  [[gradient-boosting-tabular]], add conformal intervals if needed); exact GPs
  above ~10k rows (O(n^3)) - use sparse/approximate GPs (GPyTorch) or a different model.

## Setup & code
```bash
pip install scikit-learn
```
```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, size=(30, 1))
y = np.sin(X).ravel() + rng.normal(0, 0.2, size=30)

kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0).fit(X, y)

X_new = np.array([[2.5], [5.0], [15.0]])          # 15 is far outside the data
mean, std = gp.predict(X_new, return_std=True)
for x, m, s in zip(X_new.ravel(), mean, std):
    print(f"x={x:5.1f}  mean={m:+.2f}  95% CI=[{m - 1.96*s:+.2f}, {m + 1.96*s:+.2f}]")
print(gp.kernel_)                                  # fitted hyperparameters
```
Output (scikit-learn 1.9): the interval is about +/-0.37 inside the data and
widens to about +/-1.6 at x=15, where the GP falls back to its prior. The
fitted kernel was `1.55**2 * RBF(length_scale=1.62) + WhiteKernel(noise_level=0.0949)`.

For custom hierarchical models use PyMC (`pip install pymc`): declare priors
and likelihood in a `with pm.Model():` block, call `pm.sample()`, and check
`arviz.summary()` for `r_hat` near 1.0 before trusting results.

## Choosing / trade-offs
- GP vs gradient boosting: GP wins on small, smooth problems that need
  intervals; boosting wins on large, messy, high-dimensional tabular data.
- Exact GP (scikit-learn) up to a few thousand rows; GPyTorch for GPU and
  sparse/variational GPs beyond that.
- MCMC (PyMC/NumPyro NUTS) for correctness on models up to moderate size;
  variational inference when MCMC is too slow and approximate posteriors are acceptable.
- Only need intervals around any model: conformal prediction (MAPIE) is
  simpler and has coverage guarantees without a Bayesian model.

## Gotchas
- Unscaled inputs make a single length scale meaningless; standardise X, and
  use `normalize_y=True` or centre y.
- Marginal-likelihood fitting has local optima; set `n_restarts_optimizer`.
- A GP extrapolates back to the prior mean, not the trend. Add a linear kernel
  or a mean function if the trend matters.
- Uncertainty is only as honest as the model: a wrong kernel or likelihood
  gives confidently wrong intervals. Check coverage on held-out data.
- MCMC warnings (divergences, `r_hat` > 1.01) mean the results are not
  trustworthy; reparameterise (non-centred) rather than ignore.
- Priors matter most where data is thin - which is exactly where Bayesian
  methods are used. Run a prior predictive check.

## Related
- [[linear-models]] - Bayesian linear regression is the simplest case.
- [[hyperparameter-tuning]] - Bayesian optimization uses a GP/TPE surrogate.
- [[model-evaluation-and-metrics]] - calibration and interval coverage.
- [[time-series-forecasting]] - probabilistic forecasts.
- [[causal-inference-and-uplift]] - Bayesian models are common for effect estimates.

## References
- scikit-learn Gaussian processes: https://scikit-learn.org/stable/modules/gaussian_process.html
- PyMC learning resources: https://www.pymc.io/projects/docs/en/stable/learn.html
- Rasmussen & Williams, Gaussian Processes for Machine Learning (free): https://gaussianprocess.org/gpml/
