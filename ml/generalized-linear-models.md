---
title: Generalized linear models (Poisson, Gamma, Tweedie, link functions)
category: ml
tags: [glm, generalized-linear-models, poisson-regression, gamma-regression, tweedie, link-function, exposure, offset, deviance, insurance, scikit-learn, statsmodels]
use_cases:
  - "predict counts such as claims, defects, visits or calls per period"
  - "model positive, right-skewed amounts such as claim size, order value or repair cost"
  - "price insurance from claim frequency and severity with exposure"
  - "predict a target with many exact zeros plus a skewed positive tail (pure premium, spend)"
  - "get interpretable multiplicative effects (rate ratios) a regulator or actuary can read"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-models
  - https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
  - https://www.statsmodels.org/stable/glm.html
---

# Generalized linear models (Poisson, Gamma, Tweedie, link functions)

## Summary
A GLM keeps the linear model's `X @ w` but passes it through a link function and pairs it with a target distribution that fits the data: Poisson for counts, Gamma for positive skewed amounts, Tweedie for "many zeros plus a skewed tail", Bernoulli for binary (that is logistic regression). With a log link, predictions are always positive and each coefficient is a multiplicative effect (`exp(w)` = rate ratio). GLMs are the standard in insurance pricing, epidemiology and demand counting because they are interpretable, fast, and correct about the target's variance; ordinary least squares on such targets gives negative predictions and wrong uncertainty.

## Key concepts
- **Three parts.** Distribution (exponential family), linear predictor `eta = X @ w + b`, link `g(mu) = eta`. Common pairs: Normal + identity (OLS), Bernoulli + logit (logistic), Poisson + log, Gamma + log, Tweedie + log.
- **Variance function.** What really separates them: Normal `Var = const`, Poisson `Var ∝ mu`, Gamma `Var ∝ mu^2`, Tweedie `Var ∝ mu^p`. Tweedie with `p=0` is Normal, `p=1` Poisson, `p=2` Gamma, `1<p<2` compound Poisson-Gamma (point mass at zero plus continuous positive part).
- **Log link = multiplicative model.** `mu = exp(b) * exp(w1*x1) * ...`; a coefficient of 0.7 means x1=1 multiplies the rate by `e^0.7 ≈ 2.0`.
- **Exposure and offsets.** Counts over unequal periods (0.3 vs 1.0 policy-years) need exposure. Either model the count with `offset = log(exposure)` (statsmodels) or model frequency `count/exposure` with `sample_weight=exposure` (scikit-learn); for Poisson these are equivalent.
- **Deviance.** The GLM loss and the right evaluation metric: mean Poisson/Gamma/Tweedie deviance, and `D²` (fraction of deviance explained, the GLM analogue of R²). MSE on counts overweights large values.
- **Overdispersion.** Real counts often have variance above the mean; Poisson coefficients stay fine but standard errors are too small. Use quasi-Poisson / negative binomial (statsmodels) when you need inference.
- **Frequency-severity split.** Insurance pure premium = frequency (Poisson) x severity (Gamma, fitted on rows with claims, weighted by claim count); or one Tweedie model on pure premium directly.

## When to use / scenarios
- Insurance: claim frequency, severity and pure premium for pricing ([[finance]]).
- Counts: support tickets per day, defects per batch, hospital admissions per region ([[manufacturing-iot]], [[healthcare]]).
- Retail: units sold per store-day, spend per customer with many non-buyers ([[ecommerce-retail]]).
- When stakeholders need a rate-ratio table, monotone and auditable effects, or a regulator-approved model form.
- NOT the best choice for raw accuracy on large, interaction-heavy data: gradient boosting with `objective="poisson"`/`"tweedie"`/`"gamma"` usually wins ([[gradient-boosting-tabular]]); keep the GLM as baseline or for the explainable final model.
- Non-linear single-feature effects: add splines or move to a GAM ([[generalized-additive-models-and-splines]]).

## Setup & code
```bash
pip install scikit-learn numpy        # statsmodels for p-values / offsets / negative binomial
```
Frequency, severity and pure premium on a synthetic insurance portfolio:
```python
import numpy as np
from sklearn.linear_model import LinearRegression, PoissonRegressor, GammaRegressor, TweedieRegressor
from sklearn.metrics import mean_poisson_deviance, d2_tweedie_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
n = 20000
X = np.column_stack([rng.integers(0, 2, n), rng.normal(0, 1, n)])   # young_driver flag, car power (std)
exposure = rng.uniform(0.1, 1.0, n)                                  # policy-years
rate = np.exp(-2.0 + 0.7 * X[:, 0] + 0.3 * X[:, 1])                 # true claims per year
counts = rng.poisson(rate * exposure)
freq = counts / exposure

X_tr, X_te, f_tr, f_te, w_tr, w_te = train_test_split(X, freq, exposure, random_state=0)

# Poisson GLM on frequency, weighted by exposure == count model with log(exposure) offset
pois = PoissonRegressor(alpha=0).fit(X_tr, f_tr, sample_weight=w_tr)
ols = LinearRegression().fit(X_tr, f_tr, sample_weight=w_tr)
print("true coefs  [0.7, 0.3], intercept -2.0")
print("poisson     ", pois.coef_.round(3), round(pois.intercept_, 3))
print("rate ratio young driver: x%.2f" % np.exp(pois.coef_[0]))
print("OLS negative predictions:", int((ols.predict(X_te) <= 0).sum()), "of", len(X_te))
print("poisson deviance  poisson %.4f  mean-only %.4f" % (
    mean_poisson_deviance(f_te, pois.predict(X_te), sample_weight=w_te),
    mean_poisson_deviance(f_te, np.full_like(f_te, np.average(f_tr, weights=w_tr)), sample_weight=w_te)))

# Gamma GLM on severity: positive amounts, only rows with claims, weighted by claim count
sev_mu = np.exp(7.0 + 0.2 * X[:, 1])
mask = counts > 0
sev = rng.gamma(shape=2.0, scale=sev_mu[mask] / 2.0)
gam = GammaRegressor(alpha=0).fit(X[mask], sev, sample_weight=counts[mask])
print("gamma       ", gam.coef_.round(3), round(gam.intercept_, 3), "(true [0, 0.2], 7.0)")

# Tweedie 1<p<2 on pure premium = total loss / exposure (mostly exact zeros)
loss = np.zeros(n)
loss[mask] = sev * counts[mask]
pp = loss / exposure
print("share of zero pure premium: %.3f" % (pp == 0).mean())
tw = TweedieRegressor(power=1.5, link="log", alpha=0, max_iter=1000).fit(X, pp, sample_weight=exposure)
print("tweedie p=1.5", tw.coef_.round(3), "D2:", round(d2_tweedie_score(pp, tw.predict(X), sample_weight=exposure, power=1.5), 4))
```
Output with scikit-learn 1.9.0:
```text
true coefs  [0.7, 0.3], intercept -2.0
poisson      [0.675 0.295] -1.968
rate ratio young driver: x1.96
OLS negative predictions: 26 of 5000
poisson deviance  poisson 0.8668  mean-only 0.9166
gamma        [0.001 0.222] 7.005 (true [0, 0.2], 7.0)
share of zero pure premium: 0.892
tweedie p=1.5 [0.717 0.511] D2: 0.0696
```
The Poisson and Gamma fits recover the true effects; OLS predicts negative claim rates. The Tweedie coefficients are roughly frequency + severity effects added (0.3 + 0.2 ≈ 0.5 for car power). Low D² values are normal for claims data: most variance is pure chance.

For inference (standard errors, p-values, explicit offset, overdispersion):
```python
import statsmodels.api as sm          # pip install statsmodels
res = sm.GLM(counts, sm.add_constant(X), family=sm.families.Poisson(), offset=np.log(exposure)).fit()
print(res.summary())                  # family=sm.families.NegativeBinomial() if overdispersed
```

## Choosing / trade-offs
- **Poisson vs negative binomial.** Same mean model; negative binomial adds a dispersion parameter for honest intervals. For point predictions Poisson is fine.
- **Frequency x severity vs single Tweedie.** Two models give more insight (what drives claims vs cost) and can use different features; Tweedie is one model and handles exact zeros directly. Choose `power` by CV on deviance, commonly 1.2-1.8.
- **Regularisation.** `alpha` is an L2 penalty (default 1.0 in scikit-learn, not 0). Scale features and tune it; set `alpha=0` only to reproduce a classical unpenalised fit.
- **Solver.** `solver="newton-cholesky"` is much faster when samples >> features (e.g. one-hot encoded rating factors); default `lbfgs` handles wide data.
- **GLM vs boosting.** GLM: interpretable rating tables, monotone, stable, regulator-friendly. Boosting with a Poisson/Tweedie objective: better accuracy, captures interactions; explain with SHAP ([[model-interpretability]]).

## Gotchas
- `PoissonRegressor()` defaults to `alpha=1.0`, which shrinks coefficients noticeably on unscaled or small data. Pass `alpha` deliberately.
- Fitting counts without exposure: a 1-month policy with 0 claims looks "safe" next to a 12-month one. Always weight or offset.
- Evaluating with R²/MSE instead of deviance; and comparing deviance across different `power` values (each power is its own scale).
- Gamma and log-link Tweedie with `p>=2` reject `y <= 0`; Poisson and `1<p<2` accept zeros but not negatives (refunds, corrections need handling first).
- Log link means predictions are `exp(...)`: a feature far outside the training range can explode the prediction. Cap or bin extreme inputs.
- Severity models must exclude zero-claim rows and weight by claim count when the target is average cost per claim.
- Rate ratios are only "effects" under the model; correlated rating factors share credit, so do not read them causally ([[causal-inference-and-uplift]]).

## Related
- [[linear-models]] - OLS, Ridge, Lasso and logistic regression, the Normal and Bernoulli GLMs.
- [[generalized-additive-models-and-splines]] - non-linear smooth effects on top of a GLM.
- [[loss-functions]] - Poisson, Tweedie and quantile losses used by other model families.
- [[gradient-boosting-tabular]] - boosting with count and Tweedie objectives.
- [[survival-analysis]] - time-to-event, closely related to Poisson rate models.
- [[statistics-and-ab-testing]] - inference and confidence intervals.
- [[finance]] - insurance and credit scenarios.

## References
- scikit-learn, Generalized Linear Models: https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-models
- scikit-learn example, Tweedie regression on insurance claims: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
- statsmodels GLM: https://www.statsmodels.org/stable/glm.html
