---
title: Causal inference and uplift modelling
category: ml
tags: [causal-inference, uplift, treatment-effect, ab-testing, propensity-score, dowhy, econml, meta-learners]
use_cases:
  - "estimate whether a discount, campaign or feature actually caused an outcome"
  - "target a promotion only at customers it will change (uplift / persuadables)"
  - "measure an effect from observational data when an A/B test is not possible"
  - "find which customer segments respond most to a treatment"
  - "check that a churn or risk model is not mistaken for a model of what to do"
status: draft
last_verified: 2026-10-03
sources:
  - https://www.pywhy.org/dowhy/
  - https://github.com/py-why/EconML
  - https://github.com/uber/causalml
  - https://matheusfacure.github.io/python-causality-handbook/
---

# Causal inference and uplift modelling

## Summary
Predictive models answer "what will happen?"; causal models answer "what
would happen if we did X?". Causal inference estimates the effect of a
treatment (price, coupon, drug, feature) on an outcome, either from a
randomised experiment or from observational data with assumptions stated up
front. Uplift modelling estimates that effect per individual, so you can act
only where it changes the result.

## Key concepts
- Potential outcomes: each unit has Y(1) and Y(0); you only ever see one.
  Effects: ATE (average), ATT (on the treated), CATE (conditional on features).
- Confounders affect both treatment and outcome and bias naive comparisons.
  Draw a causal graph (DAG) to decide what to adjust for.
- Do not adjust for mediators (on the causal path) or colliders (caused by
  both); adjusting for them creates bias.
- Identification assumptions for observational data: no unmeasured
  confounding, overlap (every type of unit could get either treatment),
  no interference between units. These cannot be fully tested.
- Randomised A/B tests remove confounding by design; they are the gold standard.
- Estimators: regression adjustment, propensity score matching/weighting
  (IPW), doubly robust (AIPW), double/debiased ML (DML).
- Meta-learners for CATE: S-learner (one model with T as a feature),
  T-learner (one model per arm), X-learner, R-learner, causal forests.
- Quasi-experiments: difference-in-differences, regression discontinuity,
  instrumental variables, synthetic control.

## When to use / scenarios
- Marketing/retail: uplift targeting for coupons and retention offers - skip
  "sure things" and "lost causes", avoid "sleeping dogs" who react badly.
- Product: estimate a feature's effect when rollout was not randomised.
- Pricing: elasticity estimates that adjust for promotions and seasonality.
- Healthcare/policy: effect estimates from registries (with domain experts
  and strong caveats).
- NOT for: pure prediction tasks (use [[gradient-boosting-tabular]]); and do
  not use observational methods when you can simply run an A/B test.

## Setup & code
```bash
pip install scikit-learn
```
```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(0)
n = 20_000
X = rng.normal(size=(n, 3))
# Confounding: customers with high X0 are more likely to get the coupon AND spend more.
p_treat = 1 / (1 + np.exp(-1.5 * X[:, 0]))
T = rng.binomial(1, p_treat)
true_effect = 2.0 + 1.0 * X[:, 1]                  # effect varies with X1
y = 5 * X[:, 0] + true_effect * T + rng.normal(size=n)

naive = y[T == 1].mean() - y[T == 0].mean()

# T-learner: one outcome model per arm, effect = difference of predictions.
m1 = GradientBoostingRegressor(random_state=0).fit(X[T == 1], y[T == 1])
m0 = GradientBoostingRegressor(random_state=0).fit(X[T == 0], y[T == 0])
cate = m1.predict(X) - m0.predict(X)

# Inverse propensity weighting for the average effect.
e = LogisticRegression().fit(X, T).predict_proba(X)[:, 1].clip(0.01, 0.99)
ipw = np.mean(T * y / e) - np.mean((1 - T) * y / (1 - e))

print(f"true ATE   {true_effect.mean():.2f}")
print(f"naive diff {naive:.2f}   (biased by confounding)")
print(f"T-learner  {cate.mean():.2f}")
print(f"IPW        {ipw:.2f}")
print(f"corr(CATE, true effect) {np.corrcoef(cate, true_effect)[0, 1]:.2f}")
```
Output (scikit-learn 1.9): true ATE 2.00, naive difference 7.33, T-learner
2.02, IPW 1.88, and the per-row CATE correlates 0.95 with the true effect.
The synthetic data has no hidden confounders; real data will.

Libraries: DoWhy (model the graph, identify, estimate, then refute with
placebo and random-confounder tests), EconML (DML, causal forests, meta-
learners with confidence intervals), CausalML (uplift trees and meta-learners).

## Choosing / trade-offs
- Can you randomise? Run an A/B test; use causal ML on the experiment data
  for CATE and targeting.
- Observational, many confounders: doubly robust / DML (EconML) with
  cross-fitting; report a sensitivity analysis.
- Policy changed at a known time for some units: difference-in-differences
  or synthetic control.
- Targeting: evaluate uplift models with Qini/uplift curves on held-out
  randomised data, not with AUC.

## Gotchas
- A churn model ranks who will leave, not who an offer will save; targeting
  by risk alone wastes budget on lost causes.
- Adjusting for post-treatment variables (e.g. "clicked the email") biases the estimate.
- Extreme propensities (near 0 or 1) mean no overlap; IPW explodes. Trim or
  restrict the population and say so.
- Feature importance and SHAP values are not causal effects.
- Uplift has no ground truth per row; validation needs randomised holdout data.
- Peeking at A/B results and stopping early inflates false positives; fix
  the sample size or use sequential tests.

## Related
- [[model-evaluation-and-metrics]] - A/B testing basics and metrics.
- [[gradient-boosting-tabular]] - base learners for meta-learners.
- [[model-interpretability]] - why SHAP is not causal.
- [[recommender-systems]] - online experiments for ranking changes.
- [[bayesian-and-gaussian-processes]] - Bayesian effect estimates and hierarchical pooling.
- [[marketing-content]], [[ecommerce-retail]] - main application areas.

## References
- DoWhy: https://www.pywhy.org/dowhy/
- EconML: https://github.com/py-why/EconML
- CausalML: https://github.com/uber/causalml
- Causal Inference for the Brave and True (free book): https://matheusfacure.github.io/python-causality-handbook/
