---
title: Statistics for ML and A/B testing
category: ml
tags: [statistics, ab-testing, hypothesis-testing, p-value, confidence-interval, power-analysis, bootstrap, experimentation, scipy, statsmodels]
use_cases:
  - "decide whether a new model, feature or page actually improved conversion"
  - "work out how many users an experiment needs before launching it"
  - "put a confidence interval on a model metric or business KPI"
  - "compare two models' offline scores and know if the gap is real"
  - "analyse a skewed metric like revenue per user"
status: draft
last_verified: 2026-10-03
sources:
  - https://docs.scipy.org/doc/scipy/reference/stats.html
  - https://www.statsmodels.org/stable/stats.html
  - https://exp-platform.com/
---

# Statistics for ML and A/B testing

## Summary
The statistics you need to tell a real improvement from noise: hypothesis
tests, confidence intervals, power and sample size, and the bootstrap. An
online A/B test (randomised controlled experiment) is the standard way to
prove a model or product change moved a business metric; offline metrics
only suggest it.

## Key concepts
- Null hypothesis H0 (no difference) vs alternative. The p-value is
  P(data this extreme | H0), not P(H0 is true).
- Significance level alpha (usually 0.05) = false-positive rate you accept.
  Power (usually 0.8) = chance to detect a true effect of the size you care about.
- Minimum detectable effect (MDE): the smallest lift worth detecting. Sample
  size grows with 1/MDE^2, so halving the MDE needs ~4x the users.
- Confidence interval: report the effect size with its CI, not just "significant".
- Tests: two-proportion z-test (conversion rates), Welch's t-test (means,
  unequal variances), Mann-Whitney U (ranks), chi-square (counts across
  categories), paired tests / McNemar (two models on the same test set).
- Bootstrap: resample with replacement to get a CI for any statistic
  (median, AUC, revenue per user) without distribution assumptions.
- Randomisation unit (user, session, store) must match the analysis unit.
- Multiple testing: many metrics or variants inflate false positives; use
  Holm or Benjamini-Hochberg corrections, and pick one primary metric up front.
- Variance reduction: CUPED (regress out pre-experiment behaviour) shortens tests.

## When to use / scenarios
- Product/e-commerce: new ranking model, recommendation widget, pricing page.
- Marketing: email subject lines, LLM-generated copy vs human copy.
- ML evaluation: is model B's 0.4-point accuracy gain over A real? Bootstrap
  the test set or run McNemar on paired predictions.
- LLM apps: compare prompt versions on an eval set with CIs before shipping.
- NOT for: questions where you cannot randomise (use
  [[causal-inference-and-uplift]]), or when traffic is too small to reach
  power in a reasonable time (use qualitative review or a bandit).

## Setup & code
```bash
pip install scipy statsmodels
```
```python
import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest, confint_proportions_2indep

# 1) Sample size BEFORE the test: detect 10% -> 11% conversion, alpha 0.05, power 0.8
es = proportion_effectsize(0.11, 0.10)
n = NormalIndPower().solve_power(effect_size=es, alpha=0.05, power=0.8, ratio=1.0)
print("users per arm:", int(np.ceil(n)))

# 2) Analyse the result
conv = np.array([1_620, 1_500])     # B, A conversions
users = np.array([14_800, 14_750])
z, p = proportions_ztest(conv, users)
lo, hi = confint_proportions_2indep(conv[0], users[0], conv[1], users[1], method="wald")
print(f"A={conv[1]/users[1]:.4f} B={conv[0]/users[0]:.4f} z={z:.2f} p={p:.4f} diff 95% CI=[{lo:.4f}, {hi:.4f}]")

# 3) Continuous metric (revenue per user): Welch t-test, plus bootstrap CI for skewed data
rng = np.random.default_rng(0)
a = rng.lognormal(3.0, 1.0, 5000)
b = rng.lognormal(3.03, 1.0, 5000)
print("welch:", stats.ttest_ind(b, a, equal_var=False))
res = stats.bootstrap((b, a), lambda x, y: x.mean() - y.mean(), n_resamples=2000, random_state=0)
print("bootstrap diff CI:", np.round(res.confidence_interval, 2))
```
Output (scipy 1.18.1, statsmodels 0.15.0): 14,745 users per arm are needed
to detect 10% -> 11%. The observed test gives A = 10.17%, B = 10.95%,
z = 2.17, p = 0.030, difference CI [0.08, 1.48] percentage points. On the
revenue data a true 3% lift gives Welch p = 0.050 and a bootstrap CI of
[-0.01, 3.24]: borderline, so the honest answer is "not proven yet".

## Choosing / trade-offs
- Fixed-horizon test (compute n, run to n, analyse once): simplest and valid.
  Peeking daily and stopping at p < 0.05 inflates false positives; use
  sequential tests (mSPRT, alpha spending) if you must monitor.
- Frequentist vs Bayesian A/B: Bayesian gives "P(B better) = 97%", easier to
  explain, but still needs a stopping rule and sensible priors.
- A/B vs multi-armed bandit: bandits move traffic to the winner sooner (less
  regret) but give weaker effect estimates; use for short-lived choices
  (headlines), A/B for decisions you need to defend.
- Build vs buy: GrowthBook (open source), Statsig, Optimizely, Eppo, or
  in-house with the code above for low volume.

## Gotchas
- Sample ratio mismatch: a 50/50 split arriving as 50.8/49.2 on large
  traffic signals a bug in assignment or logging; check with chi-square first.
- Novelty and primacy effects: early lift often fades; run at least a full
  weekly cycle.
- Unit mismatch: randomising by user but computing per-session CIs
  understates variance (sessions of one user are correlated).
- Ratio metrics (revenue per session) need the delta method or bootstrap.
- "Not significant" is not "no effect": check the CI and whether the test had power.
- Network effects (marketplaces, social) break independence between arms;
  use cluster or switchback designs.
- Comparing two models on the same test set: use paired tests, not
  independent-sample tests.

## Related
- [[causal-inference-and-uplift]] - when you cannot randomise, or want per-user effects.
- [[model-evaluation-and-metrics]] - which metric to test.
- [[bayesian-and-gaussian-processes]] - Bayesian reasoning about uncertainty.
- [[recommender-systems]] - offline metrics need online confirmation.
- [[llm-evaluation]] - CIs on prompt/model comparisons.
- [[ecommerce-retail]], [[marketing-content]] - main users of A/B tests.

## References
- SciPy stats: https://docs.scipy.org/doc/scipy/reference/stats.html
- statsmodels stats (power, proportions): https://www.statsmodels.org/stable/stats.html
- Kohavi, Tang, Xu, Trustworthy Online Controlled Experiments (Cambridge, 2020); material at https://exp-platform.com/
