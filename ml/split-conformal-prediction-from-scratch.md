---
title: Split conformal prediction from scratch (coverage guarantee, finite-sample quantile, adaptive intervals)
category: ml
tags: [conformal-prediction, prediction-intervals, uncertainty-quantification, calibration, coverage, heteroscedasticity, distribution-free, numpy, from-scratch, ml-basics]
use_cases:
  - "wrap any regression model with prediction intervals that have guaranteed coverage, in NumPy"
  - "see why the (n+1) finite-sample quantile correction matters for small calibration sets"
  - "make conformal intervals adapt to heteroscedastic noise with a normalized score"
  - "explain marginal vs conditional coverage of conformal prediction in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/2107.07511
  - https://doi.org/10.1080/01621459.2017.1307116
  - https://arxiv.org/abs/1905.03222
  - https://mapie.readthedocs.io/
---

# Split conformal prediction from scratch (coverage guarantee, finite-sample quantile, adaptive intervals)

## Summary
Split conformal prediction turns any point predictor into one that outputs intervals with a guaranteed coverage rate. It makes no assumption about the model or the noise, only that calibration and test data are exchangeable (for example, i.i.d.). The recipe is three steps. Compute a nonconformity score on a held-out calibration set, for example `|y − ŷ|`. Take its `⌈(n+1)(1−α)⌉`-th smallest value `q`. Predict `ŷ ± q`. Below, a polynomial regression on data whose noise grows with `x` gets 90% intervals. Absolute-residual conformal hits 90.2% overall but uses one width everywhere: it covers 100% where the noise is small and only 73% where it is large. Dividing the score by a fitted noise scale `σ(x)` gives 88.5% overall and 90.3% in the noisy region, with intervals 15% narrower on average. With 20 calibration points, the plain `np.quantile` gives 86.4% coverage, and the `(n+1)` correction restores 90.3%.

## Key concepts
- **Guarantee.** If calibration and test points are exchangeable, `P(y_test ∈ C(x_test)) ≥ 1 − α`. This is marginal coverage, averaged over test points and over the random calibration draw. It is not a guarantee for every `x`.
- **Nonconformity score.** Any function measuring how unusual `(x, y)` is under the model. For regression, `|y − ŷ(x)|` or `|y − ŷ(x)| / σ̂(x)`. For classification, `1 − p̂_y(x)` (prediction sets) or the cumulative APS score.
- **Finite-sample quantile.** Use the `⌈(n+1)(1−α)⌉`-th smallest score, not the empirical `(1−α)` quantile. The `+1` accounts for the test point itself. If that rank exceeds `n` (tiny n, tiny α), the interval is infinite.
- **Split.** The model, the scale model and the calibration scores must use disjoint data. Scoring on training data underestimates residuals and breaks the guarantee.
- **Coverage is random.** For a fixed calibration set, test coverage follows a Beta distribution centred just above `1 − α`. Its spread shrinks roughly as `1/√n`, so `n` of a few hundred to a thousand gives tight coverage.

## When to use / scenarios
- Learning: the simplest way to get honest uncertainty from a black-box model, and a clean demonstration of exchangeability-based reasoning.
- Interviews: "how do you get prediction intervals from XGBoost", "what does conformal guarantee and what not", "marginal vs conditional coverage".
- Practice: demand and price forecasts with intervals for planning, flagging predictions too uncertain to automate (route wide intervals to a human), prediction sets for classifiers in medical or moderation triage, and LLM answer filtering with a calibrated abstention threshold ([[conformal-prediction-and-uncertainty]]).
- Not for: time series or data with distribution shift, where exchangeability fails (use adaptive or weighted conformal, or retrain the calibration on recent windows); guarantees per individual or subgroup (use group-conditional or Mondrian conformal); or very small datasets where you cannot afford a held-out calibration split (use cross-conformal / jackknife+).

## Setup & code
NumPy only. Runs in about two seconds.

```python
import numpy as np

rng = np.random.default_rng(0)


def make(n):
    x = rng.uniform(0, 10, n)
    y = np.sin(x) * 3 + 0.5 * x + rng.normal(0, 0.1 + 0.3 * x)   # noise grows with x
    return x, y


def conformal_q(scores, alpha):
    """Finite-sample corrected (1-alpha) quantile of calibration scores."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]


# 1) train a point model, 2) train a scale model, 3) calibrate on held-out data
x_tr, y_tr = make(1000)
mean_coef = np.polyfit(x_tr, y_tr, 7)
mu = lambda x: np.polyval(mean_coef, x)
x_s, y_s = make(1000)
scale_coef = np.polyfit(x_s, np.abs(y_s - mu(x_s)), 2)          # predicts |residual|
sigma = lambda x: np.maximum(np.polyval(scale_coef, x), 0.05)

x_cal, y_cal = make(500)
x_te, y_te = make(20000)
alpha = 0.1

q_abs = conformal_q(np.abs(y_cal - mu(x_cal)), alpha)
q_norm = conformal_q(np.abs(y_cal - mu(x_cal)) / sigma(x_cal), alpha)
methods = {"absolute residual": (q_abs * np.ones_like(x_te)),
           "normalised by sigma(x)": (q_norm * sigma(x_te))}

print(f"target coverage {1 - alpha:.0%}, calibration n={len(x_cal)}")
print("\nmethod                   coverage  mean width  coverage x<2  coverage x>8  width x<2  width x>8")
for name, half in methods.items():
    cov = np.abs(y_te - mu(x_te)) <= half
    lo, hi = x_te < 2, x_te > 8
    print(f"{name:<24} {cov.mean():>8.3f}  {2 * half.mean():>10.2f}  {cov[lo].mean():>12.3f}  {cov[hi].mean():>12.3f}"
          f"  {2 * half[lo].mean():>9.2f}  {2 * half[hi].mean():>9.2f}")

# coverage is a random variable over the calibration draw: repeat the calibration
covs = {n: [] for n in [20, 100, 1000]}
for n in covs:
    for _ in range(500):
        xc, yc = make(n)
        q = conformal_q(np.abs(yc - mu(xc)), alpha)
        xt, yt = make(2000)
        covs[n].append(np.mean(np.abs(yt - mu(xt)) <= q))
print("\ncoverage over 500 calibration draws (absolute residual):")
for n, c in covs.items():
    c = np.array(c)
    print(f"  n_cal={n:>4}: mean {c.mean():.3f}  5th-95th pct {np.percentile(c, 5):.3f}-{np.percentile(c, 95):.3f}")

naive = []
for _ in range(500):
    xc, yc = make(20)
    q = np.quantile(np.abs(yc - mu(xc)), 1 - alpha)
    xt, yt = make(2000)
    naive.append(np.mean(np.abs(yt - mu(xt)) <= q))
print(f"\nnaive np.quantile with n_cal=20: mean coverage {np.mean(naive):.3f} (corrected: {np.mean(covs[20]):.3f})")
```

Output (Python 3.14, NumPy 2.5):
```
target coverage 90%, calibration n=500

method                   coverage  mean width  coverage x<2  coverage x>8  width x<2  width x>8
absolute residual           0.902        6.20         1.000         0.733       6.20       6.20
normalised by sigma(x)      0.885        5.25         0.851         0.903       1.42       9.21

coverage over 500 calibration draws (absolute residual):
  n_cal=  20: mean 0.903  5th-95th pct 0.781-0.981
  n_cal= 100: mean 0.901  5th-95th pct 0.845-0.946
  n_cal=1000: mean 0.899  5th-95th pct 0.881-0.918

naive np.quantile with n_cal=20: mean coverage 0.864 (corrected: 0.903)
```

Both methods deliver their promise on average, but only the normalized score spends its width where the noise is. The absolute-residual interval is 6.20 wide at every `x`. That is far too wide for small `x`, where coverage is 100% and the intervals are uninformative, and too narrow for large `x`, where only 73% of points are covered. That is a marginal guarantee hiding a conditional failure. With the `σ(x)` score the interval is 1.42 wide at small `x` and 9.21 at large `x`, and the noisy region is covered at 90.3%. Its 88.5% overall coverage is within the calibration-draw spread for n = 500 (about ±1.3 percentage points). The 85.1% at small `x` shows that the scale model's quality now matters: a quadratic fit to `|residual|` is rough there. Only the marginal guarantee survives a poor scale model, not the conditional behavior. The repeated-draw table shows the guarantee as it really is. The average is 90%, but a single 20-point calibration set can give anywhere from 78% to 98%. The naive quantile undercovers by 3.6 points at n = 20, which is exactly the gap the `(n+1)` correction closes.

## Choosing / trade-offs
- **Score choice.** Absolute residual is simplest and gives constant width. Normalized residual (`/σ̂(x)`) adapts to heteroscedasticity but needs a second model. Conformalized quantile regression (CQR) calibrates the gap between predicted lower and upper quantiles, and is usually the best default for adaptive intervals ([[quantile-regression]]).
- **Split vs cross-conformal / jackknife+.** Split conformal is cheap and exact but spends data on calibration. Jackknife+ and CV+ reuse all data for both fitting and calibration, at the cost of K model fits and a slightly weaker (1 − 2α) worst-case guarantee.
- **Conformal vs Bayesian / GP intervals.** Bayesian intervals are conditional and model-based: correct if the model is right, wrong otherwise ([[gaussian-process-regression-from-scratch]], [[bayesian-linear-regression-from-scratch]]). Conformal intervals are marginal and assumption-light. Conformalizing a Bayesian model's interval gives both adaptivity and the guarantee.
- **Libraries.** MAPIE (scikit-learn compatible) and `crepes` implement split, CV+, CQR and classification sets. Use them in production; the scratch version is for understanding and for unusual scores.

## Gotchas
- Exchangeability is the whole assumption. Time series, drift, or calibrating on last year and predicting this year voids the guarantee. Monitor realized coverage in production, as you would a metric ([[online-learning-and-concept-drift]]).
- Never calibrate on training data. In-sample residuals are too small, and coverage collapses.
- The guarantee is marginal. Report coverage per segment (region, customer type, value range) before trusting intervals for decisions about a subgroup.
- `np.quantile(scores, 1 − α)` is not the conformal quantile. Use the `⌈(n+1)(1−α)⌉`-th order statistic, and handle the infinite case for tiny `n`.
- Floor the scale model, as `np.maximum(..., 0.05)` does here. A `σ̂(x)` near zero divides scores into huge values and makes those intervals meaningless.
- Coverage of 90% with very wide intervals is useless. Report average width (or set size) together with coverage, and compare methods at equal coverage.

## Related
- [[conformal-prediction-and-uncertainty]] - conformal methods with scikit-learn and MAPIE, including classification sets.
- [[quantile-regression]] - quantile models that CQR calibrates.
- [[probability-calibration]] - calibrated probabilities, a different notion from interval coverage.
- [[regression-metrics-and-residual-analysis]] - diagnosing heteroscedastic residuals.
- [[gaussian-process-regression-from-scratch]] - model-based intervals for comparison.
- [[bayesian-and-gaussian-processes]] - Bayesian uncertainty more generally.

## References
- Angelopoulos and Bates (2021), "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification": https://arxiv.org/abs/2107.07511
- Lei, G'Sell, Rinaldo, Tibshirani and Wasserman (2018), "Distribution-Free Predictive Inference for Regression", JASA: https://doi.org/10.1080/01621459.2017.1307116
- Romano, Patterson and Candès (2019), "Conformalized Quantile Regression": https://arxiv.org/abs/1905.03222
- MAPIE documentation: https://mapie.readthedocs.io/
