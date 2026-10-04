---
title: Generalized additive models (GAMs), splines and glass-box models
category: ml
tags: [gam, generalized-additive-models, splines, b-splines, spline-transformer, periodic-features, pygam, explainable-boosting-machine, interpret, glass-box, scikit-learn]
use_cases:
  - "model a non-linear effect (U-shaped age risk, saturating ad spend) while staying interpretable"
  - "encode time of day, day of year or angles so the end wraps to the start"
  - "build a model whose every feature effect can be plotted and signed off by a regulator or clinician"
  - "beat a linear model on tabular data without switching to a black-box ensemble"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/preprocessing.html#spline-transformer
  - https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html
  - https://pygam.readthedocs.io/
  - https://interpret.ml/docs/ebm.html
---

# Generalized additive models (GAMs), splines and glass-box models

## Summary
A GAM replaces each linear term `w*x` with a smooth function `f(x)`: `g(E[y]) = b + f1(x1) + f2(x2) + ...`. Each `f` is learned from data (usually a penalised spline), so the model captures curves, thresholds and saturation, yet stays additive: you can plot every feature's effect and read it exactly. In practice you can get most of a GAM in scikit-learn by expanding features with `SplineTransformer` and fitting a regularised linear model or GLM. pyGAM adds automatic smoothness selection; Explainable Boosting Machines (EBM, from `interpret`) fit GAM-shaped models with boosting and often match black-box accuracy on tabular data.

## Key concepts
- **Basis expansion.** A spline turns one column into `n_knots + degree - 1` B-spline columns; a linear model on those columns is a smooth curve in the original feature. Same idea as polynomial features, but local: moving one knot does not wiggle the whole curve.
- **Knots and degree.** More knots = more flexible. Cubic (`degree=3`) is standard. Place knots at quantiles (`knots="quantile"`) when data is uneven.
- **Smoothness penalty.** Too many knots overfit; a penalty (ridge on spline weights, or pyGAM's second-derivative penalty `lam`) keeps curves smooth. Tune it by CV.
- **Periodic splines.** `extrapolation="periodic"` makes 23:59 and 00:00 neighbours; the right encoding for hour, weekday, day-of-year, wind direction.
- **Additivity = interpretability.** Because terms add up, the partial dependence of one feature *is* its shape function, with no averaging tricks. The cost: no interactions unless you add them explicitly (tensor terms `te(i, j)` in pyGAM, pairwise terms in EBM).
- **Link function.** A GAM is a GLM with smooth terms: logistic GAM for binary, Poisson GAM for counts ([[generalized-linear-models]]).
- **Glass-box vs post-hoc explanations.** A GAM's plots are the model itself; SHAP on a boosted ensemble is an approximation of a model you cannot read directly ([[model-interpretability]]).

## When to use / scenarios
- Credit scoring, insurance rating and clinical risk scores where each effect must be reviewed and monotone/sensible ([[finance]], [[healthcare]]).
- Energy and demand models with daily/weekly/yearly cycles (periodic splines) ([[manufacturing-iot]]).
- Marketing mix: saturating response to spend.
- Any tabular problem where a linear model underfits and you want to see *why* before reaching for boosting.
- NOT when strong high-order interactions drive the signal (images, text, complex user behaviour): use boosting or neural nets ([[gradient-boosting-tabular]]).
- NOT for very high-dimensional sparse data (bag-of-words): spline expansion multiplies the column count.

## Setup & code
```bash
pip install scikit-learn numpy        # optional: pygam, interpret
```
A spline GAM vs a linear model on data with a U-shaped age effect and a daily cycle:
```python
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, OneHotEncoder

rng = np.random.default_rng(0)
n = 2000
age = rng.uniform(18, 90, n)
hour = rng.uniform(0, 24, n)
region = rng.integers(0, 3, n)
y = 0.002 * (age - 50) ** 2 + 2 * np.sin(2 * np.pi * hour / 24) + np.array([0, 1, -1])[region] + rng.normal(0, 1, n)
X = np.column_stack([age, hour, region])

linear = make_pipeline(
    make_column_transformer((OneHotEncoder(), [2]), remainder="passthrough"),
    RidgeCV())
gam = make_pipeline(
    make_column_transformer(
        (SplineTransformer(n_knots=8, degree=3), [0]),
        (SplineTransformer(n_knots=8, degree=3, extrapolation="periodic"), [1]),  # 23:59 next to 00:00
        (OneHotEncoder(), [2])),
    RidgeCV(alphas=np.logspace(-3, 3, 13)))

for name, m in [("linear", linear), ("spline GAM", gam)]:
    print("%-11s CV R2 %.3f" % (name, cross_val_score(m, X, y, cv=5, scoring="r2").mean()))

# read the learned age shape: vary age, hold others fixed (additive, so this IS the age term)
gam.fit(X, y)
ages = [20, 35, 50, 65, 80]
grid = np.column_stack([ages, np.full(5, 6.0), np.zeros(5)])
eff = gam.predict(grid)
print("age effect (relative to age 50):", dict(zip(ages, (eff - eff[2]).round(2).tolist())))
print("true effect:                    ", dict(zip(ages, (0.002 * (np.array(ages) - 50) ** 2).round(2).tolist())))
```
Output with scikit-learn 1.9.0:
```text
linear      CV R2 0.436
spline GAM  CV R2 0.767
age effect (relative to age 50): {20: 1.87, 35: 0.6, 50: 0.0, 65: 0.48, 80: 1.75}
true effect:                     {20: 1.8, 35: 0.45, 50: 0.0, 65: 0.45, 80: 1.8}
```
The linear model cannot represent the U-shape or the cycle; the spline model recovers both and its age curve is directly readable. Swap `RidgeCV` for `LogisticRegressionCV` (binary) or `PoissonRegressor` (counts) to get a logistic or Poisson GAM.

Dedicated libraries (not run here; check their docs for current versions):
```python
from pygam import LinearGAM, s, f              # pip install pygam
gam = LinearGAM(s(0) + s(1, basis="cp") + f(2)).gridsearch(X, y)   # cp = cyclic spline; picks lam by GCV
gam.summary()

from interpret.glassbox import ExplainableBoostingRegressor  # pip install interpret
ebm = ExplainableBoostingRegressor(interactions=5).fit(X, y)   # adds top pairwise interactions
# from interpret import show; show(ebm.explain_global())      # per-feature shape plots
```

## Choosing / trade-offs
- **SplineTransformer + linear model.** No new dependency, fits in any scikit-learn pipeline, you pick knots and penalty by CV. No per-term smoothness or confidence bands.
- **pyGAM.** Classical GAM: per-term penalties chosen by GCV, confidence intervals, tensor-product interactions, link functions. Slower on large data.
- **EBM (interpret).** Boosted shape functions with automatic pairwise interactions; usually the most accurate glass-box option on tabular data, at the cost of longer training and step-shaped (not smooth) curves.
- **GAM vs boosting + SHAP.** If the effect plots must *be* the model (audit, regulation, sign-off), use a GAM/EBM. If accuracy dominates and approximate explanations are acceptable, boosting.
- **Splines vs binning.** Binning (`KBinsDiscretizer`) is simpler and common in scorecards but jumps at bin edges; splines are smooth and use fewer parameters for the same fit.

## Gotchas
- Splines extrapolate poorly: with the default `extrapolation="constant"` effects go flat outside the training range; `"linear"` continues the slope. Decide which is safe for your feature.
- Too many knots without a penalty gives wiggly curves that look like real effects. Always regularise and tune by CV.
- Periodic splines need the period to match the data range: set `knots` explicitly (e.g. `np.linspace(0, 24, 9).reshape(-1, 1)`) if your observed hours do not span exactly 0-24.
- Shape plots of correlated features trade credit between them; two plausible-looking curves can both be wrong. Check stability across CV folds.
- Additivity hides interactions: if the effect of age differs by product, a plain GAM averages it away. Add an explicit interaction term.
- Scaling: spline columns are already bounded in [0, 1], but mixing them with unscaled raw features in one ridge penalty distorts the regularisation.

## Related
- [[generalized-linear-models]] - the link functions and distributions GAMs build on.
- [[linear-models]] - the linear base; splines are a feature expansion for it.
- [[feature-engineering]] - polynomial, binned and cyclical features as alternatives.
- [[model-interpretability]] - PDP and SHAP, the post-hoc counterparts to glass-box shapes.
- [[gradient-boosting-tabular]] - the black-box alternative when interactions dominate.
- [[time-series-forecasting]] - seasonal effects modelled with periodic splines.
- [[finance]] - scorecards and rating models that need reviewable effects.

## References
- scikit-learn, SplineTransformer: https://scikit-learn.org/stable/modules/preprocessing.html#spline-transformer
- scikit-learn example, time-related feature engineering (periodic splines): https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html
- pyGAM docs: https://pygam.readthedocs.io/
- InterpretML, Explainable Boosting Machine: https://interpret.ml/docs/ebm.html
