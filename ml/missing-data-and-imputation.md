---
title: Missing data and imputation
category: ml
tags: [missing-data, imputation, mcar, mar, mnar, simpleimputer, iterativeimputer, knnimputer, multiple-imputation, missing-indicator, scikit-learn, pandas]
use_cases:
  - "train a model on a table where many cells are empty"
  - "decide whether to drop rows, fill values or let the model handle NaN"
  - "fill gaps in sensor, survey or medical records without biasing the result"
  - "report uncertainty from imputation in a statistical analysis"
  - "handle values that are missing at prediction time in production"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/impute.html
  - https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
  - https://stefvanbuuren.name/fimd/
  - https://www.statsmodels.org/stable/imputation.html
---

# Missing data and imputation

## Summary
Real tables have holes: unanswered survey questions, sensors that dropped out, lab tests nobody ordered. Most scikit-learn models refuse NaN, so you must drop, fill (impute) or use a model that handles missing values natively. The right choice depends on *why* data is missing: if missingness is random, simple fills are fine; if it depends on the value itself (rich people skip the income question), every fill is biased and the missingness pattern is itself information. For prediction, median/most-frequent fill plus a "was missing" indicator, or a gradient boosting model with native NaN support, is the strong default; for statistical inference, use multiple imputation.

## Key concepts
- **Missingness mechanisms** (Rubin):
  - **MCAR** (completely at random): unrelated to anything. Dropping rows loses data but does not bias.
  - **MAR** (at random, given observed data): e.g. older patients skip a test more often, and age is recorded. Model-based imputation using the other columns is unbiased.
  - **MNAR** (not at random): depends on the missing value itself. No imputation fixes it from the data alone; add indicators, collect the reason, or do sensitivity analysis.
- **Deletion.** Listwise (drop rows with any NaN) is simple but wastes data and biases under MAR/MNAR; drop a *column* when it is mostly empty and not predictive.
- **Simple imputation.** Mean/median (numbers), most-frequent or a constant `"missing"` category (categories). Shrinks variance and weakens correlations, but fine for prediction.
- **Missing indicator.** A 0/1 column per feature saying "this was missing" (`add_indicator=True`). Lets the model learn that missingness is predictive, which is common (a blank "last purchase date" means "never bought").
- **Model-based.** `KNNImputer` (average of nearest rows), `IterativeImputer` (MICE-style: regress each column on the others, round-robin). Better for MAR, slower, can overfit.
- **Multiple imputation.** Create m imputed datasets with randomness, fit the analysis on each, combine with Rubin's rules. Needed when you report coefficients, p-values or confidence intervals; single imputation makes them falsely precise.
- **Native NaN handling.** HistGradientBoosting, XGBoost, LightGBM and CatBoost learn which branch missing values go to ([[gradient-boosting-tabular]]); no imputation needed.
- **Time series.** Forward-fill, interpolation or seasonal fills that only use the past; never fill with future values ([[time-series-forecasting]]).

## When to use / scenarios
- Healthcare records: tests are ordered selectively (MAR/MNAR); use indicators and model-based imputation, and multiple imputation for clinical studies ([[healthcare]]).
- Surveys and customer data: skipped questions are often informative; constant "missing" category plus indicator.
- IoT and manufacturing sensors: dropouts are short gaps in a series; interpolate within a limit and flag long gaps ([[manufacturing-iot]]).
- Credit and finance: missing income or history is a signal of risk; keep it as a feature, do not just fill it away ([[finance]]).
- Production inference: a field present in training may arrive empty at serving; the fitted imputer in the pipeline must cover it.
- NOT worth model-based imputation when a tree-boosting model with native NaN support is the final model; let it handle the gaps.

## Setup & code
```bash
pip install scikit-learn pandas numpy
```
Inject 30% missingness into one feature, under MCAR and under MNAR (large values hidden), and compare imputers inside a pipeline:
```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (activates IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X, y = fetch_california_housing(return_X_y=True)
idx = rng.choice(len(X), 4000, replace=False)              # rows are sorted by region: sample randomly
X, y = X[idx][:, [0, 1, 6, 7]], y[idx]                     # income, age, lat, lon (no outlier-heavy cols)

def with_missing(X, mnar):
    X = X.copy()
    col = X[:, 0]                                           # median income
    if mnar:
        p = (col > np.quantile(col, 0.6)) * 0.75            # high incomes go missing
    else:
        p = np.full(len(col), 0.3)
    X[rng.random(len(col)) < p, 0] = np.nan
    return X

imputers = {
    "median": SimpleImputer(strategy="median"),
    "median + indicator": SimpleImputer(strategy="median", add_indicator=True),
    "knn": KNNImputer(n_neighbors=10),
    "iterative": IterativeImputer(random_state=0, max_iter=10),
}
for mnar in (False, True):
    Xm = with_missing(X, mnar)
    print("MNAR" if mnar else "MCAR", f"- {np.isnan(Xm[:, 0]).mean():.0%} missing")
    for name, imp in imputers.items():
        pipe = make_pipeline(StandardScaler(), imp, Ridge())   # imputer refit per fold
        print(f"  {name:20s} R2 {cross_val_score(pipe, Xm, y, cv=5).mean():.3f}")
    hgb = HistGradientBoostingRegressor(random_state=0)         # native NaN support
    print(f"  {'HGB, no imputation':20s} R2 {cross_val_score(hgb, Xm, y, cv=5).mean():.3f}")
```
Under MCAR the imputers land close together. Under MNAR the indicator variant should clearly beat plain median and even KNN/iterative, because "income is missing" now means "income is high" and no fill from other columns can recover that. Gradient boosting with native NaN handling is ahead in both cases here, mostly because it fits the non-linear lat/lon pattern.

Multiple imputation for inference: run `IterativeImputer(sample_posterior=True, random_state=i)` for i = 1..m, fit the analysis model on each, and pool with Rubin's rules (statsmodels `MICE` does this end-to-end for linear models).

## Choosing / trade-offs
- **Prediction, tree boosting model:** native NaN handling; optionally add indicators for interpretability.
- **Prediction, linear/NN/SVM/kNN model:** median/most-frequent + indicator as the baseline; try `IterativeImputer` or `KNNImputer` if missingness is heavy and MAR.
- **Inference (coefficients, effects, p-values):** multiple imputation; report how many rows had missing values and a sensitivity analysis for MNAR.
- **Drop rows** only when few are affected (a few %) and MCAR is plausible; **drop columns** that are mostly empty *and* show no signal in an indicator.
- **KNN vs iterative.** KNN is simple but slow on large data and needs scaled features; iterative is more flexible (any regressor as the estimator) but can propagate errors and is slower to fit.

## Gotchas
- Fitting the imputer on all data before splitting leaks test statistics; keep it inside the `Pipeline`.
- Mean imputation on skewed data (incomes, counts) lands in an unrealistic spot; use the median.
- Placeholder codes (`-999`, `0`, `""`, `"N/A"`, `"unknown"`) that are not NaN slip past imputers and distort scaling; convert them first (`pd.read_csv(na_values=...)`).
- Imputing the *target* and then training on it creates a model of the imputer; drop rows with a missing target (or use semi-supervised methods).
- An indicator that is always 0 in training but 1 in production (a new outage) is a silent distribution shift; monitor missing rates per column ([[online-learning-and-concept-drift]]).
- Single imputation followed by standard errors makes confidence intervals too narrow.
- `IterativeImputer` is still marked experimental in scikit-learn and needs the `enable_iterative_imputer` import.
- KNN imputation without scaling: distances are dominated by the largest-unit feature.

## Related
- [[feature-engineering]] - imputation as one step in a preprocessing pipeline.
- [[gradient-boosting-tabular]] - models that handle NaN natively.
- [[gaussian-mixture-models-and-em]] - EM, the classic way to fit models with missing values.
- [[label-noise-and-data-cleaning]] - other data-quality problems in the same table.
- [[statistics-and-ab-testing]] - inference where multiple imputation matters.
- [[time-series-forecasting]] - gap filling that respects time order.

## References
- scikit-learn, Imputation of missing values: https://scikit-learn.org/stable/modules/impute.html
- `IterativeImputer` API: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
- van Buuren, *Flexible Imputation of Missing Data* (free online book): https://stefvanbuuren.name/fimd/
- statsmodels, Multiple imputation (MICE): https://www.statsmodels.org/stable/imputation.html
