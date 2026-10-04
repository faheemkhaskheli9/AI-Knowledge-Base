---
title: Exploratory data analysis (EDA) before modelling
category: ml
tags: [eda, exploratory-data-analysis, data-quality, pandas, missing-values, outliers, duplicates, leakage, skew, profiling, ydata-profiling]
use_cases:
  - "look at a new CSV or database table before training anything on it"
  - "find missing values, duplicates, sentinel codes and impossible values in a dataset"
  - "spot target leakage before a model scores suspiciously well"
  - "decide which features need log transforms, encoding or grouping of rare levels"
  - "write a data-quality summary for stakeholders before a modelling project starts"
status: draft
last_verified: 2026-10-04
sources:
  - https://pandas.pydata.org/docs/user_guide/index.html
  - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
  - https://docs.profiling.ydata.ai/
  - https://skrub-data.org/stable/
---

# Exploratory data analysis (EDA) before modelling

## Summary
EDA is the pass over a dataset you do before any model: what each row is, which columns are broken, what the target looks like and which features would not exist at prediction time. Most failed tabular projects fail here, not in the model: a sentinel `999` read as an age, duplicated customers leaking across the split, or a column filled in only after the outcome happened. A one-hour checklist with pandas catches most of it. Profiling tools (ydata-profiling, skrub `TableReport`) speed up the first look but do not replace reading the questions below.

## Key concepts
- **Unit of observation.** Decide what one row is (a customer, an order, a customer-month) and check that the ID is unique at that grain. Duplicated IDs decide your split strategy ([[model-evaluation-and-metrics]], GroupKFold).
- **Missingness is information.** Measure the share per column and ask *why* it is missing. Missing-not-at-random values carry signal; a column missing exactly when the target is 0 is leakage ([[missing-data-and-imputation]]).
- **Sentinels and impossible values.** `-1`, `0`, `999`, `1900-01-01`, empty strings and `"N/A"` hide inside numeric/date columns. `describe()` min/max finds most of them.
- **Distribution shape.** Heavy right skew (money, counts, durations) suggests `log1p` for linear models and neural nets; trees do not care.
- **Cardinality.** Count levels per categorical; rare levels (<1%) get grouped, high-cardinality IDs need target encoding or dropping ([[feature-engineering]]).
- **Target first.** Class balance or target distribution, then target rate per segment. This tells you the baseline and the metric ([[imbalanced-data]]).
- **Leakage check.** For every column ask "is this known at the moment of prediction?". Time stamps after the event, IDs that encode the outcome and aggregates computed over the full dataset all leak.
- **Time.** If rows have dates, plot counts and target over time; drift or a definition change mid-history decides whether you need a time-based split ([[time-series-forecasting]]).

## When to use / scenarios
- Every new tabular dataset: churn, credit, claims, sensor logs, survey exports.
- Before trusting a model with a validation score that looks too good (often leakage).
- When inheriting a dataset from another team with no data dictionary.
- Data-quality audit reports for stakeholders ([[data-analytics]]).
- NOT a substitute for monitoring: EDA describes one snapshot; production drift needs [[online-learning-and-concept-drift]].
- For images, text or audio, the same questions apply (duplicates, label balance, corrupt files) but with different tools ([[label-noise-and-data-cleaning]]).

## Setup & code
```bash
pip install pandas numpy            # optional: ydata-profiling or skrub for an HTML report
```
A minimal checklist on a synthetic churn table with planted problems:
```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 1000
df = pd.DataFrame({
    "customer_id": np.arange(n),
    "age": rng.integers(18, 80, n).astype(float),
    "plan": rng.choice(["basic", "pro", "enterprise"], n, p=[0.7, 0.25, 0.05]),
    "monthly_spend": rng.lognormal(3, 0.8, n).round(2),
    "signup": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 600, n), unit="D"),
})
df.loc[rng.random(n) < 0.12, "age"] = np.nan          # missing values
df.loc[:4, "age"] = 999                                # sentinel / data-entry errors
df = pd.concat([df, df.iloc[:15]], ignore_index=True)  # duplicated rows
df["churned"] = (rng.random(len(df)) < 0.1 + 0.15 * (df["plan"] == "basic")).astype(int)
df["days_since_cancel_request"] = np.where(df["churned"] == 1, rng.integers(0, 30, len(df)), np.nan)  # leak

# 1. shape and grain
print(df.shape, "| dup rows:", df.duplicated().sum(), "| dup ids:", df["customer_id"].duplicated().sum())
# 2. missingness
print(df.isna().mean().round(3).sort_values(ascending=False).head(3).to_dict())
# 3. numeric ranges: max 999 is a sentinel, not an age
print(df[["age", "monthly_spend"]].describe().loc[["min", "50%", "max"]].round(1).to_dict())
# 4. skew suggests a log transform
print("spend skew:", round(df["monthly_spend"].skew(), 2), "-> log skew:", round(np.log1p(df["monthly_spend"]).skew(), 2))
# 5. categorical cardinality and rare levels
print(df["plan"].value_counts(normalize=True).round(3).to_dict())
# 6. target balance and target by segment
print("churn rate:", round(df["churned"].mean(), 3), df.groupby("plan")["churned"].mean().round(3).to_dict())
# 7. leakage smell: a feature missing exactly when the target is 0
print(pd.crosstab(df["days_since_cancel_request"].isna(), df["churned"]))
```
Output with pandas 3.0.5, numpy 2.5.1:
```text
(1015, 7) | dup rows: 12 | dup ids: 15
{'days_since_cancel_request': 0.785, 'age': 0.116, 'customer_id': 0.0}
{'age': {'min': 18.0, '50%': 51.0, 'max': 999.0}, 'monthly_spend': {'min': 1.7, '50%': 18.7, 'max': 173.2}}
spend skew: 2.3 -> log skew: 0.2
{'basic': 0.695, 'pro': 0.262, 'enterprise': 0.043}
churn rate: 0.215 {'basic': 0.262, 'enterprise': 0.114, 'pro': 0.105}
churned                      0    1
days_since_cancel_request
False                        0  218
True                       797    0
```
What it found: 15 duplicated IDs but only 12 identical rows (3 copies carry a *different* label - conflicting duplicates), age max 999, spend skew 2.3 fixed by `log1p`, a 4% "enterprise" level, and `days_since_cancel_request` perfectly separating the target - drop it. Next steps: `df.drop_duplicates("customer_id")`, `df["age"] = df["age"].mask(df["age"] > 110)`, and plot histograms (`df.hist()`) plus target rate over `signup` month.

Automatic report for the first look:
```python
from ydata_profiling import ProfileReport          # pip install ydata-profiling
ProfileReport(df, minimal=True).to_file("eda.html")
# or: from skrub import TableReport; TableReport(df)  # pip install skrub, renders in notebooks
```

## Choosing / trade-offs
- **Manual pandas vs profiling report.** Reports are fast and catch the obvious (missingness, correlations, duplicates); manual checks answer the questions specific to your target and prediction time. Do both for anything that ships.
- **Sample vs full data.** Profile a sample for speed on very large tables, but count duplicates, nulls and min/max on the full data - rare problems hide in samples.
- **How much EDA.** Enough to set the split, the baseline, the metric and the cleaning rules; then build a baseline model and let its errors direct further EDA. Endless plotting before a baseline is a common stall.
- **Where cleaning lives.** One-off fixes (dedupe, sentinel to NaN) in a data-prep script; anything learned from data (imputation values, scaling) inside the model `Pipeline` so it is fitted on train only ([[classic-ml-scikit-learn]]).

## Gotchas
- Doing EDA on the full dataset and then "discovering" features from test-set patterns: the choices themselves leak. Look at the target relationships on the training split.
- `df.describe()` skips non-numeric columns by default; numbers stored as strings (`"1,200"`) silently drop out. Check `df.dtypes` first.
- `pd.concat` keeps the original index; duplicate index labels later break `crosstab`, `reindex` and joins (`cannot reindex on an axis with duplicate labels`). Pass `ignore_index=True`.
- Correlation heatmaps only see linear, numeric relationships; use mutual information or per-bin target rates for the rest ([[information-theory-for-ml]], [[feature-selection]]).
- A feature that is "too good" (near-perfect AUC alone) is leakage until proven otherwise.
- Dropping rows with any NaN can remove a biased subset (e.g. all new customers); check who you drop.
- Profiling reports on data with PII produce an HTML file full of sample values; do not share it casually.

## Related
- [[ml-fundamentals]] - splits, baselines and overfitting that EDA sets up.
- [[missing-data-and-imputation]] - what to do once you know why values are missing.
- [[feature-engineering]] - encoding, transforms and rare-level grouping that EDA motivates.
- [[label-noise-and-data-cleaning]] - conflicting duplicates and wrong labels.
- [[statistics-and-ab-testing]] - tests and confidence intervals for the differences you spot.
- [[model-evaluation-and-metrics]] - picking a metric and split from the target distribution.
- [[data-analytics]] - scenario file for analysis-heavy projects.

## References
- pandas user guide: https://pandas.pydata.org/docs/user_guide/index.html
- `DataFrame.describe`: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html
- ydata-profiling docs: https://docs.profiling.ydata.ai/
- skrub (TableReport): https://skrub-data.org/stable/
