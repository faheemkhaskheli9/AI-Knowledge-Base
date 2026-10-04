---
title: Model selection and comparison (nested CV, significance tests)
category: ml
tags: [model-selection, nested-cross-validation, cross-validation, corrected-t-test, mcnemar, statistical-significance, repeated-cv, scikit-learn, evaluation]
use_cases:
  - "report an honest accuracy for a model I tuned with cross-validation"
  - "decide whether model A is really better than model B or it is noise"
  - "pick between several algorithms on a small dataset"
  - "compare two classifiers that were each evaluated on one fixed test set"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
  - https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html
  - https://scikit-learn.org/stable/modules/cross_validation.html
  - https://www.jmlr.org/papers/v11/cawley10a.html
  - https://doi.org/10.1023/A:1024068626366
---

# Model selection and comparison (nested CV, significance tests)

## Summary
Two questions get mixed up in most projects: **which model/hyperparameters should I pick?** (selection) and **how well will the picked model do on new data?** (estimation). Using the same CV scores for both is optimistic, because the best of many noisy scores is biased upward. Nested cross-validation separates the two. Comparing two models also needs care: CV fold scores are not independent, so a plain paired t-test reports differences as significant far too often. A corrected test, or a properly held-out test set, answers "is A really better than B?".

## Key concepts
- **Selection bias.** `GridSearchCV.best_score_` is the maximum over many candidates' CV scores. With many candidates or a small dataset it overstates performance. Cawley & Talbot (2010) show this bias can be as large as the differences between algorithms.
- **Nested CV.** An inner CV loop tunes hyperparameters. An outer CV loop scores the whole "tune then fit" procedure on folds the inner loop never saw. The outer score estimates the performance of your *procedure*. The final model is then refit with the inner search on all data.
- **Paired comparison.** Score both models on the **same** folds and analyse the per-fold differences. This removes the fold-to-fold variance that both models share.
- **Correlated folds.** In k-fold CV, training sets overlap by (k-2)/(k-1), so fold scores are positively correlated. The naive paired t-test underestimates variance and gives too-small p-values (Dietterich, 1998).
- **Corrected resampled t-test (Nadeau & Bengio, 2003).** Inflate the variance of the differences by `(1/k + n_test/n_train)` instead of `1/k`. It is the default choice for comparing two models with repeated k-fold CV.
- **McNemar's test.** For two classifiers scored on one fixed test set: count the cases where only A is right vs only B is right and test whether those counts differ. It needs just one training run per model.
- **Practical vs statistical significance.** A 0.1-point AUC gain can be significant on a huge test set and still not be worth the extra complexity, latency or cost.

## When to use / scenarios
- Small and medium datasets (hundreds to tens of thousands of rows) where a single train/test split is too noisy: medical studies, industrial sensors, churn on a small customer base.
- Reporting results in a paper, a model card or to stakeholders, where the number must not be inflated by tuning.
- Choosing between algorithm families (linear vs boosting vs neural net) before investing in one.
- A/B-style decisions about replacing a production model with a challenger.
- Not needed for large datasets where a single big held-out test set has small variance. Use train/validation/test splits and a bootstrap confidence interval ([[model-evaluation-and-metrics]]).
- Time series need time-ordered outer and inner splits (`TimeSeriesSplit`), never shuffled folds ([[data-leakage-and-validation-splits]]).

## Setup & code
```bash
pip install "scikit-learn>=1.4" scipy numpy
```
```python
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)

# 1) Nested CV: inner loop tunes, outer loop scores the whole tuning procedure.
inner = StratifiedKFold(5, shuffle=True, random_state=0)
outer = StratifiedKFold(5, shuffle=True, random_state=1)
search = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                      {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}, cv=inner)
search.fit(X, y)
print("non-nested (optimistic):", round(search.best_score_, 4))
print("nested estimate:        ", round(cross_val_score(search, X, y, cv=outer).mean(), 4))

# 2) Same folds for both models, then the corrected paired t-test.
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
a = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), X, y, cv=cv)
b = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=0), X, y, cv=cv)
d = a - b
k = len(d)
n_test, n_train = len(y) / 10, len(y) * 9 / 10
t = d.mean() / np.sqrt((1 / k + n_test / n_train) * d.var(ddof=1))
p = 2 * stats.t.sf(abs(t), df=k - 1)
print(f"LR {a.mean():.4f} vs RF {b.mean():.4f}  corrected p={p:.3f}  "
      f"naive p={stats.ttest_rel(a, b).pvalue:.3f}")
```
Output (scikit-learn 1.9): non-nested 0.9789, nested 0.9754. LR 0.9766 vs RF 0.9637, corrected p = 0.093, naive p = 0.001. The naive test calls the gap highly significant. The corrected test says it could be noise.

McNemar on a fixed test set, given predictions `pa`, `pb` and labels `yt`:
```python
from statsmodels.stats.contingency_tables import mcnemar
table = [[np.sum((pa == yt) & (pb == yt)), np.sum((pa == yt) & (pb != yt))],
         [np.sum((pa != yt) & (pb == yt)), np.sum((pa != yt) & (pb != yt))]]
print(mcnemar(table, exact=True).pvalue)
```

## Choosing / trade-offs
- **Nested CV vs a held-out test set.** Nested CV uses all data for both tuning and estimation and is better on small data. It costs `outer × inner × candidates` fits. With plenty of data, a single locked test set is simpler and cheaper.
- **Which test.** Repeated k-fold available: corrected resampled t-test. One fixed test set and two classifiers: McNemar. Many models over many datasets (benchmarking): Friedman test plus a post-hoc Nemenyi test (Demšar, 2006). Want a probability that A beats B by a margin that matters: Bayesian correlated t-test (shown in the scikit-learn example in References).
- **Repeats.** More repeats of k-fold reduce the variance of the mean score but do not remove fold correlation. That is why the correction is still needed.
- **Choose the simpler model on ties.** If the difference is within noise, prefer the cheaper, faster, more interpretable model (the "one standard error rule").

## Gotchas
- **Reporting `best_score_` as the expected performance.** It is the selection score, not an estimate. Report the nested or held-out score.
- **Different folds for each model.** Comparing means from different splits mixes split noise into the comparison. Pass the same `cv` object, with a fixed `random_state`, to every model.
- **Preprocessing outside the inner loop.** Scaling, feature selection or target encoding fitted on all data before nested CV leaks information into every fold. Put them inside the pipeline being searched.
- **Testing many pairs.** Ten pairwise tests at p < 0.05 will throw up false positives. Correct for multiple comparisons (Holm, Bonferroni) or use Friedman plus post-hoc ([[statistics-and-ab-testing]]).
- **Peeking at the test set.** Every time a decision is made after looking at test scores, the test set becomes part of selection. Keep one final evaluation for the very end.
- **The nested score belongs to the procedure, not one model.** Each outer fold may pick different hyperparameters. That variation is part of what is being measured.

## Related
- [[hyperparameter-tuning]] - the inner-loop search that nested CV wraps.
- [[data-leakage-and-validation-splits]] - picking group-aware and time-aware splitters for both loops.
- [[model-evaluation-and-metrics]] - which metric to compare, and bootstrap confidence intervals.
- [[statistics-and-ab-testing]] - hypothesis testing basics, multiple comparisons, McNemar.
- [[ml-fundamentals]] - train/validation/test splits and the bias-variance trade-off.

## References
- scikit-learn, nested vs non-nested CV: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- scikit-learn, statistical comparison of models (corrected t-test, Bayesian): https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html
- Cawley & Talbot, "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation", JMLR 11 (2010): https://www.jmlr.org/papers/v11/cawley10a.html
- Nadeau & Bengio, "Inference for the Generalization Error", Machine Learning 52 (2003): https://doi.org/10.1023/A:1024068626366
- Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms", Neural Computation 10 (1998): https://doi.org/10.1162/089976698300017197
- Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets", JMLR 7 (2006): https://www.jmlr.org/papers/v7/demsar06a.html
