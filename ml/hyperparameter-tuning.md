---
title: Hyperparameter tuning (grid, random, successive halving, Optuna)
category: ml
tags: [hyperparameter-tuning, grid-search, random-search, successive-halving, optuna, bayesian-optimization, nested-cv, scikit-learn]
use_cases:
  - "tune a random forest or gradient boosting model without overfitting the validation set"
  - "search learning rate and regularisation for a neural network on a fixed GPU budget"
  - "decide between grid search, random search and Optuna for a project"
  - "report an honest score for a tuned model with nested cross-validation"
status: draft
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/grid_search.html
  - https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
  - https://optuna.readthedocs.io/
  - https://www.jmlr.org/papers/v13/bergstra12a.html
---

# Hyperparameter tuning (grid, random, successive halving, Optuna)

## Summary
Hyperparameters are the settings chosen before training (tree depth, learning rate, regularisation strength). Tuning searches them by training many models and scoring each on validation data or cross-validation. Random search beats grid search for the same budget, successive halving cuts cost by dropping bad candidates early, and Bayesian tools like Optuna focus the budget on promising regions. The usual gains are modest compared with better data or features, and the main risk is overfitting the validation set.

## Key concepts
- **Search space.** Which parameters and ranges. Use log scales for things that span orders of magnitude (learning rate, `C`, `alpha`): `loguniform(1e-4, 1e1)`.
- **Grid search** (`GridSearchCV`): every combination. Cost multiplies with each parameter; fine for 1-2 parameters.
- **Random search** (`RandomizedSearchCV`): sample `n_iter` combinations from distributions. Usually only a few parameters matter, and random search tries more distinct values of each (Bergstra and Bengio, 2012).
- **Successive halving** (`HalvingRandomSearchCV`, experimental import): start many candidates on a small budget (rows or epochs), keep the best fraction, give them more. Hyperband repeats this with several starting budgets.
- **Bayesian / model-based search** (Optuna's TPE, scikit-optimize): uses past trials to choose the next one; plus **pruning** that stops bad trials mid-training.
- **Cross-validated scoring.** Each candidate is scored by k-fold CV on the training data; the test set is never used.
- **Nested CV.** An outer CV loop around the whole tuning procedure gives an unbiased score of "this model family plus this tuning"; the inner `best_score_` is optimistically biased.
- **Early stopping** is itself a cheap form of tuning the number of trees/epochs, see [[gradient-boosting-tabular]] and [[deep-learning-training]].

## When to use / scenarios
- After the pipeline, features and metric are settled and a default model has a baseline score; tuning is the last few percent.
- Gradient boosting (learning rate, depth, leaves, regularisation) and SVMs/regularised linear models (`C`, `alpha`) benefit the most; random forests the least.
- Deep learning: learning rate first (by far the most important), then weight decay and batch size; use a small budget per trial and pruning.
- Not when: data is tiny (you will tune to noise; prefer strong regularisation and defaults), or the gap is in data quality or leakage (tuning cannot fix those).

## Setup & code
```bash
pip install scikit-learn
# optional, for Bayesian search with pruning:
pip install optuna
```
```python
from scipy.stats import loguniform, randint
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import (HalvingRandomSearchCV, RandomizedSearchCV,
                                     cross_val_score, train_test_split)

X, y = make_classification(n_samples=4000, n_features=30, n_informative=8,
                           random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)

space = {
    "learning_rate": loguniform(0.01, 0.3),
    "max_leaf_nodes": randint(8, 64),
    "min_samples_leaf": randint(5, 100),
    "l2_regularization": loguniform(1e-3, 10),
}
base = HistGradientBoostingClassifier(random_state=0)
print("default CV:", round(cross_val_score(base, X_tr, y_tr, cv=5, scoring="roc_auc").mean(), 4))

rand = RandomizedSearchCV(base, space, n_iter=30, cv=5, scoring="roc_auc",
                          n_jobs=-1, random_state=0).fit(X_tr, y_tr)
print("random search CV:", round(rand.best_score_, 4), rand.best_params_)

# min_resources: the first round must be big enough for every CV fold to hold
# both classes, or those candidates score NaN.
halving = HalvingRandomSearchCV(base, space, n_candidates=60, factor=3, cv=5,
                                min_resources=300, scoring="roc_auc",
                                n_jobs=-1, random_state=0)
halving.fit(X_tr, y_tr)
print("halving CV:", round(halving.best_score_, 4))

# Report once on the untouched test set.
print("test ROC-AUC:", round(rand.score(X_te, y_te), 4))

# Nested CV: honest estimate of the whole tuning procedure.
inner = RandomizedSearchCV(base, space, n_iter=10, cv=3, scoring="roc_auc",
                           random_state=0)
print("nested CV:", round(cross_val_score(inner, X_tr, y_tr, cv=3, scoring="roc_auc").mean(), 4))
```
On this data every search lands within about 0.002 ROC-AUC of the defaults (0.985), which is common for `HistGradientBoosting`. Tuning matters most when the defaults are clearly off for the data.

Optuna version of the same search (define-by-run, supports pruning and dashboards):
```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    model = HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 64),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 100),
        random_state=0)
    return cross_val_score(model, X_tr, y_tr, cv=5, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print(study.best_value, study.best_params)
```

## Choosing / trade-offs
- **Grid** for 1-2 parameters with a few known values, or a final fine sweep.
- **Random** as the default for 3+ parameters: simple, parallel, no extra dependency.
- **Successive halving / Hyperband** when each fit is expensive and partial budgets (fewer rows, fewer epochs) rank candidates reliably.
- **Optuna (TPE + pruning)** when trials are expensive (deep learning, large boosting), you want conditional spaces (parameters that exist only for some choices), or a persistent study you can resume and inspect.
- **Budget.** 20-60 random trials typically capture most of the gain for boosting; log every trial with [[experiment-tracking]].

## Gotchas
- Reporting `best_score_` as the model's performance is optimistic; use a held-out test set or nested CV.
- Preprocessing outside the search (scaling, imputation, target encoding fit on all data) leaks into every fold; tune a `Pipeline` and address steps as `step__param`.
- Huge searches on small data overfit the CV folds; differences in the third decimal are usually noise. Check the spread (`cv_results_["std_test_score"]`).
- Uniform sampling of a learning rate between 0.0001 and 1 spends almost all trials above 0.1; use log-uniform.
- Best parameters at the edge of the range mean the range is wrong; widen it and search again.
- `n_jobs=-1` in both the search and the model oversubscribes CPUs; parallelise at one level.
- Time-series and grouped data need the matching CV splitter (`TimeSeriesSplit`, `GroupKFold`) in the search, or the tuned settings are tuned to leakage.

## Related
- [[model-evaluation-and-metrics]] - choosing the scoring metric and CV scheme the search optimises.
- [[gradient-boosting-tabular]] - the models that gain most from tuning.
- [[decision-trees-and-random-forests]] - depth and leaf-size parameters to search.
- [[deep-learning-training]] - learning rate, schedules and early stopping.
- [[experiment-tracking]] - recording every trial and its parameters.

## References
- scikit-learn tuning guide (grid, random, successive halving): https://scikit-learn.org/stable/modules/grid_search.html
- Nested vs non-nested CV example: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- Optuna documentation: https://optuna.readthedocs.io/
- Bergstra and Bengio, *Random Search for Hyper-Parameter Optimization* (JMLR 2012): https://www.jmlr.org/papers/v13/bergstra12a.html
