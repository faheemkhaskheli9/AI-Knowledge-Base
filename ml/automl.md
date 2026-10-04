---
title: AutoML (automated model selection and tuning)
category: ml
tags: [automl, autogluon, flaml, model-selection, stacking, baseline, tabular, scikit-learn]
use_cases:
  - "get a strong tabular baseline in an hour without hand-tuning models"
  - "compare many model families on a new dataset before committing to one"
  - "let non-ML engineers train a decent classifier or regressor"
  - "find a cheap, good-enough model under a fixed compute budget"
status: draft
last_verified: 2026-10-04
sources:
  - https://auto.gluon.ai/stable/index.html
  - https://microsoft.github.io/FLAML/
  - https://scikit-learn.org/stable/modules/grid_search.html
---

# AutoML (automated model selection and tuning)

## Summary
AutoML searches over preprocessing, model families and hyperparameters (and
often ensembles the best results) under a time budget, returning a trained
model with no manual tuning. On tabular data it gives a strong baseline in
minutes and is often hard to beat by hand. It does not replace problem
framing, leakage-free validation or feature engineering from domain knowledge.

## Key concepts
- Search space: model families (GBMs, random forests, linear, kNN, neural
  nets) x hyperparameters x preprocessing.
- Search strategy: random search, Bayesian optimisation, cost-aware search
  (FLAML starts cheap and grows), or fixed portfolios of known-good configs
  (AutoGluon).
- Ensembling and stacking: AutoGluon's top results come from multi-layer
  stack ensembles with bagging, not from one tuned model.
- Time budget is the main knob: more time = more models and bigger ensembles.
- Validation: AutoML optimises whatever split you give it; a wrong split
  (random split on time-series or grouped data) makes it optimise leakage.
- Presets trade accuracy against training time, model size and inference latency.

## When to use / scenarios
- First pass on any new tabular problem (churn, credit risk, demand, lead
  scoring) to learn the achievable ceiling quickly.
- Teams without ML specialists that need a reliable model.
- Benchmarking: is a custom model actually better than an automated search?
- Hard compute or latency budget: FLAML can search for the cheapest good model.
- NOT for: images/text/audio from scratch (use pretrained models and
  [[fine-tuning-and-peft]] / [[image-classification]]), tiny datasets where a
  simple [[linear-models]] model is enough, or when every model choice must be
  hand-justified to a regulator (use an interpretable model directly).

## Setup & code
```bash
pip install flaml[automl] scikit-learn   # light-weight, sklearn-style
pip install autogluon.tabular            # heavier, strongest ensembles
```
FLAML (cost-aware search, returns a single estimator):
```python
from flaml import AutoML
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)

automl = AutoML()
automl.fit(X_tr, y_tr, task="classification", metric="roc_auc", time_budget=60, seed=0)
print(automl.best_estimator, automl.best_config)
print("test AUC:", roc_auc_score(y_te, automl.predict_proba(X_te)[:, 1]))
```
AutoGluon (stack ensembles from a pandas DataFrame with a label column):
```python
from autogluon.tabular import TabularPredictor

train = X_tr.assign(target=y_tr)
predictor = TabularPredictor(label="target", eval_metric="roc_auc").fit(
    train, presets="medium_quality", time_limit=300)
print(predictor.leaderboard(X_te.assign(target=y_te)))
```
Pass your own validation scheme when rows are grouped or time-ordered
(FLAML: `split_type="time"` or `split_type="group"` with `groups=...`;
AutoGluon: `tuning_data=` with a hand-made holdout).

## Choosing / trade-offs
- Best accuracy, have minutes to hours and memory: AutoGluon with a
  high-quality preset; accept a large ensemble with slower inference.
- Small model, fast inference, sklearn-style object: FLAML, or AutoGluon
  with a size/latency-oriented preset or `refit_full`.
- Want full control over one model family: [[hyperparameter-tuning]] with
  Optuna on [[gradient-boosting-tabular]] is often enough.
- Small tabular data (thousands of rows): also try a tabular foundation
  model ([[deep-learning-for-tabular-data]]), which needs no search at all.
- Managed cloud AutoML (Vertex AI, SageMaker Autopilot, Azure AutoML):
  convenient inside that cloud, costs per training hour, less portable.

## Gotchas
- AutoML amplifies leakage: target-derived or post-event features get found
  and exploited fast. Audit the top features ([[model-interpretability]]).
- Random splits on time-series or per-customer data give over-optimistic
  scores; set the split type explicitly.
- Big stacked ensembles can be hundreds of MB and slow per request; check
  latency before shipping, not after.
- Results vary run to run with the budget and machine speed; fix seeds and
  record the time budget for reproducibility ([[experiment-tracking]]).
- The leaderboard score on the validation set used for selection is
  optimistic; keep a final untouched test set.
- Class imbalance and the right metric are still your job ([[imbalanced-data]]).

## Related
- [[hyperparameter-tuning]] - manual search over one model family.
- [[gradient-boosting-tabular]] - the models AutoML usually picks for tabular data.
- [[ensemble-methods]] - bagging and stacking that AutoGluon automates.
- [[deep-learning-for-tabular-data]] - TabPFN and tabular nets as alternatives.
- [[model-evaluation-and-metrics]] - picking the metric AutoML optimises.
- [[feature-engineering]] - domain features AutoML cannot invent.

## References
- AutoGluon docs: https://auto.gluon.ai/stable/index.html
- FLAML docs: https://microsoft.github.io/FLAML/
- Erickson et al., "AutoGluon-Tabular" (2020): https://arxiv.org/abs/2003.06505
