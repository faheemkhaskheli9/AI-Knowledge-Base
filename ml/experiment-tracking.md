---
title: Experiment tracking (MLflow, Weights & Biases)
category: ml
tags: [mlflow, wandb, experiment-tracking, model-registry, reproducibility, mlops]
use_cases:
  - "track hyperparameters, metrics and artifacts across many training runs"
  - "compare model runs and pick the best for deployment"
  - "make a training result reproducible months later"
  - "keep a model registry with staged versions for production"
  - "self-host experiment tracking because data cannot leave the company"
status: draft
last_verified: 2026-10-03
sources:
  - https://mlflow.org/docs/latest/ml/tracking/
  - https://docs.wandb.ai/
---

# Experiment tracking (MLflow, Weights & Biases)

## Summary
Experiment tracking records, for every training run, the code version, parameters, metrics, data version and output artifacts so runs can be compared and reproduced. MLflow is open source and self-hostable (tracking, model registry, packaging); Weights & Biases (W&B) is a hosted-first service with rich dashboards, sweeps and collaboration. Either beats a folder of notebooks and "final_v3.csv".

## Key concepts
- Run: one execution; holds params (inputs), metrics (outputs, optionally per step), artifacts (files, models), tags.
- Experiment / project: a group of comparable runs.
- Model registry: named model with versions and aliases/stages (e.g. `champion`) for promotion to serving.
- Autologging: framework hooks (`mlflow.sklearn.autolog()`, `mlflow.pytorch.autolog()`) capture params and metrics automatically.
- Reproducibility inputs: git commit, dependency lockfile, random seed, dataset hash/version, hardware.
- Sweeps / HPO: W&B Sweeps, or Optuna with a tracking callback.
- LLM-era use: both products also offer tracing and evaluation for LLM apps; see [[llm-observability]].

## When to use / scenarios
- Any project with more than a handful of training runs or more than one person.
- Regulated settings (finance, healthcare) needing an audit trail of which model, data and code produced a decision.
- Teams needing a promotion workflow from experiment to production ([[mlops-lifecycle]]).
- Self-hosted/air-gapped: MLflow server with a database and object store.
- NOT needed: one-off analysis; a CSV of results is fine there.

## Setup & code
```bash
pip install mlflow scikit-learn
mlflow ui            # local UI at http://127.0.0.1:5000 (check the port is free)
```
```python
import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X, y = load_iris(return_X_y=True)
mlflow.set_experiment("iris-baseline")
with mlflow.start_run():
    C = 0.5
    model = LogisticRegression(C=C, max_iter=500)
    score = cross_val_score(model, X, y, cv=5).mean()
    mlflow.log_param("C", C)
    mlflow.log_metric("cv_accuracy", score)
    mlflow.sklearn.log_model(model.fit(X, y), name="model")
```
W&B equivalent: `pip install wandb`, `wandb.login()`, then `run = wandb.init(project="p", config={...})`, `run.log({"loss": l})`, `run.finish()`.

## Choosing / trade-offs
- MLflow: open source, self-host friendly, strong registry and model packaging; UI is plainer; you operate the server.
- W&B: best visual comparison, sweeps, reports, team features; hosted by default (data leaves your network unless you self-host the enterprise option); pricing tiers apply.
- Lightweight alternatives: TensorBoard (curves only), DVC (data/pipeline versioning), Aim, Comet, Neptune.
- Pick by data governance first, UX second. Do not run both without a reason.

## Gotchas
- Logging per-step metrics every iteration inflates storage and slows runs; log every N steps.
- Untracked data changes: log a dataset hash or version, not just the file name.
- Local default stores (`mlruns/` directory) are per-machine; use a shared backend store for teams.
- Never log secrets or raw PII as params/artifacts.
- Unpinned dependencies make a logged run unreproducible; log `pip freeze` or a lockfile.
- API names move between major versions (e.g. model logging arguments); check the current docs.
- Forgetting `wandb.finish()` / ending the run in notebooks merges runs.

## Related
- [[mlops-lifecycle]] - registry and deployment pipeline.
- [[llm-observability]] - tracing for LLM apps.
- [[classic-ml-scikit-learn]] - CV runs worth tracking.
- [[pytorch-basics]] - training loops to instrument.

## References
- https://mlflow.org/docs/latest/ml/tracking/
- https://docs.wandb.ai/
