---
title: MLOps and LLMOps lifecycle
category: deployment
tags: [mlops, llmops, ci-cd, model-registry, monitoring, drift, versioning, reproducibility]
use_cases:
  - "take a notebook model or prompt chain to a reliable production service"
  - "set up versioning, CI and rollback for models and prompts"
  - "detect data drift or quality decay after launch"
  - "decide what to automate first for a small team shipping an AI feature"
status: draft
last_verified: 2026-10-03
sources:
  - https://mlflow.org/docs/latest/
  - https://dvc.org/doc
  - https://docs.astral.sh/uv/
---

# MLOps and LLMOps lifecycle

## Summary
MLOps is the set of practices that make models reproducible, deployable, monitored and improvable: data and code versioning, experiment tracking, evaluation gates, CI/CD, serving, monitoring and retraining. LLMOps adapts it to prompts, retrieval indexes, model-vendor changes and non-deterministic outputs, where the "model" is often a configuration of prompt + model + tools.

## Key concepts
- Artifacts to version: code, data/index snapshots, model weights or vendor model id, prompts, configs, eval sets.
- Reproducibility: pinned dependencies (`uv.lock`), seeds, recorded dataset hashes, container images.
- Evaluation gate: a fixed eval set and metrics that must pass before promotion ([[llm-evaluation]]).
- Registry and staging: candidate -> staging -> production, with rollback to the previous version.
- Monitoring: system (latency, errors, GPU), quality (user feedback, judge scores), data/concept drift, cost.
- Feedback loop: production traces become new eval and training data ([[llm-observability]]).

## When to use / scenarios
- Tabular/CV model retrained monthly: pipeline with data validation, tracking, scheduled retraining ([[experiment-tracking]]).
- LLM feature with frequent prompt edits: prompts in git, eval on every change, canary rollout.
- Vendor deprecates a model id: swap and re-run evals before switching traffic.
- Overkill for a one-off analysis or hackathon demo; add pieces as risk grows.

## Setup & code
A minimal, useful baseline (small team):
1. Git for code and prompts; lockfile for deps.
```bash
uv init && uv add mlflow pytest && git init
```
2. Track experiments/runs (MLflow shown; see its docs for the tracking server):
```python
import mlflow
with mlflow.start_run():
    mlflow.log_params({"model": "<id>", "prompt_version": "v3"})
    mlflow.log_metric("eval_accuracy", 0.87)
```
3. Data versioning with DVC or object-store snapshots with content hashes.
4. CI: on each pull request run unit tests plus the eval suite; fail on regression.
```yaml
# .github/workflows/eval.yml (sketch)
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --frozen
      - run: uv run pytest tests/ evals/
        env: { LLM_API_KEY: "${{ secrets.LLM_API_KEY }}" }
```
(Verify action versions against their repos.)
5. Deploy as a container; roll out gradually (canary/shadow), keep the previous image for rollback.
6. Monitor and alert on error rate, latency, cost per request, and a quality proxy.

## Choosing / trade-offs
- Managed platforms (cloud ML suites) vs open-source stack (MLflow, DVC, Prefect/Airflow, Langfuse): convenience vs portability and cost.
- Full CI evals with a live LLM cost money and are noisy; use a small gold set per PR and a larger nightly run.
- Retrain on schedule vs on drift trigger: schedule is simple; triggers need reliable metrics.
- Shadow/canary release adds infrastructure but lowers blast radius.

## Gotchas
- Silent vendor model updates change behavior; pin dated model ids where offered and rerun evals periodically.
- Eval set leakage: tuning prompts on the test set inflates scores; hold out data.
- Non-determinism makes single-run comparisons unreliable; repeat runs or use temperature 0 and tolerance bands.
- RAG index and embedding model are versioned artifacts too; changing the embedder requires re-indexing.
- Secrets in CI logs and notebooks ([[ai-security-privacy-compliance]]).
- Notebook-only workflows resist testing; move logic into modules early.

## Related
- [[experiment-tracking]] - runs and metrics.
- [[llm-evaluation]] - gates and metrics.
- [[llm-observability]] - production traces.
- [[python-env-uv]] - reproducible environments.
- [[inference-servers-vllm]], [[serving-with-fastapi]] - serving layers.
- [[data-labeling-and-synthetic-data]] - building eval/training sets.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/ci-cd-pipelines.md - build and test pipelines.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/deployment-strategies.md - canary, blue-green and shadow rollouts for a new model.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/infrastructure-as-code.md - reproducible infrastructure.

## References
- MLflow: https://mlflow.org/docs/latest/
- DVC: https://dvc.org/doc
