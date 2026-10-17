---
title: Model and prompt registry, versioning and rollback
category: deployment
tags: [model-registry, prompt-registry, versioning, lineage, aliases, promotion, rollback, mlflow, llmops]
use_cases:
  - "register every trained model and prompt version so we know exactly what is in production"
  - "promote a candidate model from staging to production only after it passes the eval gate"
  - "roll back to the previous model or prompt in one step when quality drops"
  - "trace a production answer back to the model, prompt, data snapshot and eval run that produced it"
  - "manage prompt versions separately from application deploys"
status: draft
last_verified: 2026-10-15
sources:
  - https://mlflow.org/docs/latest/ml/model-registry/
  - https://mlflow.org/docs/latest/ml/model-registry/workflow/
  - https://mlflow.org/docs/latest/genai/prompt-registry/
  - https://mlflow.org/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/
---

# Model and prompt registry, versioning and rollback

## Summary
A registry is the system of record for deployable AI artifacts: each trained model or prompt gets an immutable numbered version, metadata linking it to the code, data and eval run that produced it, and movable pointers ("aliases" such as `champion` or `production`) that serving code resolves at load time. Promotion and rollback then become pointer moves instead of redeploys, and any production output can be traced back to its inputs.

## Key concepts
- **Registered model / prompt**: a named entry (e.g. `ticket-router`) holding many versions.
- **Version**: immutable. Never overwrite version 3; fix a bug by creating version 4. Store a content hash (weights file, prompt text) so "same version" is checkable.
- **Alias**: a mutable named pointer to one version (`champion`, `challenger`, `production`). Serving loads `models:/ticket-router@champion`; moving the alias promotes or rolls back. MLflow deprecated fixed stages (Staging/Production/Archived) in 2.9.0 in favour of aliases and tags.
- **Lineage**: per version, record the training run id, git commit, dataset snapshot hash, base model or vendor model ID, hyperparameters, and the eval run and scores that justified promotion ([[experiment-tracking]]).
- **Promotion gate**: a version gets the production alias only if it passes the eval suite and any human sign-off ([[llm-evaluation]]).
- **Rollback**: keep the previous production version addressable (e.g. alias `previous`) and the serving stack able to load it; a rollback must not need a rebuild.
- **For LLM apps the "model" is a bundle**: vendor model ID + prompt version + tool definitions + retrieval index version + decoding settings. Version the bundle, not just one part.
- **Prompt registry**: prompts versioned and aliased like models (MLflow `mlflow.genai.register_prompt`, `load_prompt("prompts:/name@production")`), so prompt changes ship without an app deploy but still go through evals.

## When to use / scenarios
- More than one model or prompt version has ever been in production, or more than one person ships them.
- Regulated settings (credit, hiring, healthcare) that need to show which model made a decision and why it was approved ([[ai-regulation-checklist]], [[ai-security-privacy-compliance]]).
- Frequent prompt edits by non-engineers: a prompt registry with aliases plus an eval gate.
- Vendor model migration: register the new vendor model ID as a new bundle version, run evals, then move the alias.
- Canary/A-B: `champion` and `challenger` aliases routed by the gateway ([[llm-gateways-and-routing]]).
- Not needed for a single notebook experiment; git tags plus an experiment tracker are enough until something serves traffic.

## Setup & code
The core mechanics with the standard library: immutable versions with hashes and lineage, aliases, a promotion gate, and rollback.

```python
import hashlib, json, time

class Registry:
    def __init__(self):
        self.versions = {}   # name -> list of version dicts (index = version - 1)
        self.aliases = {}    # (name, alias) -> version number
        self.audit = []      # append-only log of alias moves

    def register(self, name, artifact: bytes, lineage: dict):
        vs = self.versions.setdefault(name, [])
        v = {"version": len(vs) + 1, "sha256": hashlib.sha256(artifact).hexdigest(),
             "lineage": lineage, "created": time.time()}
        vs.append(v)
        return v["version"]

    def get(self, name, ref):
        """ref is a version number or an alias name."""
        num = ref if isinstance(ref, int) else self.aliases[(name, ref)]
        return self.versions[name][num - 1]

    def set_alias(self, name, alias, version, who, reason):
        old = self.aliases.get((name, alias))
        self.aliases[(name, alias)] = version
        self.audit.append({"name": name, "alias": alias, "from": old, "to": version,
                           "who": who, "reason": reason})

    def promote(self, name, version, eval_scores, thresholds, who):
        failed = {k: s for k, s in eval_scores.items() if s < thresholds[k]}
        if failed:
            raise ValueError(f"gate failed: {failed}")
        cur = self.aliases.get((name, "production"))
        if cur is not None:
            self.set_alias(name, "previous", cur, who, "kept for rollback")
        self.versions[name][version - 1]["lineage"]["eval"] = eval_scores
        self.set_alias(name, "production", version, who, "passed gate")

    def rollback(self, name, who, reason):
        prev = self.aliases[(name, "previous")]
        self.set_alias(name, "production", prev, who, f"rollback: {reason}")

reg = Registry()
bundle_v1 = json.dumps({"model": "claude-haiku-4-5-20251001", "prompt": "Route the ticket...", "max_tokens": 300}).encode()
bundle_v2 = json.dumps({"model": "claude-sonnet-5-5", "prompt": "Route the ticket...", "max_tokens": 300}).encode()
lineage = {"git": "a1b2c3d", "dataset_sha256": "e3b0c442", "eval_run": "run-17"}
v1 = reg.register("ticket-router", bundle_v1, dict(lineage))
v2 = reg.register("ticket-router", bundle_v2, dict(lineage, eval_run="run-18"))

gate = {"accuracy": 0.90, "json_valid": 0.99}
reg.promote("ticket-router", v1, {"accuracy": 0.91, "json_valid": 1.0}, gate, who="alice")
try:
    reg.promote("ticket-router", v2, {"accuracy": 0.88, "json_valid": 1.0}, gate, who="bob")
except ValueError as e:
    print(e)                                   # gate blocks the regression
reg.promote("ticket-router", v2, {"accuracy": 0.94, "json_valid": 1.0}, gate, who="bob")
assert reg.get("ticket-router", "production")["version"] == 2
reg.rollback("ticket-router", who="oncall", reason="p95 latency doubled")
assert reg.get("ticket-router", "production")["version"] == 1
for row in reg.audit:
    print(row)
```

The same workflow on MLflow (needs `pip install mlflow` and a tracking server or local `mlruns/`; API names per the MLflow model and prompt registry docs):

<!-- skip-check: needs mlflow installed and a tracking backend -->
```python
import mlflow
from mlflow import MlflowClient

client = MlflowClient()
# register a logged model as a new version of "ticket-router"
mv = mlflow.register_model("runs:/<run_id>/model", "ticket-router")
client.set_model_version_tag("ticket-router", mv.version, "dataset_sha256", "e3b0c442")
client.set_registered_model_alias("ticket-router", "champion", mv.version)    # promote
model = mlflow.pyfunc.load_model("models:/ticket-router@champion")           # serving resolves the alias
# rollback = point the alias back at the previous version
client.set_registered_model_alias("ticket-router", "champion", int(mv.version) - 1)

# prompts get the same treatment
p = mlflow.genai.register_prompt(name="ticket-router-prompt",
                                 template="Route this ticket: {{ticket}}",
                                 commit_message="add billing category")
mlflow.genai.set_prompt_alias("ticket-router-prompt", alias="production", version=p.version)
prompt = mlflow.genai.load_prompt("prompts:/ticket-router-prompt@production")
print(prompt.format(ticket="card declined twice"))
```

## Choosing / trade-offs
- **MLflow (self-hosted, open source)** vs **cloud registries** (SageMaker, Vertex AI, Azure ML; see [[managed-cloud-ai-platforms]]) vs **Hugging Face Hub repos with tags/revisions**: portability and cost vs integration with the cloud's deploy and IAM.
- **Prompts in git** (reviewed via pull request, shipped with the app) vs **prompt registry** (hot-swappable, editable by non-engineers): git is simpler and auditable; a registry decouples prompt releases from deploys but needs its own eval gate and access control.
- **Alias resolved at load time** (fast rollback, but a running process may cache the old version) vs **version pinned in the deploy config** (fully reproducible deploys, rollback = redeploy).
- **Weights in the registry** vs **pointer to object storage**: large checkpoints usually live in object storage; the registry keeps the URI and hash ([[model-checkpointing-and-export]]).

## Gotchas
- An alias move is a production change: restrict who can set `production`, and log who moved it and why.
- Caches: MLflow caches alias-resolved prompts (60 s TTL by default per its docs), and long-running servers may load a model once at startup; a rollback is not effective until they re-resolve. Test rollback time, not just rollback.
- Registering only the model and not the prompt, tool schema, retrieval index or decoding settings makes rollback incomplete.
- Vendor model aliases (dateless IDs that move) break reproducibility; register the pinned snapshot ID ([[anthropic-claude]], [[openai-gpt]]).
- Changing the embedding model invalidates the vector index; version them together ([[vector-databases]]).
- Deleting old versions breaks lineage for past decisions; archive instead, and keep them for as long as your audit or legal retention requires.
- Eval scores attached to a version are only comparable if the eval set version is recorded too.

## Related
- [[mlops-lifecycle]] - where the registry sits in the wider pipeline.
- [[experiment-tracking]] - runs that produce registered versions.
- [[llm-evaluation]] - the promotion gate.
- [[llm-observability]] - log the bundle version on every trace so incidents map to a version.
- [[llm-gateways-and-routing]] - champion/challenger traffic splits.
- [[model-checkpointing-and-export]] - the artifacts being registered.
- [[ai-regulation-checklist]] - record-keeping duties a registry helps meet.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/deployment-strategies.md - canary and blue-green rollouts around an alias switch.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/feature-flags.md - flag-style switches between model versions.

## References
- MLflow Model Registry: https://mlflow.org/docs/latest/ml/model-registry/
- MLflow registry workflows (aliases, stage deprecation): https://mlflow.org/docs/latest/ml/model-registry/workflow/
- MLflow Prompt Registry: https://mlflow.org/docs/latest/genai/prompt-registry/
- MLflow prompt aliases and lifecycle: https://mlflow.org/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/
