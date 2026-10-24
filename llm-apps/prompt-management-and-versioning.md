---
title: Prompt management and versioning
category: llm-apps
tags: [prompts, prompt-management, versioning, prompt-registry, labels, rollback, langfuse, mlflow, llmops, eval-gates]
use_cases:
  - "let product people change a prompt without a code deploy, but with review and rollback"
  - "know exactly which prompt version produced a given answer in production"
  - "only promote a new prompt version to production after it passes our eval set"
  - "run two prompt versions side by side for an A/B test"
status: draft
last_verified: 2026-10-24
sources:
  - https://raw.githubusercontent.com/langfuse/langfuse-docs/main/content/docs/prompt-management/data-model.mdx
  - https://raw.githubusercontent.com/langfuse/langfuse-docs/main/content/docs/prompt-management/features/caching.mdx
  - https://raw.githubusercontent.com/langfuse/langfuse-docs/main/content/docs/prompt-management/features/guaranteed-availability.mdx
  - https://raw.githubusercontent.com/langfuse/langfuse-docs/main/content/docs/prompt-management/features/prompt-version-control.mdx
  - https://raw.githubusercontent.com/langfuse/langfuse-docs/main/content/docs/prompt-management/features/config.mdx
  - https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/prompt-registry/index.mdx
  - https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/prompt-registry/manage-prompt-lifecycles-with-aliases.mdx
---

# Prompt management and versioning

## Summary
In an LLM app the prompt is behaviour-defining code that changes more often than the code around it. Prompt management treats each prompt as a versioned artifact: every edit creates an immutable version, a movable label (`production`, `staging`) says which version each environment serves, every request logs the version it used, and a label only moves after the new version passes the eval set. You can do this in Git (prompts as files, deployed with the app) or with a prompt registry (Langfuse, MLflow Prompt Registry and similar) that the app reads at run time with a local cache, so prompt changes ship and roll back without a code deploy.

## Key concepts
- **Immutable versions**: an edit never changes an existing version; it creates version n+1. That makes "which prompt produced this output" answerable forever. Langfuse numbers versions 1, 2, 3; MLflow describes prompt versions as immutable, with commit messages like Git.
- **Labels / aliases**: mutable pointers to a version. Code asks for `support-triage@production`, not for version 7; deploying or rolling back is moving the pointer. In Langfuse, `production` is the default label served when none is given and `latest` always points at the newest version; MLflow uses aliases loaded as `prompts:/<name>@<alias>`.
- **Template variables**: both Langfuse and MLflow use `{{variable}}` placeholders, filled by `prompt.compile(...)` (Langfuse) or `prompt.format(...)` (MLflow). Fail loudly on a missing variable instead of sending a literal `{{customer_name}}` to the model.
- **Prompt plus config**: model ID, temperature, max tokens, tools and the output schema change behaviour as much as the text does; version them together (Langfuse prompts carry a `config` object for this) ([[structured-output]]).
- **Runtime fetch with cache**: Langfuse SDKs cache prompts client-side (default TTL 60 seconds) and serve the stale copy while refreshing in the background; if the cache is empty and the service is down, `get_prompt` raises unless you pass a `fallback` prompt or pre-fetch at startup.
- **Trace linkage**: attach prompt name and version to every LLM span, so cost, latency, errors and user feedback can be grouped by version ([[llm-observability]]).
- **Eval gate**: a version is promoted only when it scores at least as well as the current `production` version on a fixed eval set ([[llm-as-judge-and-eval-driven-development]]).

## When to use / scenarios
- Non-engineers (support leads, domain experts) own the wording of prompts: a registry with review and labels lets them ship changes without a release, while engineers keep the eval gate.
- Several environments or tenants need different versions at once: labels per environment (`staging`, `production`) or per tenant.
- A/B tests of prompt variants: two labels, traffic split at the gateway or in the app, metrics compared by version ([[ab-testing-and-canary-for-models]]).
- Regulated or audited apps that must show which instructions were in force when an answer was given ([[ai-regulation-checklist]]).
- Not needed for one prompt owned by the engineers who deploy the app: keep it in the repo next to its tests and eval set; Git already gives versions, review and rollback.
- Not a replacement for evals: versioning makes changes traceable and reversible, not correct ([[llm-evaluation]]).

## Setup & code
Langfuse (`pip install langfuse`; keys from environment variables). Create a version with a label, then fetch by label with a fallback for availability:
<!-- skip-check: needs a Langfuse server and API keys -->
```python
from langfuse import Langfuse

langfuse = Langfuse()                                   # reads LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST
langfuse.create_prompt(
    name="support-triage",
    type="text",
    prompt="Classify this ticket for {{product}} into one of {{labels}}.\nTicket: {{ticket}}",
    config={"model": "<model-id>", "temperature": 0},
    labels=["staging"],                                 # promote to production after the eval passes
)
prompt = langfuse.get_prompt("support-triage",          # no label -> the "production" version
                             cache_ttl_seconds=300,
                             fallback="Classify this ticket into one of {{labels}}.\nTicket: {{ticket}}")
text = prompt.compile(product="Acme Cloud", labels="billing, bug, account", ticket="Charged twice")
print(prompt.version, prompt.is_fallback, prompt.config, text)
```
MLflow Prompt Registry: register versions, point an alias at one, load by alias:
<!-- skip-check: needs an MLflow tracking server -->
```python
import mlflow

p = mlflow.genai.register_prompt(
    name="support-triage",
    template="Classify this ticket for {{ product }} into one of {{ labels }}.\nTicket: {{ ticket }}",
    commit_message="Add product name to the instructions",
)
mlflow.genai.set_prompt_alias("support-triage", alias="production", version=p.version)
prod = mlflow.genai.load_prompt("prompts:/support-triage@production")
print(prod.format(product="Acme Cloud", labels="billing, bug, account", ticket="Charged twice"))
```
Runnable, stdlib only: a minimal registry showing the moving parts (content-addressed immutable versions, labels, strict `{{var}}` rendering, a TTL cache with last-known-good fallback, and an eval gate that moves `production` only when the candidate is no worse).
```python
import hashlib
import json
import re
import time

class PromptRegistry:
    def __init__(self):
        self.versions = {}     # name -> list of {"version", "template", "config", "sha"}
        self.labels = {}       # (name, label) -> version number

    def create(self, name, template, config, labels=()):
        sha = hashlib.sha256(json.dumps([template, config], sort_keys=True).encode()).hexdigest()[:12]
        history = self.versions.setdefault(name, [])
        if history and history[-1]["sha"] == sha:
            return history[-1]["version"]                    # identical content: no new version
        version = len(history) + 1
        history.append({"version": version, "template": template, "config": config, "sha": sha})
        for label in ("latest", *labels):
            self.labels[(name, label)] = version
        return version

    def get(self, name, label="production", version=None):
        v = version or self.labels.get((name, label))
        if v is None:
            raise KeyError(f"{name}: no version has label {label!r}")   # never fall back silently
        return dict(self.versions[name][v - 1], name=name)

    def set_label(self, name, label, version):
        self.labels[(name, label)] = version

VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")

def render(prompt, **values):
    missing = set(VAR.findall(prompt["template"])) - set(values)
    if missing:
        raise ValueError(f"{prompt['name']} v{prompt['version']}: missing variables {sorted(missing)}")
    return VAR.sub(lambda m: str(values[m.group(1)]), prompt["template"])

class CachedClient:
    """Serve from cache within the TTL; on a failed refresh keep the last good copy, else the fallback."""
    def __init__(self, registry, ttl_s=60, fallback=None, clock=time.monotonic):
        self.registry, self.ttl, self.fallback, self.clock = registry, ttl_s, fallback or {}, clock
        self.cache, self.available = {}, True

    def get(self, name, label="production"):
        key, now = (name, label), self.clock()
        hit = self.cache.get(key)
        if hit and now - hit[0] < self.ttl:
            return hit[1]
        try:
            if not self.available:
                raise ConnectionError("registry unreachable")
            prompt = self.registry.get(name, label)
            self.cache[key] = (now, prompt)
            return prompt
        except ConnectionError:
            if hit:
                return hit[1]                                # stale but known good
            return dict(self.fallback[name], name=name, version="fallback")

reg = PromptRegistry()
cfg = {"model": "<model-id>", "temperature": 0}
v1 = reg.create("triage", "Label the ticket as one of {{labels}}.\nTicket: {{ticket}}", cfg, labels=["production"])
v2 = reg.create("triage", "You triage tickets. Answer with exactly one of {{labels}}.\nTicket: {{ticket}}", cfg)
print("production ->", reg.get("triage")["version"], "| latest ->", reg.get("triage", "latest")["version"])
print(render(reg.get("triage"), labels="billing|bug", ticket="Charged twice"))
try:
    render(reg.get("triage"), labels="billing|bug")
except ValueError as e:
    print("refused:", e)

# Eval gate: score candidate and current production on the same fixed set, promote only if no worse.
EVAL = [("Charged twice this month", "billing"), ("App crashes on login", "bug"),
        ("Refund still pending", "billing"), ("Button does nothing", "bug")]

def fake_model(prompt_text):
    """Stand-in for an LLM call: the v2 wording 'exactly one' makes it answer cleanly."""
    ticket = prompt_text.rsplit("Ticket: ", 1)[1].lower()
    label = "billing" if any(w in ticket for w in ("charged", "refund")) else "bug"
    return label if "exactly one" in prompt_text else f"I think this is {label}."

def score(prompt):
    return sum(fake_model(render(prompt, labels="billing|bug", ticket=t)).strip() == y for t, y in EVAL) / len(EVAL)

def promote_if_better(registry, name, candidate_version, min_gain=0.0):
    current, cand = registry.get(name), registry.get(name, version=candidate_version)
    s_cur, s_cand = score(current), score(cand)
    ok = s_cand >= s_cur + min_gain
    if ok:
        registry.set_label(name, "production", candidate_version)
    return ok, s_cur, s_cand

print("promote v2:", promote_if_better(reg, "triage", v2), "-> production is v", reg.get("triage")["version"])

# Runtime: cache with TTL, then an outage of the registry.
t = [0.0]
client = CachedClient(reg, ttl_s=60, clock=lambda: t[0],
                      fallback={"triage": {"template": "Label: {{labels}}\nTicket: {{ticket}}", "config": cfg}})
print("first fetch v", client.get("triage")["version"])
client.available = False; t[0] = 120
print("registry down, TTL expired -> still serving v", client.get("triage")["version"])
cold = CachedClient(reg, fallback=client.fallback)
cold.available = False
print("cold start during outage ->", cold.get("triage")["version"])
```

## Choosing / trade-offs
| Option | Pick when | Watch out for |
|---|---|---|
| Prompts as files in the app repo | engineers own prompts; changes ship with code anyway | every wording change needs a deploy; non-engineers need Git access |
| Open-source registry (Langfuse, MLflow) | non-engineers edit prompts; want labels, history and trace linkage | another service in the request path (mitigated by caching and fallbacks); self-hosting or a vendor contract |
| Registry inside an observability/eval platform | already using it for traces and evals | coupling prompts to that vendor; export format |
| Hybrid: registry for editing, export to repo at release | need both fast iteration and code-reviewed releases | two sources of truth unless one is clearly authoritative |

- **Runtime fetch vs build-time bake**: fetching lets a label move take effect within the cache TTL, but a bad promotion also reaches production that fast. Baking the version into the build makes rollbacks a redeploy. Many teams fetch at runtime for prompts with an eval gate and bake the rest.
- **Granularity**: one prompt per task step (classify, extract, answer) keeps evals targeted; one giant system prompt makes every change risky.
- **Who can move `production`**: restrict label changes to a reviewed path (CI job after the eval passes), even if anyone can create versions.

## Gotchas
- Changing the model ID under an unchanged prompt is a behaviour change; version the model and parameters with the prompt, and re-run evals when either changes ([[model-selection]]).
- `latest` in production means every saved draft goes live; serve a named label instead.
- Missing label or version: Langfuse documents that it returns 404 rather than silently using another label; make the app fail at startup (pre-fetch) instead of on the first user request.
- Prompt text is part of the prompt-cache prefix; a small edit near the top invalidates cached prefixes and raises cost and latency until the cache warms again ([[prompt-caching-and-cost]]).
- Registry-edited prompts bypass code review; they can also be a path for prompt injection or leaking instructions if edit rights are broad. Treat edit access like deploy access ([[prompt-injection]]).
- Variables filled with user input are untrusted; delimit them and never let them change the template itself ([[guardrails-and-safety]]).
- Without the version on each trace, a quality drop cannot be tied to a prompt change; log it from day one ([[opentelemetry-genai-tracing]]).

## Related
- [[prompt-engineering]] - writing the prompts this topic versions.
- [[llm-as-judge-and-eval-driven-development]] - the eval gate that decides promotions.
- [[llm-evaluation]] - building the fixed eval set.
- [[llm-observability]] - tracing outputs back to a prompt version.
- [[ab-testing-and-canary-for-models]] - comparing two versions on live traffic.
- [[dspy-and-prompt-optimization]] - generating prompt versions automatically.
- [[context-engineering]] - what goes into the prompt around the template.
- [[model-registry-and-versioning]] - the same versioning ideas for model weights.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/feature-flags.md - labels as runtime flags, rollout and kill switches.

## References
- Langfuse prompt data model (versions, `production` and `latest` labels, config): https://raw.githubusercontent.com/langfuse/langfuse-docs/main/content/docs/prompt-management/data-model.mdx
- Langfuse prompt caching (client-side cache, 60 s default TTL, background refresh): https://raw.githubusercontent.com/langfuse/langfuse-docs/main/content/docs/prompt-management/features/caching.mdx
- Langfuse guaranteed availability (pre-fetch, `fallback`, `is_fallback`): https://raw.githubusercontent.com/langfuse/langfuse-docs/main/content/docs/prompt-management/features/guaranteed-availability.mdx
- Langfuse version control and labels (`create_prompt`, `update_prompt`, label resolution): https://raw.githubusercontent.com/langfuse/langfuse-docs/main/content/docs/prompt-management/features/prompt-version-control.mdx
- Langfuse prompt config (`config` on `create_prompt`, `prompt.config`): https://raw.githubusercontent.com/langfuse/langfuse-docs/main/content/docs/prompt-management/features/config.mdx
- MLflow Prompt Registry (`register_prompt`, `load_prompt`, `format`, immutable versions): https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/prompt-registry/index.mdx
- MLflow prompt aliases (`set_prompt_alias`, `prompts:/<name>@<alias>`): https://raw.githubusercontent.com/mlflow/mlflow/master/docs/docs/genai/prompt-registry/manage-prompt-lifecycles-with-aliases.mdx
