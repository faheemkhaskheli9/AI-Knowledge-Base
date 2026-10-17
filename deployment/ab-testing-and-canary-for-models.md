---
title: A/B testing, shadow traffic and canary rollouts for models and prompts
category: deployment
tags: [ab-testing, canary, shadow-traffic, progressive-delivery, rollback, online-evaluation, kserve, argo-rollouts, istio]
use_cases:
  - "roll out a new model version or prompt to 5% of traffic and roll back automatically if quality drops"
  - "test a new LLM against production traffic without showing its answers to users"
  - "run an A/B test between two prompts and know whether the difference is real"
  - "choose online metrics and guardrails for replacing a fraud or ranking model"
status: draft
last_verified: 2026-10-15
sources:
  - https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary
  - https://argo-rollouts.readthedocs.io/en/stable/features/canary/
  - https://argo-rollouts.readthedocs.io/en/stable/features/analysis/
  - https://istio.io/latest/docs/tasks/traffic-management/mirroring/
  - https://exp-platform.com/
---

# A/B testing, shadow traffic and canary rollouts for models and prompts

## Summary
Offline evals tell you a new model or prompt *should* be better; only production traffic tells you whether it *is*. Three techniques cover the path: **shadow** (the candidate gets a copy of real requests, its output is logged, never served), **canary** (a small, growing share of real users get the candidate while guardrail metrics are compared with the baseline, and traffic snaps back on a breach), and **A/B test** (a fixed split held long enough to measure the business metric with statistical confidence). The same machinery applies to classic ML models, LLM model swaps, prompt changes and RAG pipeline changes.

## Key concepts
- **What counts as a "version":** model weights or API model ID, prompt template, decoding settings, retrieval index, tool set. Version all of them together and tag every request and trace with the version ([[opentelemetry-genai-tracing]]).
- **Shadow (mirroring / dark launch):** zero user risk; measures latency, errors, cost and output *differences* (agreement with the current model, judge scores on disagreements). Cannot measure user reactions. Shadow calls cost money and must not have side effects (no emails sent, no DB writes, no tool calls that act).
- **Canary:** route 1% -> 5% -> 25% -> 50% -> 100% with pauses; at each step compare the canary with the baseline on guardrails (error rate, p95 latency, cost per request, refusal rate, safety-filter hits, quick quality proxies). Abort = all traffic back to the last good version.
- **A/B test:** randomised, sticky assignment (hash of user ID + experiment name), pre-registered primary metric (task success, conversion, resolution rate, fraud loss) and guardrails, sample size planned up front, read out with a confidence interval ([[statistics-and-ab-testing]]).
- **Interleaving** (for ranking/search/recsys): mix results from both rankers in one list and count which ranker's items get clicked; needs far less traffic than a split A/B test ([[learning-to-rank]]).
- **Bandits** shift traffic toward the winner while learning; good for many prompt variants with fast feedback, worse for clean causal readouts ([[multi-armed-bandits]]).
- **Online metrics for LLM features:** explicit feedback (thumbs, ratings) is sparse and biased; implicit signals (copy, retry/regenerate, edit distance of accepted text, escalation to a human, conversation abandoned) are denser; sampled LLM-judge scores on live traffic fill the gap ([[llm-evaluation]]).
- **Rollback must be fast and boring:** traffic switch, not a redeploy. Keep the previous model/prompt loaded and warm.

## When to use / scenarios
| Situation | Technique |
|---|---|
| New fraud or credit model; a wrong decision costs money ([[finance]]) | Shadow for weeks (compare decisions and would-be losses), then a small canary |
| Swap the LLM behind a support bot to a cheaper model | Shadow + judge on disagreements, canary on guardrails, A/B on resolution rate |
| Prompt wording change | Canary with guardrails; A/B only if the change targets a business metric |
| Ranking or recommendation model ([[recommender-systems]]) | Interleaving or A/B on engagement; guardrail on latency |
| Infrastructure-only change (new GPU type, quantised weights, new inference server) | Shadow for latency/output parity, then canary; no A/B needed if outputs match |
| Very low traffic | Offline evals plus shadow; A/B will not reach significance in reasonable time |

## Setup & code
Numpy/scipy simulation of the whole path: sticky hashing assignment, shadow agreement, a canary ramp with automatic rollback on guardrails, and an A/B readout with a confidence interval.
```python
import hashlib
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# 1) Sticky assignment: hash(user, experiment) -> bucket, so a user always sees the same variant.
def variant(user_id: str, experiment: str, treatment_pct: float) -> str:
    h = int(hashlib.sha256(f"{experiment}:{user_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "treatment" if h < treatment_pct / 100 else "control"

users = [f"u{i}" for i in range(20000)]
share = np.mean([variant(u, "prompt-v4", 10) == "treatment" for u in users])
print(f"treatment share at 10%: {share:.3f}; stable: {variant('u7', 'prompt-v4', 10) == variant('u7', 'prompt-v4', 10)}")

# 2) Shadow traffic: new model sees a copy of real requests, its answers are logged, never shown.
old = rng.integers(0, 3, 2000)                       # labels from the current classifier
new = np.where(rng.random(2000) < 0.93, old, rng.integers(0, 3, 2000))
agree = (old == new).mean()
print(f"shadow agreement {agree:.3f}; {int((old != new).sum())} disagreements go to a judge/human review")

# 3) Canary with guardrails: ramp traffic, compare against baseline at each step, roll back on breach.
def simulate(n, err, p_lat_ms, good):
    return {"err": rng.random(n) < err, "lat": rng.lognormal(np.log(p_lat_ms), 0.4, n), "good": rng.random(n) < good}

def canary(candidate, steps=(1, 5, 25, 50, 100), per_step=40000):
    for pct in steps:
        n_c = int(per_step * pct / 100); n_b = per_step - n_c
        if n_b == 0:
            print(f"  {pct:3d}%: fully rolled out"); return "promoted"
        b, c = simulate(n_b, **BASELINE), simulate(n_c, **candidate)
        # errors: one-sided two-proportion z-test "candidate worse"
        p1, p2 = c["err"].mean(), b["err"].mean(); p = (c["err"].sum() + b["err"].sum()) / (n_c + n_b)
        z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n_c + 1 / n_b)); p_err = 1 - stats.norm.cdf(z)
        p95_c, p95_b = np.percentile(c["lat"], 95), np.percentile(b["lat"], 95)
        good_c, good_b = c["good"].mean(), b["good"].mean()
        print(f"  {pct:3d}%: err {p1:.4f} vs {p2:.4f} (p={p_err:.3f})  p95 {p95_c:.0f} vs {p95_b:.0f} ms  thumbs-up {good_c:.3f} vs {good_b:.3f}")
        if p_err < 0.01 or p95_c > 1.2 * p95_b or good_c < good_b - 0.02:
            print(f"  ROLLBACK at {pct}%: guardrail breached"); return "rolled back"
    return "promoted"

BASELINE = dict(err=0.010, p_lat_ms=800, good=0.80)
print("good candidate:"); canary(dict(err=0.010, p_lat_ms=780, good=0.81))
print("slow candidate:"); canary(dict(err=0.010, p_lat_ms=1100, good=0.81))

# 4) A/B readout on the primary metric (task success), with a 95% CI on the difference.
n = 15000
succ_c, succ_t = rng.random(n) < 0.62, rng.random(n) < 0.64
d = succ_t.mean() - succ_c.mean()
se = np.sqrt(succ_t.mean() * (1 - succ_t.mean()) / n + succ_c.mean() * (1 - succ_c.mean()) / n)
print(f"A/B success: control {succ_c.mean():.3f}, treatment {succ_t.mean():.3f}, diff {d:+.3f} "
      f"(95% CI {d - 1.96 * se:+.3f} .. {d + 1.96 * se:+.3f})")
```

Kubernetes tooling does the traffic part. KServe: add `canaryTrafficPercent` to the predictor of an `InferenceService` with a new `storageUri`; KServe splits traffic between the last rolled-out revision and the new one, and setting it back to 0 is the rollback (KServe docs: canary is supported only in serverless deployment mode).
```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "sklearn-iris"
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: "gs://kserve-examples/models/sklearn/1.0/model-2"
    canaryTrafficPercent: 10
```
Argo Rollouts generalises this for any Deployment: a `canary` strategy with `steps` (`setWeight`, `pause`, `analysis`) and an `AnalysisTemplate` that queries Prometheus or similar; a failed analysis aborts the rollout and sets the canary weight back to zero. Istio `VirtualService` `mirror` plus `mirrorPercentage` copies a share of requests to a shadow version and discards its responses. For LLM API calls without Kubernetes, weighted routing in an LLM gateway ([[llm-gateways-and-routing]]) or a feature-flag service gives the same split at the application layer.

## Choosing / trade-offs
- **Shadow vs canary first:** shadow when a wrong output is expensive or irreversible; skip straight to a 1% canary when outputs are low-stakes and shadow cost (a second LLM call per request) is not worth it.
- **Step size and pause length:** smaller steps and longer pauses catch rare failures but slow delivery; the pause must cover enough requests for the guardrail test to have power (a 1% step on low traffic may see too few requests to judge anything).
- **Automated vs manual promotion:** automate rollback on hard guardrails (errors, latency, safety); keep a human decision for promotion when quality metrics are noisy.
- **Assignment unit:** per user (consistent experience, needed for retention metrics) vs per request (more samples, but users see both variants, which contaminates conversational products) vs per session/conversation (usual choice for chat).
- **Metric choice:** fast proxies (thumbs, retries) give early signal but can disagree with slow business outcomes (churn, chargebacks, resolution). Use proxies for canary guardrails, the business metric for the A/B decision.
- **Sequential monitoring:** peeking at a fixed-horizon test every hour inflates false positives; use a sequential method or only read out at the planned sample size.

## Gotchas
- LLM outputs are non-deterministic: shadow "disagreement" includes sampling noise. Measure the baseline against itself (two runs of the current version) to know the noise floor.
- Caches hide differences: a response or [[semantic-caching]] layer keyed without the model/prompt version serves the old variant's answers to the new arm.
- Novelty and learning effects: users react to *any* change at first; run A/B tests over full weekly cycles.
- Sample ratio mismatch (the observed split differs from the configured one) signals a bug in assignment or logging; check it before reading any metric.
- Shadow traffic doubles load on downstream dependencies (vector DB, tools, provider rate limits); rate-limit the shadow path and make it non-blocking so it cannot slow the user's request.
- Provider-side model changes: an alias model ID can move to a new snapshot under you; pin dated IDs for the baseline arm or the comparison is meaningless ([[model-selection]]).
- Guardrails must include cost per request and refusal/safety rates, not only latency and errors: a new model can be faster and wordier, or more cautious.
- Many simultaneous experiments interact; use mutually exclusive layers or account for interaction.

## Related
- [[statistics-and-ab-testing]] - hypothesis tests, power and sample size.
- [[multi-armed-bandits]], [[thompson-sampling-and-ucb-from-scratch]] - adaptive allocation instead of fixed splits.
- [[llm-evaluation]], [[agent-evaluation]] - offline gates before any traffic, online judges during it.
- [[mlops-lifecycle]] - where rollout sits in the lifecycle.
- [[llm-gateways-and-routing]] - weighted routing and fallbacks between model versions.
- [[llm-observability]], [[opentelemetry-genai-tracing]] - tagging traces with the variant for comparison.
- [[inference-servers-vllm]], [[serving-with-fastapi]] - the serving layer being rolled out.
- [[online-learning-and-concept-drift]] - why a model that won last quarter may lose now.
- [[semantic-caching]], [[feature-stores]] - versioned caches and features that both arms must respect.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/deployment-strategies.md - blue-green, canary and shadow in general.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/feature-flags.md - flags for application-level splits and kill switches.

## References
- KServe canary rollout strategy: https://kserve.github.io/website/docs/model-serving/predictive-inference/rollout-strategies/canary
- Argo Rollouts canary: https://argo-rollouts.readthedocs.io/en/stable/features/canary/
- Argo Rollouts analysis and progressive delivery: https://argo-rollouts.readthedocs.io/en/stable/features/analysis/
- Istio traffic mirroring: https://istio.io/latest/docs/tasks/traffic-management/mirroring/
- Kohavi, Tang, Xu, Trustworthy Online Controlled Experiments (Cambridge University Press, 2020): https://exp-platform.com/
