---
title: User feedback and human review loops for LLM apps (thumbs, implicit signals, annotation queues)
category: llm-apps
tags: [user-feedback, thumbs-up-down, implicit-feedback, human-review, annotation-queue, corrections, langfuse, argilla, scores, llm-as-judge-calibration, cohen-kappa, wilson-interval, selection-bias, eval-dataset, data-flywheel]
use_cases:
  - "add thumbs up/down to our chatbot and actually use the feedback to improve it"
  - "find the worst answers our assistant gave last week and turn them into eval cases"
  - "set up a queue where domain experts review and correct a sample of LLM outputs"
  - "check whether our LLM-as-judge agrees with human reviewers before trusting it"
  - "compare two prompt versions on user satisfaction without being fooled by who bothers to click"
status: draft
last_verified: 2026-10-24
sources:
  - https://github.com/langfuse/langfuse-docs/blob/main/content/docs/observability/features/user-feedback.mdx
  - https://github.com/langfuse/langfuse-docs/blob/main/content/docs/evaluation/evaluation-methods/scores-via-sdk.mdx
  - https://github.com/langfuse/langfuse-docs/blob/main/content/docs/evaluation/evaluation-methods/annotation-queues.mdx
  - https://github.com/langfuse/langfuse-docs/blob/main/content/docs/observability/features/corrections.mdx
  - https://github.com/langfuse/langfuse-python/blob/main/langfuse/_client/client.py
  - https://github.com/argilla-io/argilla/blob/develop/README.md
  - https://doi.org/10.1177/001316446002000104
  - https://doi.org/10.1080/01621459.1927.10502953
---

# User feedback and human review loops for LLM apps (thumbs, implicit signals, annotation queues)

## Summary
Offline evals tell you how an LLM app does on the cases you thought of; production feedback tells you where it fails on the ones you did not. A feedback loop has four parts: capture signals attached to the exact trace that produced an answer (explicit ratings, implicit behaviour, downstream outcomes), sample and route the interesting traces to human reviewers, turn reviewed failures and corrections into eval cases and training data, and measure whether a change helped. This topic covers the data model, the statistics that keep the numbers honest (response bias, small samples, judge-human agreement) and a runnable stdlib implementation, using Langfuse's scores, annotation queues and corrections as the reference tool.

## Key concepts
- **Attach feedback to the trace**: the backend returns its trace or message ID to the client, and the feedback call references it, so a rating can be joined to the prompt version, retrieved context, model and tool calls that produced the answer ([[llm-observability]]). In Langfuse feedback is a *score* on a trace or observation: `name`, `value`, `data_type` (`NUMERIC`, `CATEGORICAL`, `BOOLEAN`, `TEXT`), optional `comment`, and an optional score `id` that acts as an idempotency key so a user changing their vote updates rather than duplicates.
- **Explicit feedback**: thumbs, stars, free-text comments. Langfuse's guide lists the trade-off: a clear, actionable signal, but low response rates and unhappy users more likely to respond.
- **Implicit feedback**: behaviour that implies quality without asking: copying the answer, accepting a code suggestion, regenerating, rephrasing the same question soon after, abandoning the session, escalating to a human. High volume and no user effort, but ambiguous and needs interpretation (a copy may be of a wrong answer).
- **Outcome feedback**: a downstream business result recorded server-side later: ticket closed without escalation, form submitted, refund avoided. Usually the most trustworthy signal and the slowest.
- **Selection bias**: the share of thumbs-up among people who clicked is not the satisfaction rate of all users. Compare variants on the same population, track response rate per variant, and estimate quality from a random sample reviewed by humans.
- **Small-sample uncertainty**: report a confidence interval for rates, such as the Wilson score interval for a proportion (Wilson 1927), not a bare percentage; two variants whose intervals overlap heavily have not been separated ([[statistics-and-ab-testing]]).
- **Human review queues**: Langfuse annotation queues assign traces, observations or sessions to reviewers with a fixed score config (the dimensions and allowed values) and a keyboard-driven UI. Fill them from a stratified sample: all negative feedback, a random slice of everything (to catch failures nobody flagged), and high-risk categories.
- **Corrections**: the reviewer writes what the model should have said. Langfuse stores one corrected output per trace or observation (a score with `dataType: "CORRECTION"`); exported with the input, it is an eval reference answer and supervised or preference fine-tuning data ([[fine-tuning-and-peft]], [[rlhf-and-preference-optimization]]).
- **Judge calibration**: an LLM-as-judge scales review to every trace only after it agrees with humans on a shared sample. Measure agreement with Cohen's kappa (Cohen 1960), which corrects raw agreement for chance; raw percent agreement looks high whenever one label dominates ([[llm-as-judge-and-eval-driven-development]]).
- **Data flywheel**: reviewed failures become regression cases in the offline eval set; every prompt or model change runs against it before release ([[llm-evaluation]], [[prompt-management-and-versioning]]).

## When to use / scenarios
- Any user-facing assistant past prototype: support bots, internal knowledge assistants, copilots ([[customer-support]]).
- Choosing between prompt or model versions in production, together with an A/B split ([[ab-testing-and-canary-for-models]]).
- Regulated or high-stakes domains where a sample of outputs must be reviewed by experts anyway (health, legal, finance); the queue is the audit trail ([[ai-regulation-checklist]]).
- Building a fine-tuning or preference dataset from real traffic instead of synthetic prompts ([[data-labeling-and-synthetic-data]]).
- Not a replacement for an offline eval set: feedback arrives after users were affected, is sparse and biased. Build the eval set first and feed it from feedback.
- Low-volume internal tools may get too few clicks to measure; schedule expert review of a random sample instead.

## Setup & code
Recording feedback with Langfuse's Python SDK (from its docs; needs a Langfuse project and keys). A browser can send scores directly with a public key that has only score-ingestion permission:
<!-- skip-check: needs the langfuse package, a Langfuse server and API keys -->
```python
from langfuse import get_client

langfuse = get_client()                         # reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL
langfuse.create_score(                          # explicit feedback from the UI, via your backend
    trace_id=trace_id, name="user-feedback", value=1, data_type="BOOLEAN",
    comment=user_comment, score_id=f"user-feedback-{trace_id}",   # idempotent: a changed vote overwrites
)
langfuse.create_score(                          # outcome feedback recorded later
    trace_id=ticket.trace_id, name="ticket-resolution", value=1 if ticket.closed else 0,
)
```
Runnable, stdlib only: a feedback store with idempotent votes, implicit signals, response-bias-aware comparison of two prompt versions with Wilson intervals, a stratified review queue, judge-human agreement with Cohen's kappa, and export of reviewed failures as eval cases. Users and outcomes are simulated so the true quality is known and the biases are visible.
```python
import math
import random
from collections import Counter, defaultdict

rng = random.Random(0)

def wilson(successes, n, z=1.96):
    """Wilson score interval for a proportion (z=1.96 is the normal quantile for 95%)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)

def cohen_kappa(a, b):
    """Agreement between two raters on the same items, corrected for chance agreement."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0

class FeedbackStore:
    def __init__(self):
        self.traces, self.scores = {}, {}         # scores keyed by score_id -> last write wins
    def log_trace(self, trace_id, **attrs):
        self.traces[trace_id] = attrs
    def score(self, trace_id, name, value, score_id=None):
        self.scores[score_id or f"{name}-{trace_id}-{len(self.scores)}"] = (trace_id, name, value)
    def by_name(self, name):
        out = {}
        for tid, n, v in self.scores.values():
            if n == name:
                out[tid] = v
        return out

# Simulated traffic: two prompt versions with different TRUE answer quality (unknown in real life).
TRUE_GOOD = {"v1": 0.70, "v2": 0.78}             # simulation inputs, used only to generate events
P_CLICK = {True: 0.05, False: 0.20}              # simulated: unhappy users click more often
store = FeedbackStore()
for i in range(6000):
    tid, version = f"t{i}", ("v1" if i % 2 else "v2")
    good = rng.random() < TRUE_GOOD[version]
    category = rng.choice(["billing", "billing", "account", "legal"])
    store.log_trace(tid, version=version, category=category, good=good, question=f"q{rng.randrange(800)}")
    if rng.random() < P_CLICK[good]:
        store.score(tid, "user-feedback", 1 if good else 0, score_id=f"user-feedback-{tid}")
        if rng.random() < 0.1:                   # some users change their vote: same id, no duplicate
            store.score(tid, "user-feedback", 1 if good else 0, score_id=f"user-feedback-{tid}")
    retried = rng.random() < (0.05 if good else 0.35)
    if retried:
        store.score(tid, "implicit-retry", 1)

fb = store.by_name("user-feedback")
retry = store.by_name("implicit-retry")
print("explicit feedback, by version (response rate, thumbs-up rate with 95% Wilson CI):")
for v in ("v1", "v2"):
    ids = [t for t, a in store.traces.items() if a["version"] == v]
    votes = [fb[t] for t in ids if t in fb]
    lo, hi = wilson(sum(votes), len(votes))
    true_rate = sum(store.traces[t]["good"] for t in ids) / len(ids)
    print(f"  {v}: responded {len(votes) / len(ids):.1%}, thumbs-up {sum(votes) / len(votes):.1%} "
          f"[{lo:.1%}, {hi:.1%}]  | simulated true quality {true_rate:.1%}  | retry rate "
          f"{sum(t in retry for t in ids) / len(ids):.1%}")
print("  -> clickers under-represent happy users: the thumbs-up rate is far below true quality,")
print("     but the comparison between versions on the same population still points the same way.")

# Stratified review queue: every negative vote + a random slice + every high-risk category item.
negative = [t for t, v in fb.items() if v == 0]
random_slice = rng.sample(sorted(store.traces), 150)
high_risk = [t for t, a in store.traces.items() if a["category"] == "legal"][:100]
queue = list(dict.fromkeys(negative[:150] + random_slice + high_risk))   # dedupe, keep order
print(f"review queue: {len(queue)} items ({len(negative[:150])} negative, 150 random, {len(high_risk)} high-risk)")

# Reviewers label the queue; estimate quality from the RANDOM slice only (unbiased by clicks).
human = {t: store.traces[t]["good"] for t in queue}               # stand-in for expert labels
s = sum(human[t] for t in random_slice)
lo, hi = wilson(s, len(random_slice))
print(f"quality from random human review: {s / len(random_slice):.1%} [{lo:.1%}, {hi:.1%}]")

# Calibrate an LLM judge against the humans before letting it score all traffic.
def judge(tid):                                    # simulated judge: right most of the time, lenient on errors
    truth = store.traces[tid]["good"]
    return truth if rng.random() < 0.85 else (rng.random() < 0.7)
judge_labels = [judge(t) for t in queue]
human_labels = [human[t] for t in queue]
raw = sum(a == b for a, b in zip(judge_labels, human_labels)) / len(queue)
print(f"judge vs human: raw agreement {raw:.1%}, Cohen's kappa {cohen_kappa(judge_labels, human_labels):.2f}")

# Export reviewed failures as regression eval cases, one per distinct question.
cases, seen = [], set()
for t in queue:
    a = store.traces[t]
    if not human[t] and a["question"] not in seen:
        seen.add(a["question"])
        cases.append({"input": a["question"], "category": a["category"], "source_trace": t,
                      "expected": None})               # filled from the reviewer's correction
by_cat = defaultdict(int)
for c in cases:
    by_cat[c["category"]] += 1
print(f"{len(cases)} new eval cases from reviewed failures, by category: {dict(by_cat)}")
```

## Choosing / trade-offs
| Signal | Volume | Bias / ambiguity | Best use |
|---|---|---|---|
| Thumbs up/down | Low | Response bias; unhappy users over-represented | Finding failures; comparing variants on the same population |
| Free-text comment | Very low | Very selective | Explaining failures; triage categories |
| Implicit (retry, rephrase, copy, accept) | High | Ambiguous; define and validate each against human labels | Trend monitoring, routing traces to review |
| Outcome (resolved, converted, escalated) | Medium, delayed | Confounded by things outside the answer | The metric the business cares about |
| Random human review | Low, costly | Unbiased if the sample is random | Ground-truth quality estimate, judge calibration |
| LLM-as-judge on all traffic | Every trace | Judge bias; drifts with judge model changes | Scale, after kappa against humans is acceptable |

- **Tooling**: Langfuse (open source, self-hostable) covers scores, queues and corrections next to tracing; other observability platforms have similar feedback APIs ([[llm-observability]]). Argilla is an open-source annotation tool for building datasets with domain experts; its README states that it is now in maintenance mode (bug fixes, no new features), which matters for a new long-lived deployment. Label Studio is the general-purpose alternative ([[data-labeling-and-synthetic-data]]).
- **Where reviewers' time goes**: all negatives plus a random slice is a common split; the random slice is the only part that estimates overall quality.
- **Privacy**: traces in a review queue contain user data; redact before review or restrict queue access, and respect retention rules ([[pii-redaction-and-pseudonymization]]).

## Gotchas
- Feedback with no trace ID is close to useless: you cannot see which prompt, context or model produced the answer.
- The thumbs-up rate is not satisfaction. A UI change that makes the buttons more visible changes the rate without any change in quality; track response rate alongside.
- A version that gets fewer clicks is not better; compare rates among responders and confirm with random review.
- Repeated votes, double-clicks and retries create duplicate scores unless the score ID is deterministic (one per user per message).
- Implicit signals need a definition and a check: "regenerate" can mean "wrong" or "show me another option". Validate each against human labels before alerting on it.
- Raw judge-human agreement looks good when most answers are fine; use kappa or per-class precision and recall for the failure class.
- Reviewers drift and disagree: write a rubric with examples, measure inter-annotator agreement on overlapping items, and recalibrate.
- Feeding user feedback straight into fine-tuning can teach the model to please rather than to be right; route it through review first ([[rlhf-and-preference-optimization]]).
- A browser-side key used to submit feedback must only be able to write scores, not read traces.

## Related
- [[llm-observability]] - the traces feedback attaches to.
- [[llm-as-judge-and-eval-driven-development]] - judges calibrated against the human labels collected here.
- [[llm-evaluation]] - the offline eval set that reviewed failures feed.
- [[rag-evaluation]] - retrieval-specific review dimensions (groundedness, context relevance).
- [[ab-testing-and-canary-for-models]] - online comparison of prompt and model versions.
- [[prompt-management-and-versioning]] - tagging traces with prompt versions so feedback can be compared.
- [[statistics-and-ab-testing]] - confidence intervals and significance for feedback rates.
- [[data-labeling-and-synthetic-data]] - annotation tools and labelling workflows.
- [[rlhf-and-preference-optimization]] - using corrections and pairwise preferences as training data.
- [[pii-redaction-and-pseudonymization]] - cleaning traces before humans review them.
- [[customer-support]] - a scenario where outcome feedback (resolution) is available.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/feature-flags.md - rolling out the variants feedback compares.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/observability-logs-metrics-traces.md - trace IDs and correlation across services.

## References
- Langfuse, User Feedback (explicit vs implicit, trace ID to frontend, server-side feedback): https://github.com/langfuse/langfuse-docs/blob/main/content/docs/observability/features/user-feedback.mdx
- Langfuse, Scores via SDK (`create_score`, data types, idempotent score id): https://github.com/langfuse/langfuse-docs/blob/main/content/docs/evaluation/evaluation-methods/scores-via-sdk.mdx
- Langfuse, Annotation Queues: https://github.com/langfuse/langfuse-docs/blob/main/content/docs/evaluation/evaluation-methods/annotation-queues.mdx
- Langfuse, Corrections: https://github.com/langfuse/langfuse-docs/blob/main/content/docs/observability/features/corrections.mdx
- Langfuse Python SDK `create_score` signature (`score_id`, `data_type`) and environment variables: https://github.com/langfuse/langfuse-python/blob/main/langfuse/_client/client.py
- Argilla README (purpose, maintenance-mode notice): https://github.com/argilla-io/argilla/blob/develop/README.md
- Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and Psychological Measurement 20(1). https://doi.org/10.1177/001316446002000104
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. JASA 22(158). https://doi.org/10.1080/01621459.1927.10502953
