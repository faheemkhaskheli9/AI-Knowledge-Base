---
title: LLM observability (tracing, logging, Langfuse, OpenTelemetry)
category: deployment
tags: [observability, tracing, logging, langfuse, opentelemetry, evals, monitoring]
use_cases:
  - "see every prompt, tool call and token count for requests in my LLM app"
  - "debug why an agent gave a bad answer in production"
  - "track cost, latency and error rate per feature or user"
  - "add monitoring to a RAG chatbot without leaking customer data"
status: draft
last_verified: 2026-10-03
sources:
  - https://langfuse.com/docs/observability/get-started
  - https://github.com/open-telemetry/semantic-conventions-genai
  - https://opentelemetry.io/docs/
---

# LLM observability (tracing, logging, Langfuse, OpenTelemetry)

## Summary
LLM apps fail in ways that ordinary logs do not show: a plausible but wrong answer, a wrong tool call, a retrieval miss. Observability records each request as a trace (nested spans for retrieval, model calls, tools) with inputs, outputs, tokens, latency and cost, so you can debug, monitor and build eval datasets from real traffic.

## Key concepts
- Trace = one user request; span = a step inside it (LLM call, retriever, tool). Generations carry model name, token usage and cost.
- Sessions/users group traces into conversations; tags and metadata allow filtering by feature or release.
- OpenTelemetry (OTel) is the vendor-neutral standard; GenAI semantic conventions (attributes such as `gen_ai.request.model`, `gen_ai.usage.input_tokens`) now live in a dedicated repo and are evolving.
- Tools: Langfuse (open source, self-hostable or cloud), plus other LLM observability platforms and any OTel backend.
- Feedback and scores (thumbs, LLM-judge, rule checks) attach to traces and feed [[llm-evaluation]].

## When to use / scenarios
- Any production LLM feature; from day one for agents with multiple tool calls ([[agents]]).
- Cost attribution per customer or feature ([[cost-and-latency]]).
- Regression hunting after a prompt or model change: compare traces and scores by release.
- Regulated settings: audit trail of what the model saw and said, within retention rules ([[ai-security-privacy-compliance]]).

## Setup & code
Langfuse Python SDK (env vars from its docs):
```bash
uv add langfuse
export LANGFUSE_PUBLIC_KEY=...   LANGFUSE_SECRET_KEY=...   LANGFUSE_BASE_URL=https://cloud.langfuse.com
```
```python
from langfuse import get_client, observe
langfuse = get_client()

@observe()                       # records inputs, outputs and timing as a span
def answer(question: str) -> str:
    return "..."                 # call your LLM / retriever here

answer("What is our refund policy?")
langfuse.flush()                 # needed in short-lived scripts
```
Explicit span (from the Langfuse quickstart):
```python
with langfuse.start_as_current_observation(as_type="span", name="process-request") as span:
    span.update(output="Processing complete")
langfuse.flush()
```
Provider/framework integrations (OpenAI wrapper, LangChain callbacks, OTel exporters) are listed in the Langfuse docs; check them for exact import paths for your SDK version.

Plain structured logging is a fine start:
```python
import json, logging, time, uuid
log = logging.getLogger("llm")
def log_call(model, in_tok, out_tok, latency_s, ok, request_id=None):
    log.info(json.dumps({"id": request_id or str(uuid.uuid4()), "model": model,
        "in": in_tok, "out": out_tok, "latency_s": round(latency_s, 3), "ok": ok}))
```

## Choosing / trade-offs
- Hosted platform (fast to adopt) vs self-hosted (data stays in your network) vs plain OTel to your existing stack (no new vendor, fewer LLM-specific views).
- Log full prompts/outputs (best debugging) vs metadata only (privacy, storage cost); sample or redact in production.
- Sync vs async export: async batching keeps latency low but may lose spans on crash; flush on shutdown.
- Online LLM-judge scoring adds cost; sample a percentage.

## Gotchas
- Prompts routinely contain PII; mask before export and set retention limits.
- Forgetting `flush()` in serverless/short scripts drops traces.
- Token counts from providers beat local estimates; use usage fields.
- High-cardinality metadata (raw user text as a tag) bloats storage.
- OTel GenAI conventions are still changing; pin library versions.
- Tracing streamed responses requires closing the span after the stream ends, not at return.

## Related
- [[llm-evaluation]] - turning traces into eval sets.
- [[agents]] and [[multi-agent-systems]] - where traces matter most.
- [[serving-with-fastapi]] - instrumenting the API layer.
- [[mlops-lifecycle]] - monitoring in the wider lifecycle.
- [[guardrails-and-safety]] - logging guardrail triggers.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/observability-logs-metrics-traces.md - logs, metrics and traces in general.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/slis-slos-error-budgets.md - setting latency and quality targets.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/incident-response-and-postmortems.md - what to do when a model regresses in production.

## References
- Langfuse get started: https://langfuse.com/docs/observability/get-started
- OTel GenAI conventions: https://github.com/open-telemetry/semantic-conventions-genai
