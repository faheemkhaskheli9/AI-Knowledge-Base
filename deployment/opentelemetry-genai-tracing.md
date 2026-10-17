---
title: OpenTelemetry GenAI tracing (semantic conventions for LLMs and agents)
category: deployment
tags: [opentelemetry, otel, tracing, semantic-conventions, gen_ai, agents, tool-calls, otlp, observability]
use_cases:
  - "trace an agent's model calls and tool calls with OpenTelemetry instead of a vendor SDK"
  - "send LLM traces to our existing Jaeger, Tempo, Datadog or Langfuse backend over OTLP"
  - "use the standard gen_ai.* attributes so dashboards work across OpenAI, Anthropic and agent frameworks"
  - "record token usage and tool errors per agent run without logging prompt contents"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/open-telemetry/semantic-conventions-genai
  - https://opentelemetry.io/docs/specs/semconv/gen-ai/
  - https://pypi.org/project/opentelemetry-instrumentation-openai-v2/
  - https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation-genai
  - https://opentelemetry.io/docs/languages/python/
---

# OpenTelemetry GenAI tracing (semantic conventions for LLMs and agents)

## Summary
OpenTelemetry (OTel) defines GenAI semantic conventions: standard span names, attributes (`gen_ai.*`), metrics and events for model calls, embeddings, retrieval, tool execution and agent runs. If your code, SDK instrumentations and agent frameworks all emit these, one trace shows an agent run end to end and any OTLP backend can chart tokens, latency and errors per model or tool. This topic covers the conventions and how to emit them; [[llm-observability]] covers why and which platform.

## Key concepts
- **Status: Development.** The GenAI conventions are not stable and still change (they now live in their own repo, `open-telemetry/semantic-conventions-genai`, built on the core conventions). Pin instrumentation versions and expect renames.
- **Span per operation**, typed by `gen_ai.operation.name`. Well-known values include `chat`, `generate_content`, `text_completion`, `embeddings`, `retrieval`, `execute_tool`, `invoke_agent`, `create_agent`, `invoke_workflow`, `plan`, and memory operations (`search_memory`, `create_memory`, ...).
- **Span names** follow `{operation} {target}`: `chat {gen_ai.request.model}`, `execute_tool {gen_ai.tool.name}`, `invoke_agent {gen_ai.agent.name}`, `retrieval {gen_ai.data_source.id}`.
- **Span kind:** model calls to a remote API are `CLIENT`; in-process agent runs and tool executions are `INTERNAL`.
- **Core attributes:** `gen_ai.provider.name` (well-known values such as `anthropic`, `openai`, `aws.bedrock`, `gcp.vertex_ai`), `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` (input includes cached tokens; `gen_ai.usage.cache_read.input_tokens` breaks them out), `gen_ai.response.finish_reasons`, `gen_ai.conversation.id`, `gen_ai.agent.name`, `gen_ai.tool.name`, `gen_ai.tool.call.id`, `error.type`.
- **Content is opt-in.** `gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`, `gen_ai.tool.call.arguments` / `.result` may hold PII; instrumentations must not capture them by default. The spec gives `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` as the example opt-in switch.
- **Older vs newer conventions.** The Python OpenAI instrumentation defaults to semconv v1.30.0 (which used `gen_ai.system` and per-message events) unless `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` is set. Mixed fleets will emit both shapes for a while.
- **Metrics** exist too (for example `gen_ai.client.operation.duration`, and the newer `gen_ai.client.inference.*` token and latency metrics); derive dashboards from metrics, debug from traces.
- **Export path:** SDK -> OTLP exporter -> (optional) OpenTelemetry Collector for batching, redaction and fan-out -> one or more backends.

## When to use / scenarios
- You already run OTel for the rest of the stack and want LLM calls inside the same distributed trace as the HTTP request, DB query and queue message.
- Multi-framework agents (one service on the OpenAI SDK, another on LangChain or the OpenAI Agents SDK) that should share one schema.
- Vendor neutrality: send the same spans to a self-hosted Jaeger/Tempo plus an LLM-specific UI that ingests OTLP ([[llm-observability]] covers Langfuse, which accepts OTel).
- Cost and SLO reporting per model, tenant or feature from `gen_ai.usage.*` ([[cost-and-latency]]).
- Not the right tool alone for: offline evaluation datasets and human feedback UIs (pair it with an LLM observability platform, [[llm-evaluation]]); a single-script prototype (print/log is enough).

## Setup & code
Install (versions current on 2026-10-15; the GenAI instrumentations are pre-release `b` versions):
```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
pip install opentelemetry-instrumentation-openai-v2   # official OpenAI SDK instrumentation
export OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental     # newest gen_ai shape
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=span_only  # only if content capture is allowed
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318            # your collector
```
Auto-instrumenting an SDK is one line (`OpenAIInstrumentor().instrument()` from `opentelemetry.instrumentation.openai_v2`). Other official instrumentations live under `instrumentation-genai/` in opentelemetry-python-contrib; the semconv repo's `reference/` README lists which client libraries and agent frameworks emit which spans.

Manual instrumentation of an agent loop, so tool calls and the agent run get spans even when no auto-instrumentation covers them (tested with opentelemetry-sdk 1.45.1; the model call is stubbed):
<!-- skip-check: needs opentelemetry-sdk, which is not in the check environment -->
```python
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode

exporter = InMemorySpanExporter()      # production: OTLPSpanExporter from opentelemetry.exporter.otlp.proto.http.trace_exporter
provider = TracerProvider(resource=Resource.create({"service.name": "support-agent"}))
provider.add_span_processor(SimpleSpanProcessor(exporter))   # production: BatchSpanProcessor
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("support-agent")
MODEL = "claude-haiku-4-5"

def chat(messages):
    with tracer.start_as_current_span(f"chat {MODEL}", kind=SpanKind.CLIENT) as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "anthropic")
        span.set_attribute("gen_ai.request.model", MODEL)
        resp = {"model": MODEL, "in": 812, "out": 64, "stop": "tool_use"}   # stand-in for client.messages.create(...)
        span.set_attribute("gen_ai.response.model", resp["model"])
        span.set_attribute("gen_ai.usage.input_tokens", resp["in"])
        span.set_attribute("gen_ai.usage.output_tokens", resp["out"])
        span.set_attribute("gen_ai.response.finish_reasons", [resp["stop"]])
        return resp

def execute_tool(name, call_id, args):
    with tracer.start_as_current_span(f"execute_tool {name}", kind=SpanKind.INTERNAL) as span:
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", name)
        span.set_attribute("gen_ai.tool.call.id", call_id)
        try:
            if args.get("order_id") is None:
                raise ValueError("missing order_id")
            return {"status": "shipped"}
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute("error.type", type(e).__name__)
            raise

with tracer.start_as_current_span("invoke_agent support-bot", kind=SpanKind.INTERNAL) as root:
    root.set_attribute("gen_ai.operation.name", "invoke_agent")
    root.set_attribute("gen_ai.agent.name", "support-bot")
    root.set_attribute("gen_ai.conversation.id", "conv-123")
    chat([{"role": "user", "content": "Where is order 42?"}])
    execute_tool("get_order_status", "call_1", {"order_id": 42})
    chat([])

for s in exporter.get_finished_spans():
    print(s.name, dict(s.attributes))
```

What the standard buys you on the backend side: queries that work no matter which SDK emitted the spans. Stdlib sketch over spans as a backend stores them:
```python
from collections import defaultdict

# Spans as a backend would store them (OTLP flattened): one agent run with two model calls and a tool.
spans = [
    {"trace": "t1", "id": "a", "parent": None, "name": "invoke_agent support-bot", "ms": 2400, "status": "OK",
     "attrs": {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "support-bot", "gen_ai.conversation.id": "c9"}},
    {"trace": "t1", "id": "b", "parent": "a", "name": "chat claude-haiku-4-5", "ms": 900, "status": "OK",
     "attrs": {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "anthropic", "gen_ai.request.model": "claude-haiku-4-5",
               "gen_ai.usage.input_tokens": 812, "gen_ai.usage.output_tokens": 64}},
    {"trace": "t1", "id": "c", "parent": "a", "name": "execute_tool get_order_status", "ms": 1100, "status": "ERROR",
     "attrs": {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "get_order_status", "error.type": "timeout"}},
    {"trace": "t1", "id": "d", "parent": "a", "name": "chat claude-haiku-4-5", "ms": 350, "status": "OK",
     "attrs": {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "anthropic", "gen_ai.request.model": "claude-haiku-4-5",
               "gen_ai.usage.input_tokens": 950, "gen_ai.usage.output_tokens": 120}},
]

def print_tree(spans, parent=None, depth=0):
    for s in (s for s in spans if s["parent"] == parent):
        print(f"{'  ' * depth}{s['name']:<34}{s['ms']:>6} ms  {s['status']}")
        print_tree(spans, s["id"], depth + 1)

print_tree(spans)

# Questions you answer from GenAI attributes, independent of which SDK or framework emitted them.
tokens = defaultdict(lambda: [0, 0])
tool_errors = defaultdict(int)
for s in spans:
    a = s["attrs"]
    if a.get("gen_ai.operation.name") == "chat":
        tokens[a["gen_ai.request.model"]][0] += a.get("gen_ai.usage.input_tokens", 0)
        tokens[a["gen_ai.request.model"]][1] += a.get("gen_ai.usage.output_tokens", 0)
    if a.get("gen_ai.operation.name") == "execute_tool" and s["status"] == "ERROR":
        tool_errors[(a["gen_ai.tool.name"], a.get("error.type", "_OTHER"))] += 1
print("tokens per model (in, out):", dict(tokens))
print("tool errors:", dict(tool_errors))
root = next(s for s in spans if s["parent"] is None)
for s in spans:
    if s["parent"] == root["id"]:
        print(f"  {s['name']:<34} {s['ms'] / root['ms']:.0%} of agent latency")
```

## Choosing / trade-offs
| Decision | Option A | Option B |
|---|---|---|
| Instrumentation | Auto (SDK/framework instrumentors): fast, consistent, may lag SDK releases | Manual spans: full control, needed for your own tools and agent loop |
| Content capture | Off (default): safe, cheap; debug with IDs and metadata | On (`span_only` / events / upload hook to object storage): full debugging, PII and storage cost |
| Where content goes | Span attributes: simple, but big spans and backend size limits | Events/logs or external upload with a reference: smaller spans, separate retention |
| Export | Direct OTLP to vendor: fewer moving parts | Via Collector: redaction, sampling, multi-backend fan-out, retries |
| Sampling | Head sampling (decide at start): cheap, may drop the failing run | Tail sampling in the Collector (keep errors, slow and expensive traces): better signal, needs a stateful collector |
| Conventions version | Default (older, stable within a release) | `gen_ai_latest_experimental`: newest attributes, breaking changes possible |

## Gotchas
- Context propagation across async code, threads and queues: a tool call run in a thread pool or a background worker becomes a separate trace unless you pass the context. Agent frameworks with their own executors are a frequent cause of orphan spans.
- Streaming: end the model span when the stream finishes, not when the call returns; record time to first chunk if you care about it.
- Two instrumentations for the same call (SDK instrumentor plus framework instrumentor) produce duplicate spans and double-counted tokens; the spec says not to record two spans for one call, but mixing libraries can still do it. Check one trace before trusting dashboards.
- Attribute size limits in SDKs and backends silently truncate long prompts; do not rely on spans as the system of record for transcripts.
- `gen_ai.request.model` (what you asked for) and `gen_ai.response.model` (the snapshot that answered) can differ; alias IDs resolve to dated snapshots. Group cost by response model.
- `BatchSpanProcessor` drops unsent spans if a short-lived process (serverless, CLI) exits without `provider.shutdown()` or `force_flush()`.
- Prompt content in traces is sensitive data: apply the same retention, access control and redaction as your logs ([[ai-security-privacy-compliance]]). Redact in the Collector, not in every service.
- Treat conventions churn as a migration: dashboards keyed on `gen_ai.system` break when services move to `gen_ai.provider.name`.

## Related
- [[llm-observability]] - why trace LLM apps, platforms such as Langfuse that ingest OTel.
- [[agents]], [[multi-agent-systems]], [[tool-calling]] - the loops these spans describe.
- [[model-context-protocol]] - MCP tool calls have their own OTel conventions in the same repo.
- [[agent-evaluation]], [[llm-evaluation]] - turning traces into eval datasets.
- [[cost-and-latency]] - token and latency budgets the metrics feed.
- [[semantic-caching]] - log cache hits as spans so savings and false hits are visible.
- [[ab-testing-and-canary-for-models]] - comparing variants by trace attributes.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/observability-logs-metrics-traces.md - OTel, traces and sampling in general.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/slis-slos-error-budgets.md - SLOs built on these metrics.

## References
- OTel GenAI semantic conventions repo (spans, agent spans, metrics, events, registry): https://github.com/open-telemetry/semantic-conventions-genai
- GenAI conventions on the OTel site: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- OpenAI instrumentation (env vars, content capture modes): https://pypi.org/project/opentelemetry-instrumentation-openai-v2/
- Python GenAI instrumentations: https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/instrumentation-genai
- OpenTelemetry Python: https://opentelemetry.io/docs/languages/python/
