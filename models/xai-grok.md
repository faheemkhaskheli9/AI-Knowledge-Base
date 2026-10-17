---
title: xAI Grok models
category: models
tags: [xai, grok, llm, api, openai-compatible, reasoning, long-context, grok-imagine]
use_cases:
  - "call Grok from an app that already uses the OpenAI SDK"
  - "process a 500K-token codebase or contract bundle in one request"
  - "add a cheaper coding model behind an internal developer agent"
  - "compare Grok against Claude and GPT on our own eval set before switching"
status: draft
last_verified: 2026-10-15
sources:
  - https://docs.x.ai/developers/models
  - https://docs.x.ai/developers/pricing
  - https://docs.x.ai/developers/models/grok-4.7
  - https://docs.x.ai/developers/grok-4-7
  - https://docs.x.ai/developers/quickstart
  - https://docs.x.ai/developers/regions
  - https://docs.x.ai/developers/model-capabilities/text/reasoning
  - https://docs.x.ai/developers/release-notes
---

# xAI Grok models

## Summary
xAI serves closed-weight Grok models through an API at `api.x.ai` that is
compatible with the OpenAI SDK (Responses and Chat Completions), plus a native
gRPC SDK (`xai-sdk`). The lineup has reasoning text models with long context
(500K to 1M tokens), a coding model, and separate image, video and voice APIs.

## Key concepts
- Naming: the bare name (`grok-4.7`) points to the latest stable version,
  `-latest` to the newest version, and a dated suffix pins one release that
  never changes. The docs suggest the bare name or `-latest` for most users;
  pin a dated ID when you need reproducible outputs.
- Reasoning is always on for grok-4.7 / 4.6 / 4.5; `reasoning_effort` picks
  the level (grok-4.7: `low`, `medium`, `high` (default), `xhigh`). It cannot
  be switched off.
- On the Responses API, grok-4.7 returns `reasoning.encrypted_content` even if
  you did not ask for it in `include`.
- Long-context pricing: once a prompt goes past 200K tokens, input, cached and
  output rates roughly double.
- Regional endpoint: `https://us.api.x.ai/v1` (US data handling) offers only some
  models (grok-4.7, grok-4.6 at time of writing) at 1.1x standard rates.
- Also on Google Cloud Vertex AI with a publisher prefix (`xai/grok-4.7`); see
  [[managed-cloud-ai-platforms]].

Current IDs as of 2026-10-15 (sources: docs.x.ai models, pricing and per-model
pages, read through search snippets rather than the full pages, so re-check
before committing to a price):

| Model | API ID | Context | $/MTok in / cached / out (prompt <200K) |
|---|---|---|---|
| Flagship | `grok-4.7` | 500K | 2.00 / - / 6.00 |
| Previous flagship | `grok-4.6` | 500K | 2.00 / 0.50 / 6.00 |
| Older | `grok-4.5` | 500K | 2.00 / 0.30 / 6.00 |
| Cheaper, longest context | `grok-4.3` | 1M | 1.25 / 0.20 / 2.50 |
| Coding | `grok-build-0.1` | 256K | 1.00 / 0.20 / 2.00 |

`grok-4.20` variants (reasoning, non-reasoning, multi-agent) are also on the
pricing page. Image, video (Grok Imagine) and voice are billed per image, second,
minute or character; see the pricing page.
`grok-imagine-image-quality` retires on 2026-11-02; move to another Grok Imagine image model listed on the pricing page. <!-- valid-until: 2026-11-02 -->
"Grok 4.7 Fast" is the same model served faster at 2x token rates, and the docs
say it is only offered in Cursor and Grok Build, not on the public API.

## When to use / scenarios
- You already use the OpenAI SDK and want a second vendor: change `base_url`
  and the model ID ([[llm-gateways-and-routing]]).
- Very long inputs in one call (whole repos, deposition bundles) with grok-4.3's
  1M window, mindful of the 200K price step ([[long-context]]).
- Internal coding agents on a cheaper coding model (`grok-build-0.1`);
  benchmark against [[anthropic-claude]] and [[openai-gpt]] on your tasks.
- Not for: data that must stay on your hardware (closed weights; use
  [[meta-llama]], [[qwen]] or [[google-gemma]]), or workloads needing EU data
  residency (no EU regional endpoint was listed; check the regions page).

## Setup & code
```bash
pip install openai          # or: pip install xai-sdk (native gRPC client)
export XAI_API_KEY=...
```
<!-- skip-check: needs an xAI API key and network -->
```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1")
r = client.responses.create(
    model="grok-4.7",
    reasoning={"effort": "low"},
    input="Classify this ticket as billing, tech or other: 'card declined twice'",
)
print(r.output_text)
```
The native `xai-sdk` takes the bare host (`api.x.ai`, or `us.api.x.ai`), not
the `/v1` URL.

Cost check with the 200K-token price step (runs offline; rates copied from the
table above, long-context rates from the pricing page):
```python
RATES = {  # $/MTok: (input, output) below 200K prompt tokens, (input, output) at or above
    "grok-4.6": ((2.00, 6.00), (4.00, 12.00)),
    "grok-4.3": ((1.25, 2.50), (2.50, 5.00)),
    "grok-build-0.1": ((1.00, 2.00), (2.00, 4.00)),
}

def cost(model, prompt_tokens, output_tokens):
    short, long_ = RATES[model]
    rin, rout = long_ if prompt_tokens >= 200_000 else short
    return (prompt_tokens * rin + output_tokens * rout) / 1e6

for p in (150_000, 250_000):
    print(p, {m: round(cost(m, p, 4_000), 3) for m in RATES})
```

## Choosing / trade-offs
- grok-4.7 for hardest tasks; grok-4.3 when cost or a 1M window matters more
  than top quality; grok-build-0.1 for code-heavy agent loops.
- Lower `reasoning_effort` cuts latency and output-token spend; you cannot
  disable reasoning on the 4.5+ models, so a simple classifier may be cheaper
  on another vendor's non-reasoning tier.
- US regional endpoint costs 10% more and serves fewer models.

## Gotchas
- Bare model names move: outputs can change under you. Pin dated IDs in
  production and watch the release notes and retirement pages (a batch of
  older models retired on 2026-05-15).
- Crossing 200K prompt tokens doubles the rate for the whole request, not just
  the overflow.
- Batch discounts apply to some models only.
- Encrypted reasoning content adds payload size; do not log it as if it were
  readable text.
- Third-party price aggregators list Grok prices that differ from xAI's own;
  use docs.x.ai only.

## Related
- [[model-selection]] - cross-provider decision guide.
- [[openai-gpt]] - same SDK shape; the main comparison point.
- [[reasoning-models]] - how reasoning effort trades cost for quality.
- [[long-context]] - when a 1M window helps and when RAG is better.
- [[llm-gateways-and-routing]] - adding Grok as one route behind a gateway.
- [[cost-and-latency]] - keeping per-request spend down.

## References
- https://docs.x.ai/developers/models
- https://docs.x.ai/developers/pricing
- https://docs.x.ai/developers/models/grok-4.7
- https://docs.x.ai/developers/grok-4-7
- https://docs.x.ai/developers/quickstart
- https://docs.x.ai/developers/regions
- https://docs.x.ai/developers/model-capabilities/text/reasoning
- https://docs.x.ai/developers/release-notes
- https://docs.x.ai/developers/migration/may-15-retirement
