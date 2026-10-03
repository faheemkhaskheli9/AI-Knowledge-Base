---
title: DeepSeek models
category: models
tags: [deepseek, open-weights, mit, moe, cheap-api, reasoning]
use_cases:
  - "cut LLM API cost with a very cheap OpenAI-compatible provider"
  - "self-host an MIT-licensed MoE model with 1M context"
  - "run bulk document summarisation at the lowest price per token"
  - "use DeepSeek through the Anthropic-format endpoint in an existing client"
status: draft
last_verified: 2026-10-03
sources:
  - https://api-docs.deepseek.com/
  - https://api-docs.deepseek.com/quick_start/pricing
  - https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
  - https://huggingface.co/deepseek-ai
---

# DeepSeek models

## Summary
DeepSeek (China) publishes frontier-class open-weight MoE models (MIT license
on the V4.1-Flash card) and sells a very low-priced API compatible with both
OpenAI and Anthropic request formats.

## Key concepts
- API model IDs (2026-10): `deepseek-flash` (DeepSeek-V4.1-Flash, default,
  supports vision) and `deepseek-v4-pro` (V4-Pro-0813, no vision). Both: 1M
  context, 384K max output, thinking mode, JSON output, tool calls.
- Old names like `deepseek-v4-flash` are retired and routed to Flash pricing.
- Pricing per 1M tokens (pricing page, ranges reflect peak/off-peak; off-peak is
  half): Flash input miss $0.15-0.3, output $0.6-1.2, cache-hit input
  $0.003-0.006; Pro input miss $0.66-1.32, output $1.98-3.96.
- Open weights: V4.1-Flash is 552B backbone MoE (8B active at prefill, 16B at
  decode), MIT license, continuous reasoning-effort setting 1-100. HF org also
  lists V4-Pro-0813 (1.7T) and a vision experiment.
- Base URLs: `https://api.deepseek.com` (OpenAI format) and
  `https://api.deepseek.com/anthropic`.

## When to use / scenarios
- Cost-dominated, high-volume jobs (summaries, extraction, translation).
- Teams wanting MIT-licensed weights to self-host (via vLLM/SGLang, or
  quantized in llama.cpp/Ollama).
- Reasoning-heavy tasks with tunable effort.
- Not for: regulated data you cannot send to a China-hosted API (self-host
  instead or choose another vendor; see [[ai-security-privacy-compliance]]).

## Setup & code
```bash
pip install openai
export DEEPSEEK_API_KEY=...
```
```python
import os
from openai import OpenAI

c = OpenAI(base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"])
r = c.chat.completions.create(
    model="deepseek-flash",
    messages=[{"role": "user", "content": "Summarise in one line: ..."}],
)
print(r.choices[0].message.content)
```

## Choosing / trade-offs
- Flash for nearly everything; Pro when quality matters and vision is not needed.
- API is cheap but peak-hour pricing doubles; batch work off-peak.
- Self-hosting a 552B MoE needs a multi-GPU node; the API is far simpler.

## Gotchas
- Prices vary by time of day and cache hit/miss; model cost with both.
- Concurrency limits apply (Flash 2500, Pro 500 per docs).
- Check data-processing terms and jurisdiction before sending sensitive data.

## Related
- [[qwen]], [[meta-llama]] - other open-weight options.
- [[mixture-of-experts]] - architecture.
- [[reasoning-models]] - reasoning effort.
- [[model-licenses]] - MIT vs community licenses.

## References
- https://api-docs.deepseek.com/quick_start/pricing
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
