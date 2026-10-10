---
title: LLM gateways and model routing
category: deployment
tags: [gateway, model-routing, fallbacks, load-balancing, litellm, openrouter, multi-provider, rate-limits]
use_cases:
  - "call Claude, GPT, Gemini and a self-hosted model through one OpenAI-compatible API"
  - "fail over to another provider when the primary model is down or rate-limited"
  - "give each team its own API key, budget and usage report in front of shared LLM accounts"
  - "send easy requests to a cheap model and hard ones to a strong model"
status: draft
last_verified: 2026-10-03
sources:
  - https://docs.litellm.ai/docs/routing
  - https://docs.litellm.ai/docs/proxy/quick_start
  - https://openrouter.ai/docs/guides/routing/model-fallbacks
---

# LLM gateways and model routing

## Summary
An LLM gateway sits between your apps and model providers and exposes one API (usually OpenAI-compatible) for many models. It handles retries, fallbacks across providers, load balancing across keys/regions, per-team keys and budgets, and central logging. Model routing decides *which* model serves a request: by availability (failover), by load, by cost, or by request difficulty. Use a library router in-process for one app, a self-hosted proxy for many apps, or a hosted gateway when you want no infrastructure.

## Key concepts
- **Alias vs deployment:** apps call a stable alias (`chat-default`); the gateway maps it to one or more real deployments (provider + model + key + region). Swapping models becomes a config change.
- **Retry vs fallback:** retry the same deployment on transient errors (429, 5xx, timeout); fall back to a *different* model/provider once retries are exhausted or on a non-transient error.
- **Load balancing:** spread one alias over several deployments (shuffle, weighted, least-busy, latency-based, rate-limit-aware) and cool down a deployment that keeps failing.
- **Difficulty routing:** a cheap classifier or rule picks small vs large model per request. Only worth it when an eval shows the small model passes on the easy slice ([[llm-evaluation]]).
- **Virtual keys and budgets:** the gateway issues its own keys per team/app, enforces spend and rate limits, and keeps real provider keys out of apps.
- **Three shapes:** in-process library (LiteLLM `Router`), self-hosted proxy (LiteLLM Proxy, or a generic API gateway with LLM plugins), hosted service (OpenRouter, cloud-vendor gateways).

## When to use / scenarios
- Production assistant that must stay up when one provider has an outage: alias with a cross-provider fallback chain.
- Company with several teams sharing provider accounts: self-hosted proxy with virtual keys, budgets and usage logs ([[llm-observability]]).
- Prototype comparing many models quickly: a hosted gateway with one key for all of them.
- High-volume classification where most inputs are easy: difficulty routing to a [[small-language-models]] tier.
- Mixing hosted APIs with a self-hosted [[inference-servers-vllm]] endpoint behind one URL.
- **When NOT to use:** one app, one provider, and you need provider-specific features (prompt caching controls, extended thinking, computer use, citations). Call the native SDK directly ([[api-sdk-setup]]); a translation layer can lag or drop those features.

## Setup & code
In-process router with fallback (`pip install litellm`). Model IDs are placeholders; take current ones from each provider's models page.
```python
import os
from litellm import Router

router = Router(
    model_list=[
        {"model_name": "chat-default",   # alias your app uses
         "litellm_params": {"model": "anthropic/<claude-model-id>",
                            "api_key": os.environ["ANTHROPIC_API_KEY"]}},
        {"model_name": "chat-fallback",
         "litellm_params": {"model": "openai/<gpt-model-id>",
                            "api_key": os.environ["OPENAI_API_KEY"]}},
    ],
    fallbacks=[{"chat-default": ["chat-fallback"]}],
    num_retries=2,
)

resp = router.completion(model="chat-default",
                         messages=[{"role": "user", "content": "Summarize RAG in one line."}])
print(resp.model, resp.choices[0].message.content)  # resp.model shows which deployment answered
```
Self-hosted proxy for many apps (`uv tool install 'litellm[proxy]'`), `config.yaml`:
```yaml
model_list:
  - model_name: chat-default
    litellm_params:
      model: anthropic/<claude-model-id>
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: local-small
    litellm_params:
      model: openai/<served-model-name>      # any OpenAI-compatible server, e.g. vLLM
      api_base: http://localhost:8000/v1
      api_key: none
```
```bash
litellm --config config.yaml   # serves an OpenAI-compatible API on port 4000
```
Any OpenAI SDK client then points at the proxy:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:4000", api_key="<proxy-or-virtual-key>")
client.chat.completions.create(model="chat-default",
                               messages=[{"role": "user", "content": "hi"}])
```
Hosted gateway (OpenRouter): OpenAI-compatible at `https://openrouter.ai/api/v1`; pass a `models` list for server-side fallback. You are billed for the model that actually answered, returned in the response's `model` field.
```python
import os
from openai import OpenAI
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
r = client.chat.completions.create(
    model="<primary-model-slug>",
    extra_body={"models": ["<primary-model-slug>", "<fallback-model-slug>"]},
    messages=[{"role": "user", "content": "hi"}])
print(r.model)
```
Minimal difficulty router, no gateway needed:
```python
def pick_model(prompt: str) -> str:
    # ponytail: length/keyword heuristic; replace with a trained classifier once you have labeled traffic
    hard = len(prompt) > 2000 or any(k in prompt.lower() for k in ("prove", "refactor", "legal"))
    return "chat-strong" if hard else "chat-small"
```

## Choosing / trade-offs
- **Library vs proxy vs hosted:** library = no extra hop, one app only; proxy = central keys/budgets/logs for many apps, but you run and secure another service; hosted = zero ops and many models on one key, but a third party sees your traffic and adds a margin or fee (check its pricing page).
- **Fallback to a different provider** buys availability but changes behavior: prompts tuned for one model, tool-call formats and structured output can degrade on another. Eval every model in the chain.
- **Difficulty routing** saves money only if the router is cheap and accurate; misroutes to the small model cost quality, and a router LLM call adds latency.
- **Weighted/latency/cost-based balancing** across deployments of the *same* model is low risk; across *different* models it is really fallback and needs the same eval care.

## Gotchas
- Fallbacks hide outages: alert on fallback rate, or you run on the backup model for weeks without noticing.
- Prompt caching is per provider and per exact prefix; shuffling one alias across providers or keys destroys cache hit rates ([[prompt-caching-and-cost]]).
- Retries plus fallbacks multiply cost and latency on a bad day; cap total attempts and set an overall timeout.
- Translation layers normalize to the OpenAI schema; provider-only parameters may need passthrough fields or be silently dropped. Test the features you rely on.
- Streaming, tool calls and usage/token counts are where gateways diverge most from native SDKs; verify them end to end.
- The gateway holds every provider key and sees every prompt: treat it as sensitive infrastructure (auth on the admin UI, TLS, log redaction) ([[ai-security-privacy-compliance]]).
- Data-residency and zero-retention guarantees are per provider; a fallback can route regulated data to a provider you did not approve.

## Related
- [[cost-and-latency]] - routing as one of several cost levers.
- [[llm-observability]] - logging and tracing the gateway's traffic.
- [[model-selection]] - picking the models that go in the chain.
- [[serving-with-fastapi]] - writing a thin custom gateway yourself.
- [[inference-servers-vllm]] - self-hosted backends behind the gateway.
- [[api-sdk-setup]] - native SDKs for provider-specific features.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/api-gateway-and-service-mesh.md - the general gateway pattern.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/rate-limiting.md - token bucket and per-tenant limits.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/resilience-patterns.md - retries, timeouts and circuit breakers behind fallbacks.

## References
- LiteLLM Router (load balancing, retries, fallbacks): https://docs.litellm.ai/docs/routing
- LiteLLM Proxy quick start: https://docs.litellm.ai/docs/proxy/quick_start
- OpenRouter model fallbacks: https://openrouter.ai/docs/guides/routing/model-fallbacks
