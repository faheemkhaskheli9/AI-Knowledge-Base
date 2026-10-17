---
title: LLM cost calculator (worked token math)
category: deployment
tags: [cost, pricing, tokens, prompt-caching, batch-api, self-hosting, break-even, budgeting, calculator]
use_cases:
  - "estimate the monthly API bill for a chatbot before launch"
  - "work out how much prompt caching saves on a long system prompt"
  - "compare real-time vs batch API cost for a nightly classification job"
  - "decide at what volume self-hosting an open model beats paying per token"
  - "compare cost per request across Claude and OpenAI tiers"
status: draft
last_verified: 2026-10-15
sources:
  - https://platform.claude.com/docs/en/about-claude/pricing
  - https://platform.claude.com/docs/en/build-with-claude/prompt-caching
  - https://platform.claude.com/docs/en/build-with-claude/batch-processing
  - https://developers.openai.com/api/docs/models
---

# LLM cost calculator (worked token math)

## Summary
A runnable, standard-library calculator for LLM API spend: cost per request from input/output tokens, the effect of prompt caching and batch discounts (which stack on Claude), multi-turn growth, and a self-hosting break-even that tells you the most you could pay per GPU-hour before an API becomes cheaper. Prices are copied from the model tables in [[anthropic-claude]] and [[openai-gpt]] (themselves from the vendors' pages) and are only as fresh as this file's `last_verified` date.

## Key concepts
- **Per-request cost** = uncached_input x input_rate + cache_write x write_rate + cache_read x read_rate + output x output_rate, all per million tokens (MTok).
- **Claude prompt-caching multipliers** (pricing page, 2026-10-15): 5-minute cache write = 1.25x base input, 1-hour write = 2x, cache read = 0.05x on Opus 5.5 and Sonnet 5.5, 0.025x on Fable 5.1, 0.1x on other models such as Haiku 4.5.
- **Claude Batch API**: 50% off input and output; caching multipliers stack with the batch discount and with data-residency pricing (pricing page).
- **US-only inference** (`inference_geo: "us"`) on Claude 4.6 and later: 1.1x on all token categories (pricing page).
- **Output and thinking tokens** are billed at the output rate; reasoning-heavy settings raise cost even when the visible answer is short ([[reasoning-models]], [[test-time-compute-scaling]]).
- **Multi-turn chat** resends history every turn, so total input grows roughly quadratically with turns unless cached or summarized.
- **Self-host break-even**: self-hosting wins when (GPU cost per hour) < (tokens served per hour x API price per token). Tokens served per hour = measured throughput x utilization x 3600. Throughput must be measured on your model, hardware and traffic; it is not quoted here.

Prices used (USD per MTok, input / output):

| Model ID | Input | Output | Cache read | Source |
|---|---|---|---|---|
| `claude-fable-5-1` | 10 | 50 | 0.25 | [[anthropic-claude]], pricing page |
| `claude-opus-5-5` | 4 | 20 | 0.20 | [[anthropic-claude]], pricing page |
| `claude-sonnet-5-5` | 2 | 10 | 0.10 | [[anthropic-claude]], pricing page |
| `claude-haiku-4-5-20251001` | 1 | 5 | 0.10 | [[anthropic-claude]], pricing page |
| `gpt-6-astra` | 10 | 50 | not listed here | [[openai-gpt]] |
| `gpt-6.1-sol` | 2 | 10 | not listed here | [[openai-gpt]] |
| `gpt-6-luna` | 0.1 | 0.5 | not listed here | [[openai-gpt]] |

OpenAI cached-input and batch rates differ from Anthropic's and are not in [[openai-gpt]]; the calculator therefore applies caching and batch only to Claude models. Add them from OpenAI's pricing page before comparing cached workloads.

## When to use / scenarios
- Before launch: estimate cost per request and per month from expected traffic and prompt sizes.
- Choosing a tier: the same workload on Haiku vs Sonnet vs Opus, then check quality with evals ([[model-selection]], [[llm-evaluation]]).
- Support bot with a long fixed system prompt or knowledge pack: quantify caching ([[prompt-caching-and-cost]]).
- Nightly bulk jobs (tagging, summarization, eval runs): batch vs real-time.
- Infra planning: whether a dedicated GPU running an open model ([[inference-servers-vllm]], [[gpu-cloud-options]]) would be cheaper at your volume.
- Not a substitute for metering: once live, compute cost from the `usage` fields on each response ([[llm-observability]]).

## Setup & code
Pure standard library; edit `PRICES` when the vendor pages change.

```python
PRICES = {  # USD per million tokens; cache_read = price per MTok of cache hits (None = not modelled)
    "claude-fable-5-1":          {"in": 10.0, "out": 50.0, "cache_read": 0.25, "vendor": "anthropic"},
    "claude-opus-5-5":           {"in": 4.0,  "out": 20.0, "cache_read": 0.20, "vendor": "anthropic"},
    "claude-sonnet-5-5":         {"in": 2.0,  "out": 10.0, "cache_read": 0.10, "vendor": "anthropic"},
    "claude-haiku-4-5-20251001": {"in": 1.0,  "out": 5.0,  "cache_read": 0.10, "vendor": "anthropic"},
    "gpt-6-astra":               {"in": 10.0, "out": 50.0, "cache_read": None, "vendor": "openai"},
    "gpt-6.1-sol":               {"in": 2.0,  "out": 10.0, "cache_read": None, "vendor": "openai"},
    "gpt-6-luna":                {"in": 0.1,  "out": 0.5,  "cache_read": None, "vendor": "openai"},
}
CACHE_WRITE_MULT = {"5m": 1.25, "1h": 2.0}   # x base input (Anthropic pricing page)
BATCH_MULT = 0.5                              # Anthropic Batch API: 50% off, stacks with caching
US_GEO_MULT = 1.1                             # Anthropic inference_geo="us" on 4.6+ models

def request_cost(model, input_tokens, output_tokens, cache_read=0, cache_write=0,
                 ttl="5m", batch=False, us_only=False):
    """input_tokens = uncached input; cache_read/cache_write are separate token counts."""
    p = PRICES[model]
    if (cache_read or cache_write or batch or us_only) and p["vendor"] != "anthropic":
        raise ValueError("caching/batch/geo multipliers only modelled for Anthropic; add the vendor's rates")
    cost = (input_tokens * p["in"] + output_tokens * p["out"]
            + cache_write * p["in"] * CACHE_WRITE_MULT[ttl]
            + cache_read * (p["cache_read"] or 0)) / 1e6
    if batch:
        cost *= BATCH_MULT
    if us_only:
        cost *= US_GEO_MULT
    return cost

# 1) Per request: 2,000 input + 400 output tokens
for m in PRICES:
    print(f"{m:28s} ${request_cost(m, 2000, 400):.5f}/request  "
          f"${request_cost(m, 2000, 400) * 50_000 * 30:,.0f}/month at 50k req/day")

# 2) Caching: 20k-token static knowledge pack + 300-token question, 400-token answer.
#    Assume one cache write per 5-minute window and 100 requests per window.
m, static, q, out, per_window = "claude-sonnet-5-5", 20_000, 300, 400, 100
no_cache = per_window * request_cost(m, static + q, out)
with_cache = (request_cost(m, q, out, cache_write=static)
              + (per_window - 1) * request_cost(m, q, out, cache_read=static))
print(f"\n{m} per 100 requests: no cache ${no_cache:.2f}, cached ${with_cache:.2f} "
      f"({1 - with_cache / no_cache:.0%} saved)")

# 3) Batch + cache for a nightly job of 200k documents (1,500 in, 100 out, 1,000-token shared instructions)
docs = 200_000
rt = docs * request_cost("claude-haiku-4-5-20251001", 1500 + 1000, 100)
bt = docs * request_cost("claude-haiku-4-5-20251001", 1500, 100, cache_read=1000, batch=True)
print(f"nightly job on Haiku 4.5: real-time ${rt:,.0f}, batch+cache ${bt:,.0f} "
      "(cache hits in batch are best-effort; treat as an optimistic bound)")

# 4) Multi-turn chat growth: each turn adds 150 user + 250 assistant tokens to the history
def conversation_cost(model, turns, system=1500, user=150, reply=250, cached_history=False):
    total, history = 0.0, 0
    for _ in range(turns):
        prefix = system + history
        if cached_history:
            total += request_cost(model, user, reply, cache_read=prefix)  # ignores per-turn write cost
        else:
            total += request_cost(model, prefix + user, reply)
        history += user + reply
    return total
for t in (5, 20, 50):
    print(f"{t:2d}-turn chat on Sonnet 5.5: ${conversation_cost('claude-sonnet-5-5', t):.3f} "
          f"uncached vs ~${conversation_cost('claude-sonnet-5-5', t, cached_history=True):.3f} cached")

# 5) Self-host break-even: the maximum GPU $/hour at which self-hosting matches the API price.
#    Throughputs are a sensitivity sweep, NOT benchmarks: measure your own (see inference-servers-vllm).
def breakeven_gpu_per_hour(api_model, tokens_per_s, utilization, out_share):
    p = PRICES[api_model]
    blended = (1 - out_share) * p["in"] + out_share * p["out"]   # $/MTok for your in/out mix
    return tokens_per_s * utilization * 3600 / 1e6 * blended
print("\nmax $/GPU-hour before the API is cheaper (out_share=0.2):")
for tps in (500, 2000, 8000):
    row = ", ".join(f"{m}: ${breakeven_gpu_per_hour(m, tps, 0.5, 0.2):.2f}"
                    for m in ("gpt-6-luna", "claude-haiku-4-5-20251001", "claude-sonnet-5-5"))
    print(f"  {tps:5d} tok/s at 50% util -> {row}")
```

How to read the break-even: if your measured throughput at realistic utilization gives a figure above what you would actually pay per GPU-hour (all-in: GPU, storage, egress, engineers on call), self-hosting is cheaper; if it is below, keep the API. Low-priced tiers move the break-even up a lot, and the open model must also match quality on your evals.

## Choosing / trade-offs
- **Cheaper tier vs fewer tokens**: dropping a tier often saves more than trimming prompts, but only if quality holds; measure both.
- **5-minute vs 1-hour cache**: 1-hour writes cost 2x, so they pay off only when the gap between requests is longer than 5 minutes but the prefix is reused at least twice within the hour (pricing page arithmetic).
- **Batch vs real-time**: 50% cheaper on Claude, but results can take hours; use for anything a user is not waiting on.
- **API vs self-host**: self-host has fixed cost and an ops burden; it wins only at sustained high utilization, and loses on spiky traffic unless you autoscale to zero.
- **Reasoning effort**: higher effort raises output tokens; sweep effort on your evals and price each level with this calculator.

## Gotchas
- Thinking tokens, tool-use system prompts, tool definitions and tool results all count as tokens; estimate from real `usage` logs, not prompt length in characters.
- Tokenizers differ between vendors and model generations; the same text is a different token count on each ([[tokenization]]).
- Cache hits need a byte-identical prefix and a minimum length; a timestamp at the top of the system prompt kills them ([[prompt-caching-and-cost]]).
- Retries, timeouts and agent loops multiply calls; budget per task, not per call ([[agents]]).
- Some prices are promotional or tied to context length or region; a dated price needs a `valid-until` marker in the source table.
- Averages hide heavy users; model the p95 user too.

## Related
- [[cost-and-latency]] - the levers this calculator quantifies.
- [[prompt-caching-and-cost]] - caching mechanics and usage fields.
- [[batch-inference-and-batch-apis]] - running the batch jobs priced here.
- [[anthropic-claude]], [[openai-gpt]] - price tables this file copies.
- [[gpu-cloud-options]], [[inference-servers-vllm]] - the self-host side of the break-even.
- [[llm-gateways-and-routing]] - per-tenant budgets and routing to cheaper models.
- [[test-time-compute-scaling]] - pricing extra samples and thinking.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/back-of-envelope-estimation.md - traffic and capacity estimates that feed the calculator.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/capacity-planning-and-autoscaling.md - utilization assumptions for self-hosting.

## References
- Anthropic pricing (base, cache, batch, data residency): https://platform.claude.com/docs/en/about-claude/pricing
- Anthropic prompt caching: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Anthropic batch processing: https://platform.claude.com/docs/en/build-with-claude/batch-processing
- OpenAI models and prices: https://developers.openai.com/api/docs/models
