---
title: LLM cost and latency optimization
category: deployment
tags: [cost, latency, ttft, throughput, caching, batching, model-routing, tokens]
use_cases:
  - "cut the monthly LLM bill of a production chatbot without hurting quality"
  - "reduce time-to-first-token for an interactive assistant"
  - "decide whether to self-host a model or keep using a hosted API"
  - "estimate cost per request before launching a feature"
status: draft
last_verified: 2026-10-03
sources:
  - https://platform.claude.com/docs/en/cli-sdks-libraries/overview
  - https://docs.vllm.ai/en/latest/getting_started/quickstart/
---

# LLM cost and latency optimization

## Summary
LLM cost is driven by tokens (input and output priced separately, output usually higher) times model tier; latency is time-to-first-token (TTFT) plus tokens-per-second times output length. Most savings come from sending fewer tokens, caching, using a smaller model where it is good enough, and batching work that does not need an immediate answer. No prices are quoted here; read your vendor's current pricing page.

## Key concepts
- Cost per request = input_tokens x input_rate + output_tokens x output_rate (+ cached-token and tool-use rates where applicable).
- TTFT depends on prompt length, queueing and model size; decode speed depends on model size and hardware.
- Output tokens dominate latency: generation is sequential, so shorter answers are faster and cheaper.
- Prompt caching: providers discount repeated prompt prefixes; structure prompts with stable content first ([[prompt-caching-and-cost]]).
- Batch APIs: asynchronous processing at a lower rate for non-urgent jobs (check the vendor for current terms).
- Model routing: easy requests to a small model, hard ones to a large one.

## When to use / scenarios
- Support bot with a long fixed system prompt: caching the prefix; cap answer length.
- Nightly document classification: batch API or a small/self-hosted model.
- Real-time voice agent: TTFT and streaming dominate ([[voice-agents]]).
- RAG with big context: retrieve fewer, better chunks and rerank instead of stuffing context.

## Setup & code
Measure first: log tokens and timings on every call.
```python
import time, anthropic
client = anthropic.Anthropic()
t0 = time.perf_counter(); first = None; out = []
with client.messages.stream(model="<model-id>", max_tokens=300,
        messages=[{"role": "user", "content": "Explain TCP slow start."}]) as s:
    for text in s.text_stream:
        first = first or time.perf_counter() - t0
        out.append(text)
    u = s.get_final_message().usage
print(f"TTFT={first:.2f}s total={time.perf_counter()-t0:.2f}s in={u.input_tokens} out={u.output_tokens}")
```
Back-of-envelope budget (fill in rates from the pricing page):
```python
def cost(in_tok, out_tok, in_rate_per_m, out_rate_per_m):
    return in_tok / 1e6 * in_rate_per_m + out_tok / 1e6 * out_rate_per_m
# monthly = cost(avg_in, avg_out, ...) * requests_per_day * 30
```
Levers, roughly in order of effort:
1. Trim prompts and history (summarize old turns); set `max_tokens`.
2. Cache stable prefixes; add an application cache for repeated identical queries.
3. Route by difficulty; use the smallest model that passes your eval ([[llm-evaluation]]).
4. Batch offline work; stream online work.
5. Self-host with continuous batching ([[inference-servers-vllm]]) once utilization is high and steady.

## Choosing / trade-offs
- Hosted API (pay per token, no ops, elastic) vs self-hosted (fixed GPU cost, cheap at sustained high utilization, ops burden, privacy control). Break-even depends on utilization, so compute it with your numbers.
- Quality vs cost: validate each downgrade on your own eval set, not on public benchmarks.
- Latency vs throughput: bigger batches raise throughput and per-request latency.
- Reasoning/thinking models cost extra output tokens; use only where they pay off ([[reasoning-models]]).
- Semantic caching saves calls but can return a wrong cached answer for near-miss queries.

## Gotchas
- Output tokens, hidden reasoning tokens and tool-call round trips are easy to omit from estimates.
- Agent loops multiply calls; cap iterations and spend per task.
- Long conversations resend the entire history each turn; cost grows quadratically in total.
- Cache hits require byte-identical prefixes; a timestamp at the top of the prompt defeats caching.
- Retries on timeouts can double-bill.
- Averages hide tails: track p95/p99 latency and cost per user.

## Related
- [[prompt-caching-and-cost]] - provider caching mechanics.
- [[llm-observability]] - tracking tokens and latency.
- [[gpu-cloud-options]] - renting GPUs for self-hosting.
- [[model-selection]], [[small-language-models]] - cheaper models.
- [[quantization]] - cheaper self-hosting.

## References
- Check your provider's pricing and model pages (volatile).
- vLLM docs: https://docs.vllm.ai/en/latest/getting_started/quickstart/
