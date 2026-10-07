---
title: Long context
category: concepts
tags: [long-context, rope, kv-cache, context-window, lost-in-the-middle, rag]
use_cases:
  - "decide whether to stuff whole contracts or codebases into the prompt or use RAG"
  - "estimate GPU memory for serving a model with a 128k-token window"
  - "analyse long legal, financial or clinical documents in one request"
  - "reduce cost of repeated long prompts with caching and compaction"
  - "extend an open model's context window with RoPE scaling"
status: stable
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/2307.03172
  - https://arxiv.org/abs/2104.09864
  - https://arxiv.org/abs/2309.00071
  - https://arxiv.org/abs/2205.14135
---

# Long context

## Summary
The context window is the number of tokens (prompt plus output) a model can attend to at once. Modern models range from tens of thousands to over a million tokens, enabling whole-document analysis, but cost, latency and recall quality all degrade with length, so long context complements rather than replaces retrieval.

## Key concepts
- **Quadratic attention, linear KV cache**: compute grows ~n^2 for prefill; KV cache memory grows ~n per sequence (see [[transformers-and-attention]]).
- **Positional encoding limits**: models trained on short sequences fail beyond their trained length unless positions are rescaled. **RoPE** scaling methods (position interpolation, NTK-aware, YaRN) extend context with modest further training.
- **Efficient attention**: FlashAttention, sliding-window, GQA, sparse/linear variants, and ring/sequence parallelism for training.
- **Effective vs advertised context**: retrieval of a single fact is easy; multi-fact reasoning over the whole window is much harder. "Lost in the middle": information in the middle is used less reliably than at the start or end.
- **Prompt caching**: repeated long prefixes can be cached by providers to cut cost and latency (see [[prompt-caching-and-cost]]).
- **Compaction / summarisation / memory**: for agents running long, summarise or offload old context ([[agent-memory]]).

## When to use / scenarios
- Single-pass work over a bounded corpus: one contract, a 300-page report, a code repository slice, a long transcript.
- Cross-document comparison where chunking would lose relationships.
- Prefer RAG when: the corpus is far larger than the window, queries touch small parts, data changes often, or cost per query matters.
- Real-world: a law firm asks questions across one case file; a support team summarises a long ticket history; a bank screens a full loan file.

## Setup & code
Estimate KV-cache memory before choosing hardware (pure Python):

```python
def kv_gb(layers, kv_heads, head_dim, tokens, bytes_per=2, batch=1):
    return 2 * layers * kv_heads * head_dim * tokens * bytes_per * batch / 1e9

# illustrative 8B-class GQA config: 32 layers, 8 KV heads, head_dim 128
print(round(kv_gb(32, 8, 128, 128_000), 1), "GB per 128k-token sequence in fp16")
```
Order prompts to exploit the window: put long documents first and the question last, and ask the model to quote supporting passages before answering.

Open-model RoPE scaling lives in the model config (for example a `rope_scaling` entry in Transformers configs); the supported types and field names vary by model and library version, so check the model card and current Transformers docs.

## Choosing / trade-offs
- Long context (simple, no index, higher per-query cost) vs RAG (cheaper per query, needs retrieval quality) vs hybrid (retrieve then read big chunks).
- Longer window models cost more per token at high lengths and are slower to first token.
- Extending an open model yourself risks quality loss on short inputs; prefer checkpoints trained for long context.

## Gotchas
- A model accepting 1M tokens does not mean it reasons well over 1M tokens; test with your own needle and multi-hop evals ([[llm-evaluation]]).
- Costs scale with input tokens on every call unless cached.
- Irrelevant context actively hurts accuracy; do not fill the window just because you can.
- Output limit is usually much smaller than the input limit.
- Prompt-injection surface grows with untrusted content in the window ([[prompt-injection]]).

## Related
- [[transformers-and-attention]] - why long context is expensive.
- [[rag-basics]] - the main alternative.
- [[prompt-caching-and-cost]] - lowering repeated-prefix cost.
- [[agent-memory]] - managing context in long-running agents.
- [[quantization]] - KV-cache and weight quantization for long windows.

## References
- Liu et al., Lost in the Middle: https://arxiv.org/abs/2307.03172
- Su et al., RoFormer (RoPE): https://arxiv.org/abs/2104.09864
- Peng et al., YaRN: https://arxiv.org/abs/2309.00071
- Dao et al., FlashAttention: https://arxiv.org/abs/2205.14135
