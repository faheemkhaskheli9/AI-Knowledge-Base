---
title: KV cache from scratch in NumPy (prefill vs decode, position ids, memory math)
category: concepts
tags: [kv-cache, transformer, inference, autoregressive-decoding, prefill, attention, gqa, mqa, llm-serving, numpy, from-scratch]
use_cases:
  - "implement a key/value cache for a decoder-only transformer and prove it gives identical logits"
  - "understand why LLM decoding is fast after the first token and what prefill vs decode means"
  - "estimate KV cache memory for a model, context length and batch size before choosing hardware"
  - "explain KV caching, GQA and MQA in an LLM inference interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/1911.02150
  - https://arxiv.org/abs/2305.13245
  - https://arxiv.org/abs/2309.06180
  - https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
---

# KV cache from scratch in NumPy (prefill vs decode, position ids, memory math)

## Summary
A decoder-only transformer generates one token at a time, and each new token attends to the keys and values of every earlier token. Those keys and values never change once computed, because causal attention stops earlier tokens from seeing later ones. The KV cache stores them per layer, so each decode step runs only the new token through the network. Below, a 2-layer NumPy decoder generates 48 tokens after a 16-token prompt in two ways: by recomputing the full sequence at every step, and with a cache. Both produce identical tokens, with logits that match to 2e-15. The cached run pushes 63 token rows through the weights instead of 1,896, about 30x fewer. Feeding the new token at position 0 instead of its real position changes the logits by up to 2.7, which is the classic cache bug. The cache is also what fills GPU memory: a Llama-3-8B-shaped model needs 128 KiB per token in fp16, or 1 GiB for one 8k-token sequence.

## Key concepts
- **Why caching is exact.** With a causal mask, the hidden state of token `t` depends only on tokens `0..t`. Its K and V at every layer are therefore fixed once computed and can be reused for all later steps.
- **Prefill.** The prompt is processed in one parallel forward pass that fills the cache and gives the logits for the first new token. It is compute-bound and sets time-to-first-token.
- **Decode.** Each later step feeds one token, appends its K and V to the cache, and attends over the whole cache. It is memory-bandwidth-bound, since weights and cache are read for a single row, and it sets the time per output token.
- **Position ids.** The new token's position is the cache length, not 0. With absolute or rotary encodings the position must be applied before K is stored. RoPE caches keys already rotated ([[rope-rotary-embeddings-from-scratch-numpy]]).
- **Memory.** `bytes = 2 (K and V) × layers × kv_heads × head_dim × tokens × bytes_per_value`, and it grows linearly with context length × batch.
- **GQA / MQA.** Grouped-query and multi-query attention share each K/V head across several query heads. The cache shrinks by `query_heads / kv_heads` with little quality loss.

## When to use / scenarios
- Learning: makes "prefill vs decode", time-to-first-token and tokens-per-second concrete, and shows why decode cost is linear in context length rather than quadratic.
- Interviews: "why is generation sequential", "what does the KV cache store", "how much memory for 32 users at 8k context", "what problem do GQA and PagedAttention solve".
- Practice: sizing GPUs for serving, choosing max batch and context in vLLM/TGI, debugging a custom generation loop that drifts from `model.generate`, prompt/prefix caching (reuse the cache of a shared system prompt), and speculative decoding (roll back rejected cache entries).
- Not for: encoder-only models (BERT), which have no autoregressive loop; or training, which processes whole sequences in parallel with no cache.

## Setup & code
NumPy only. Random weights are enough: the point is that cached and uncached decoding agree.

```python
import time
import numpy as np

rng = np.random.default_rng(0)
V, D, H, L = 100, 64, 4, 2          # vocab, model dim, heads, layers
Dh = D // H


def init(*shape):
    return rng.normal(0, shape[0] ** -0.5, shape)


emb = init(V, D)
layers = [dict(Wq=init(D, D), Wk=init(D, D), Wv=init(D, D), Wo=init(D, D),
               W1=init(D, 4 * D), W2=init(4 * D, D)) for _ in range(L)]
W_out = init(D, V)


def pos_enc(positions):
    i = np.arange(D // 2)
    ang = positions[:, None] / 10000 ** (2 * i / D)
    return np.concatenate([np.sin(ang), np.cos(ang)], -1)


def norm(x):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)


def heads(x):                      # (T, D) -> (H, T, Dh)
    return x.reshape(len(x), H, Dh).transpose(1, 0, 2)


def forward(tokens, start, cache=None):
    """Run `tokens` sitting at positions start.. ; append their K/V to `cache` if given."""
    T = len(tokens)
    x = emb[tokens] + pos_enc(np.arange(start, start + T))
    for li, p in enumerate(layers):
        h = norm(x)
        q, k, v = heads(h @ p["Wq"]), heads(h @ p["Wk"]), heads(h @ p["Wv"])
        if cache is not None:
            if li in cache:
                k = np.concatenate([cache[li][0], k], 1)
                v = np.concatenate([cache[li][1], v], 1)
            cache[li] = (k, v)
        S = k.shape[1]                                   # total keys visible
        s = q @ k.transpose(0, 2, 1) / np.sqrt(Dh)       # (H, T, S)
        # causal mask: query at absolute pos start+t sees keys 0..start+t
        mask = np.arange(S)[None, :] > (S - T + np.arange(T))[:, None]
        s = np.where(mask, -1e9, s)
        a = np.exp(s - s.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        x = x + (a @ v).transpose(1, 0, 2).reshape(T, D) @ p["Wo"]
        x = x + np.maximum(norm(x) @ p["W1"], 0) @ p["W2"]
    return norm(x) @ W_out                               # (T, V) logits


def generate_nocache(prompt, n):
    toks, all_logits = list(prompt), []
    for _ in range(n):
        logits = forward(np.array(toks), 0)[-1]          # recompute everything
        all_logits.append(logits)
        toks.append(int(logits.argmax()))
    return toks, np.array(all_logits)


def generate_cache(prompt, n, wrong_pos=False):
    cache = {}
    logits = forward(np.array(prompt), 0, cache)[-1]     # prefill
    toks, all_logits = list(prompt), []
    for _ in range(n):
        all_logits.append(logits)
        toks.append(int(logits.argmax()))
        pos = 0 if wrong_pos else len(toks) - 1          # new token's absolute position
        logits = forward(np.array([toks[-1]]), pos, cache)[-1]   # decode: one token
    return toks, np.array(all_logits)


prompt = rng.integers(0, V, 16).tolist()
n = 48
t0 = time.perf_counter(); a_toks, a_log = generate_nocache(prompt, n); t_a = time.perf_counter() - t0
t0 = time.perf_counter(); b_toks, b_log = generate_cache(prompt, n); t_b = time.perf_counter() - t0
_, c_log = generate_cache(prompt, n, wrong_pos=True)

print(f"prompt {len(prompt)} tokens, generate {n}")
print(f"same tokens with and without cache: {a_toks == b_toks}, max |logit diff| {np.abs(a_log - b_log).max():.1e}")
print(f"cache with position always 0: max |logit diff| {np.abs(a_log - c_log).max():.2f}")
print(f"time no-cache {t_a*1e3:.0f} ms, with cache {t_b*1e3:.0f} ms")

# token-rows pushed through the projections / MLP
P = len(prompt)
rows_nocache = sum(P + i for i in range(n))
rows_cache = P + (n - 1)
print(f"token rows processed: no-cache {rows_nocache}, cache {rows_cache} ({rows_nocache / rows_cache:.0f}x fewer)")


def kv_bytes(layers, kv_heads, head_dim, tokens, bytes_per=2):
    return 2 * layers * kv_heads * head_dim * tokens * bytes_per


print("\nKV cache size, fp16, Llama-3-8B shape (32 layers, head dim 128):")
for name, kvh in [("MHA, 32 KV heads", 32), ("GQA, 8 KV heads", 8), ("MQA, 1 KV head", 1)]:
    per_tok = kv_bytes(32, kvh, 128, 1)
    print(f"  {name:<17} {per_tok / 1024:>4.0f} KiB/token  "
          f"8k ctx {kv_bytes(32, kvh, 128, 8192) / 2**30:>5.2f} GiB  "
          f"32 seqs x 8k {kv_bytes(32, kvh, 128, 8192 * 32) / 2**30:>6.1f} GiB")
```

Output (Python 3.14, NumPy 2.5; timings depend on the machine):
```
prompt 16 tokens, generate 48
same tokens with and without cache: True, max |logit diff| 2.4e-15
cache with position always 0: max |logit diff| 2.74
time no-cache 39 ms, with cache 9 ms
token rows processed: no-cache 1896, cache 63 (30x fewer)

KV cache size, fp16, Llama-3-8B shape (32 layers, head dim 128):
  MHA, 32 KV heads   512 KiB/token  8k ctx  4.00 GiB  32 seqs x 8k  128.0 GiB
  GQA, 8 KV heads    128 KiB/token  8k ctx  1.00 GiB  32 seqs x 8k   32.0 GiB
  MQA, 1 KV head      16 KiB/token  8k ctx  0.12 GiB  32 seqs x 8k    4.0 GiB
```

The cache changes nothing about the maths: the tokens are identical, and the logits differ only at floating-point rounding level. What changes is the work. Without a cache, step `i` re-embeds and re-projects all `16 + i` tokens, so the total is quadratic in output length. With a cache, every token passes through the weights exactly once. Attention still reads the whole cache at every step, so per-token attention cost grows with context, but the expensive matrix multiplies do not. The wall-clock gap here is only 4x because NumPy overhead dominates at this toy size. On a GPU at real sizes the gap is the difference between usable and unusable. The wrong-position run is the most common bug in hand-written decode loops: the code runs and produces plausible-looking tokens, and the logits are simply wrong. The memory table explains why serving is about the cache. Llama 3 8B uses 8 KV heads (GQA) for 32 query heads, so 32 concurrent 8k-token sequences need 32 GiB of cache. That is twice the roughly 16 GB of fp16 weights, and full multi-head attention would need four times more.

## Choosing / trade-offs
- **Memory vs compute.** The cache trades memory for compute. On long contexts and large batches, KV memory, not weights, limits how many requests fit on a GPU.
- **GQA vs MHA vs MQA.** GQA (most current open models) keeps most of MHA's quality at a fraction of the cache. MQA is the extreme and costs more quality. Multi-head latent attention (DeepSeek) compresses K/V into a low-rank latent instead ([[attention-variants-and-efficient-attention]]).
- **Paged vs contiguous.** Pre-allocating a max-length buffer per request wastes memory on requests that end early. PagedAttention (vLLM) allocates the cache in fixed-size blocks like virtual memory and shares blocks between requests with a common prefix ([[inference-servers-vllm]]).
- **Cache quantization.** An int8/fp8 KV cache halves memory with a small accuracy cost, and is often a better trade than shrinking the batch ([[quantization]]).
- **Sliding window / eviction.** Keeping only the last W tokens (sliding-window attention) or dropping low-attention entries bounds memory on very long contexts, at the cost of forgetting ([[long-context]]).

## Gotchas
- Position ids must continue from the cache length. With left padding in a batch, every row has its own offset, so pass explicit position ids and an attention mask that hides the pad slots.
- Apply RoPE (or any position transform) to K before storing it. Rotating cached keys again on every step double-rotates them.
- Batch decoding with different prompt lengths needs per-row cache lengths or padding plus masks. A shared length breaks silently.
- Prefill and decode have different bottlenecks. Benchmark time-to-first-token and inter-token latency separately; one tokens/s number hides which one is slow.
- With speculative decoding or beam search, the cache must be rolled back or reordered (beam reindexing) together with the tokens. Forgetting to do so gives subtly wrong continuations.
- `concatenate` on every step, as in this demo, is O(n²) copying. Real implementations pre-allocate or use paged blocks.
- Prefix caching only helps when the cached prefix is byte-identical, including the system prompt and tool definitions. A timestamp at the top of the prompt defeats it ([[prompt-caching-and-cost]]).

## Related
- [[transformer-block-from-scratch-numpy]] - the block whose K/V this caches.
- [[self-attention-from-scratch-numpy]] - causal attention, the reason caching is exact.
- [[rope-rotary-embeddings-from-scratch-numpy]] - position handling with a cache.
- [[attention-variants-and-efficient-attention]] - GQA, MQA, MLA and FlashAttention.
- [[inference-servers-vllm]] - PagedAttention and continuous batching in production.
- [[cost-and-latency]] - time-to-first-token, tokens/s and GPU sizing.
- [[decoding-strategies-from-scratch-numpy]] - what to do with the logits each step produces.

## References
- Vaswani et al. (2017), "Attention Is All You Need": https://arxiv.org/abs/1706.03762
- Shazeer (2019), "Fast Transformer Decoding: One Write-Head is All You Need" (MQA): https://arxiv.org/abs/1911.02150
- Ainslie et al. (2023), "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints": https://arxiv.org/abs/2305.13245
- Kwon et al. (2023), "Efficient Memory Management for Large Language Model Serving with PagedAttention": https://arxiv.org/abs/2309.06180
- Meta Llama 3 8B config (layers, heads, KV heads): https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
