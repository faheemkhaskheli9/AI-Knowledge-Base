---
title: ALiBi from scratch in NumPy (linear attention biases, per-head slopes, length extrapolation, KV cache)
category: concepts
tags: [alibi, positional-encoding, attention-bias, length-extrapolation, long-context, transformer, kv-cache, bloom, mpt, numpy, from-scratch]
use_cases:
  - "understand what ALiBi adds to attention scores and why it needs no position embeddings"
  - "see why a model with ALiBi keeps its attention focused when run on longer inputs than it was trained on"
  - "compute ALiBi slopes for any head count (including non-powers of 2) when porting a model such as BLOOM or MPT"
status: draft
last_verified: 2026-10-05
sources:
  - https://arxiv.org/abs/2108.12409
  - https://github.com/ofirpress/attention_with_linear_biases
---

# ALiBi from scratch in NumPy (linear attention biases, per-head slopes, length extrapolation, KV cache)

## Summary
ALiBi (Attention with Linear Biases, Press, Smith & Lewis 2021, "Train Short, Test Long") drops position embeddings completely. Instead it subtracts `m_h * (i - j)` from every attention score, where `i - j` is the distance from the query back to the key and `m_h` is a fixed slope per head. Heads with large slopes look only at the last few tokens and heads with small slopes look far back. The penalty depends only on distance, so nothing the model sees changes when the sequence gets longer. That is why ALiBi models degrade gracefully past their training length, where learned or sinusoidal embeddings fail. BLOOM and MPT use it. This file implements the slopes, the biased causal attention and an incremental KV-cache step in NumPy, then measures how attention spreads as the sequence grows, with and without the bias.

## Key concepts
- **The bias.** `score[i, j] = q_i . k_j / sqrt(d) - m_h * (i - j)` for `j <= i`. It is added before softmax, is not learned, and nothing touches the token embeddings or values.
- **Slopes.** For `n` heads (power of 2) the slopes are the geometric sequence `2^(-8/n), 2^(-16/n), ..., 2^(-8)`. For 8 heads: 1/2, 1/4, ..., 1/256. For other head counts, take the slopes for the nearest lower power of 2 and fill the rest with every other slope from the `2n` sequence (the reference implementation's rule).
- **Half-life.** A bias of `m * dist` multiplies attention weight by `exp(-m * dist)`, so each head's weight halves every `ln 2 / m` tokens: 1.4 tokens for slope 1/2, 177 tokens for 1/256. Content scores can override this, but the bias sets each head's default reach.
- **Length extrapolation.** Without a position bias, a query's softmax spreads over every key, so attention entropy grows like `ln T` and gets diluted on long inputs. With ALiBi, distant keys get a large negative bias, so each head's effective span stays the same at any length.
- **Relative and KV-cache friendly.** Only `i - j` matters, so shifting all positions changes nothing. A decoding step needs the distances from the new token to the cached keys, nothing else.

## When to use / scenarios
- Reading or porting a model card or checkpoint that uses ALiBi (BLOOM, MPT, some code and encoder models): you need the exact slope rule, or outputs are silently wrong.
- Training a small or medium decoder where you want reasonable behaviour on inputs somewhat longer than the training length without fine-tuning.
- Interview and teaching: the simplest relative position method, easier to explain than RoPE ([[rope-rotary-embeddings-from-scratch-numpy]]).
- Not the default for new large LLMs: most current open models use RoPE plus a context-extension method (see [[positional-encodings]], [[long-context]]).

## Setup & code
NumPy only, under a second.

```python
import numpy as np

rng = np.random.default_rng(0)


def alibi_slopes(n_heads):
    """Geometric slopes 2^(-8/n), 2^(-16/n), ...; non-power-of-2 head counts interleave from the 2n sequence."""
    def pow2(n):
        start = 2 ** (-8 / n)
        return [start ** (i + 1) for i in range(n)]
    if np.log2(n_heads).is_integer():
        return np.array(pow2(n_heads))
    c = 2 ** int(np.floor(np.log2(n_heads)))
    return np.array(pow2(c) + pow2(2 * c)[0::2][:n_heads - c])


def attention(q, k, v, slopes=None):
    """Causal attention for one sequence. q, k, v: (heads, T, d). ALiBi adds -m * (i - j) to each score."""
    h, T, d = q.shape
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d)                     # (h, T, T)
    dist = np.arange(T)[:, None] - np.arange(T)[None, :]                # i - j, >= 0 below the diagonal
    if slopes is not None:
        scores = scores - slopes[:, None, None] * dist
    scores = np.where(dist >= 0, scores, -np.inf)                       # causal mask
    w = np.exp(scores - scores.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return w @ v, w


def decode_step(q_t, k_cache, v_cache, slopes):
    """One incremental step with a KV cache: the new query only needs distances to cached keys."""
    t = k_cache.shape[1] - 1
    s = (q_t[:, None, :] @ k_cache.transpose(0, 2, 1))[:, 0] / np.sqrt(q_t.shape[-1])
    s = s - slopes[:, None] * (t - np.arange(t + 1))
    w = np.exp(s - s.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return (w[:, None, :] @ v_cache)[:, 0]


print("slopes, 8 heads :", np.round(alibi_slopes(8), 4))
print("slopes, 12 heads:", np.round(alibi_slopes(12), 4))
m = alibi_slopes(8)
print("half-life ln2/m (tokens):", np.round(np.log(2) / m, 1))

# KV-cache decoding gives the same row as the full matrix
h, d, T = 8, 32, 64
q, k, v = (rng.standard_normal((h, T, d)) for _ in range(3))
full, _ = attention(q, k, v, m)
step = decode_step(q[:, -1], k, v, m)
assert np.allclose(full[:, -1], step)
print("KV-cache step matches full attention:", np.abs(full[:, -1] - step).max() < 1e-12)


# Length behaviour of the last query row with random q/k (content scores have std ~1).
# No position bias: attention spreads over everything, entropy -> ln(T). ALiBi: each head keeps a fixed span.
def last_row_stats(T, slopes):
    q, k = rng.standard_normal((h, d)), rng.standard_normal((h, T, d))
    dist = (T - 1) - np.arange(T)
    s = np.einsum("hd,htd->ht", q, k) / np.sqrt(d) - slopes[:, None] * dist
    w = np.exp(s - s.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return -(w * np.log(w + 1e-30)).sum(-1), (w * dist).sum(-1)


print("\nT     | no bias: entropy (ln T)  mean dist | ALiBi entropy h0/h3/h7 | ALiBi mean dist h0/h3/h7")
for T in (512, 2048, 8192):
    e0, d0 = last_row_stats(T, np.zeros(h))
    ea, da = last_row_stats(T, m)
    print(f"{T:5d} | {e0.mean():8.2f} ({np.log(T):.2f}) {d0.mean():12.0f} | "
          f"{ea[0]:6.2f} {ea[3]:5.2f} {ea[7]:5.2f}      | {da[0]:6.1f} {da[3]:5.1f} {da[7]:6.1f}")
```

Output (Python 3.14, NumPy 2.5):
```
slopes, 8 heads : [0.5    0.25   0.125  0.0625 0.0312 0.0156 0.0078 0.0039]
slopes, 12 heads: [0.5    0.25   0.125  0.0625 0.0312 0.0156 0.0078 0.0039 0.7071 0.3536
 0.1768 0.0884]
half-life ln2/m (tokens): [  1.4   2.8   5.5  11.1  22.2  44.4  88.7 177.4]
KV-cache step matches full attention: True

T     | no bias: entropy (ln T)  mean dist | ALiBi entropy h0/h3/h7 | ALiBi mean dist h0/h3/h7
  512 |     5.74 (6.24)          263 |   1.66  3.42  5.42      |    2.0  17.7  176.5
 2048 |     7.06 (7.62)         1014 |   1.57  3.15  6.17      |    2.2  14.5  251.6
 8192 |     8.57 (9.01)         4083 |   1.19  3.36  5.99      |    2.7  14.2  257.3
```

How to read it:
- The 12-head slopes are the 8-head sequence plus 4 slopes in between (0.71, 0.35, 0.18, 0.09), taken from the 16-head sequence. A port that just plugs 12 into the power-of-2 formula gets different slopes and quietly breaks a pretrained checkpoint.
- With no position bias, the mean attended distance is about `T/2` (263, 1014, 4083) and entropy tracks `ln T`. Each extra token dilutes attention, which is one reason models fail beyond their training length.
- With ALiBi, the sharp heads keep the same span at every length (head 0 attends about 2 tokens back, head 3 about 15). The widest head (slope 1/256) is clipped by the sequence at T = 512 and settles near 250 tokens from 2048 on. Nothing grows with `T`, so a model trained at 512 sees familiar attention patterns at 8192.
- The KV-cache step matches the full computation exactly, so ALiBi adds no state to the cache, only a bias row per step.
- Random `q` and `k` stand in for a trained model. A real model learns content scores that override the bias for important tokens. The distance prior and how it scales are the same.

## Choosing / trade-offs
- **ALiBi vs RoPE.** ALiBi extrapolates better out of the box and is trivial to implement. RoPE has become the standard in large LLMs; it keeps more flexible long-range retrieval, and context-extension methods (position interpolation, NTK/YaRN scaling) plus long-context fine-tuning recover the extrapolation. The ALiBi slopes bias heads towards recency, which can hurt tasks that must retrieve one exact fact far back.
- **ALiBi vs learned absolute or sinusoidal embeddings.** Absolute embeddings have nothing for unseen positions (learned) or out-of-distribution inputs (sinusoidal), and perplexity blows up past the training length. ALiBi avoids that.
- **Extrapolation is not long-context ability.** ALiBi stops the model from breaking on longer inputs, but most of the extra context gets little attention. Do not expect a model trained at 2k to use evidence 30k tokens back.
- **Kernel support.** Fused kernels need to support an additive bias. FlashAttention and PyTorch SDPA support ALiBi slopes or an attention mask; check that your serving stack does before choosing it.

## Gotchas
- Sign: the bias is negative and grows with distance. A sign flip makes the model attend to the start of the sequence and still produce plausible-looking garbage.
- Causal only as written. Encoder (bidirectional) variants use `|i - j|` or separate slopes per direction; check the model's code.
- Implementations differ in how they broadcast: some add `m * j` (key position only) rather than `-m * (i - j)`. Within a causal softmax row these give the same weights, because `m * i` is constant across the row. With padding or packed sequences they can diverge; use positions that restart at each sequence.
- Left padding shifts positions of real tokens. Since only distances matter this is harmless for ALiBi itself, but the pad keys must still be masked.
- Mixed precision: the bias reaches `-m * T`, about -32 at T = 8192 for the widest head; fine in fp16/bf16, but adding the bias after a fp16 overflow in `q . k` does not help. Compute scores in fp32 inside the softmax.

## Related
- [[positional-encodings]] - absolute, sinusoidal, RoPE and ALiBi side by side.
- [[rope-rotary-embeddings-from-scratch-numpy]] - the rotary alternative most LLMs use.
- [[self-attention-from-scratch-numpy]] - the attention layer the bias is added to.
- [[kv-cache-from-scratch-numpy]] - the incremental decoding that ALiBi keeps cheap.
- [[long-context]] - context extension, and why extrapolating is not the same as using long context.
- [[attention-variants-and-efficient-attention]] - fused kernels and which biases they support.

## References
- Press, Smith & Lewis (2021), "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation": https://arxiv.org/abs/2108.12409
- Reference implementation, including the slope rule for non-power-of-2 heads: https://github.com/ofirpress/attention_with_linear_biases
