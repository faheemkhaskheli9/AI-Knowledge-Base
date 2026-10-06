---
title: Positional encodings (sinusoidal, learned, RoPE, ALiBi)
category: concepts
tags: [deep-learning, transformers, positional-encoding, sinusoidal, learned-positions, rope, rotary, alibi, relative-position, length-extrapolation, context-extension, pytorch]
use_cases:
  - "add position information to a transformer built from scratch"
  - "choose a positional scheme for a model that must handle longer inputs than it was trained on"
  - "understand why a model degrades past its training context length"
  - "extend the context window of a RoPE model by scaling positions"
  - "encode 2-D image patch or time-series positions in a transformer"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/2104.09864
  - https://arxiv.org/abs/2108.12409
  - https://arxiv.org/abs/2306.15595
  - https://arxiv.org/abs/2309.00071
---

# Positional encodings (sinusoidal, learned, RoPE, ALiBi)

## Summary
Self-attention treats its input as a set: shuffle the tokens and each output is shuffled the same way. Positional encodings add word-order information. Early transformers **added** a position vector to each token embedding (fixed sinusoids or a learned table). Most current LLMs instead inject position inside attention: **RoPE** rotates queries and keys so their dot product depends on relative distance, and **ALiBi** subtracts a distance-proportional penalty from attention scores. The choice decides how the model behaves on sequences longer than it was trained on.

## Key concepts
- **Why it is needed.** Without positions, "dog bites man" and "man bites dog" give the same set of token representations. Causal masking leaks some order in decoders, but explicit positions work far better.
- **Absolute, added to the input.**
  - *Sinusoidal* (original Transformer): `PE[p, 2i] = sin(p / 10000^(2i/d))`, `PE[p, 2i+1] = cos(...)`. No parameters, defined for any `p`, but models still generalise poorly past training length.
  - *Learned*: an `nn.Embedding(max_len, d)` table (BERT, GPT-2, ViT). Simple and flexible, but has no vectors at all beyond `max_len`.
- **Relative, inside attention.**
  - *RoPE (rotary)*: rotate each pair of query/key dimensions by angle `p * theta_i`. Then `q_m . k_n` depends only on `m - n`. Used by LLaMA, Mistral, Qwen, Gemma and most open LLMs. `theta` base (10,000 originally; much larger in long-context models) sets the wavelengths.
  - *ALiBi*: no embeddings; add `-slope_h * |m - n|` to attention logits, with a fixed slope per head. Extrapolates to longer inputs out of the box (BLOOM, MPT).
  - *T5 relative bias*: a learned scalar bias per bucketed distance, per head.
- **Context extension for RoPE.** Position interpolation (scale positions by `train_len / new_len`), NTK-aware scaling (raise the base), and YaRN (scale frequency bands differently, plus attention temperature) let a RoPE model run at 4-32x its training length after a short fine-tune.
- **Beyond 1-D.** Vision transformers use learned or 2-D sinusoidal/rotary encodings over patch rows and columns; time-series transformers often add calendar features (hour, weekday) as extra "positions".

## When to use / scenarios
- Training a small transformer from scratch for text, logs or event sequences: RoPE is the default; learned absolute is fine for fixed short lengths.
- Fine-tuning an existing model: keep whatever it was pretrained with; swapping schemes destroys the pretrained weights' meaning.
- Need inference on longer inputs than training (long documents, long event histories) without retraining: ALiBi, or RoPE plus interpolation/YaRN and a short fine-tune ([[long-context]]).
- Image patches at a different resolution than pretraining: interpolate the learned 2-D position table (timm does this for ViT).
- NOT needed for models that already have order built in (RNNs, CNNs) or for genuinely unordered sets (point clouds, set transformers).

## Setup & code
```bash
pip install torch
```
Sinusoidal table and RoPE, plus a check that RoPE scores depend only on relative distance:
```python
import math
import torch

def sinusoidal(max_len, d):
    pos = torch.arange(max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe                                   # add to token embeddings: x + pe[:T]

def rope(x, base=10000.0):
    """x: (..., T, d) queries or keys, d even. Rotates dimension pairs (i, i + d/2)."""
    T, d = x.shape[-2], x.shape[-1]
    inv_freq = base ** (-torch.arange(0, d, 2).float() / d)
    ang = torch.arange(T).float()[:, None] * inv_freq            # (T, d/2)
    cos, sin = ang.cos().repeat(1, 2), ang.sin().repeat(1, 2)    # (T, d)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * cos + torch.cat([-x2, x1], -1) * sin

print(sinusoidal(50, 16).shape)

torch.manual_seed(0)
q, k = torch.randn(1, 16), torch.randn(1, 16)
def score(m, n):                                # same q at position m, same k at position n
    Q = rope(q.expand(64, 16))[m]
    K = rope(k.expand(64, 16))[n]
    return (Q @ K).item()
print(round(score(10, 7), 4), round(score(40, 37), 4))   # equal: only m - n = 3 matters
```
ALiBi bias added to attention logits:
```python
def alibi_bias(n_heads, T):
    slopes = torch.tensor([2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)])
    dist = torch.arange(T)[None, :] - torch.arange(T)[:, None]   # n - m
    return slopes[:, None, None] * dist.clamp(max=0)             # (heads, T, T); causal mask handles n > m

# scores = q @ k.transpose(-1, -2) / sqrt(d) + alibi_bias(H, T)
print(alibi_bias(4, 5)[0])
```
In Hugging Face `transformers`, RoPE scaling is a config setting (`rope_scaling` with a `rope_type` such as `"linear"`, `"dynamic"` or `"yarn"`); check the model's config docs for the supported keys.

## Choosing / trade-offs
- **Learned absolute.** Simplest, good within a fixed length, useless beyond it. Fine for classifiers over short inputs and ViTs at a fixed resolution.
- **Sinusoidal.** No parameters and defined everywhere, but extrapolation is weak in practice. Mostly of historical and teaching value.
- **RoPE.** Strong quality, relative by construction, works with fast attention kernels and KV caches, and has a mature context-extension toolbox. Needs scaling plus fine-tuning to go far beyond training length.
- **ALiBi.** Best zero-shot extrapolation, trivial to implement, but the linear distance penalty biases attention toward recent tokens, which can hurt long-range retrieval.
- **NoPE (no positions) in causal decoders.** Works surprisingly well at small scale because the causal mask leaks order, but it is not the mainstream choice.

## Gotchas
- RoPE implementations pair dimensions differently: interleaved `(0,1), (2,3)...` (original paper, GPT-NeoX-style variants) vs split halves `(i, i + d/2)` (LLaMA in Hugging Face). Loading weights with the wrong convention gives garbage without any error.
- RoPE is applied to queries and keys only, never to values, and after the head split.
- With a KV cache, new tokens must use their absolute positions (`past_length + i`), not `0..n`; position-id bugs show up as degraded output only after the first generated token.
- Padding on the left (common for batched generation) shifts positions; pass explicit `position_ids` or an attention mask the model uses to derive them.
- A learned table cannot be extended by simply enlarging it; new rows are untrained. Interpolate the existing table instead and fine-tune.
- Claimed context lengths are not effective lengths; evaluate retrieval at the lengths you need ([[long-context]]).

## Related
- [[transformers-and-attention]] - where positions enter the attention computation.
- [[long-context]] - extending and evaluating long context windows.
- [[tokenization]] - positions count tokens, not characters or words.
- [[sequence-to-sequence-and-ctc]] - encoder-decoder models that use these encodings.
- [[pretraining-and-scaling-laws]] - training-length choices that positions must match.
- [[alibi-attention-bias-from-scratch-numpy]] - ALiBi built in NumPy, with per-head slopes and a length-extrapolation check.

## References
- Vaswani et al., Attention Is All You Need (sinusoidal): https://arxiv.org/abs/1706.03762
- Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding: https://arxiv.org/abs/2104.09864
- Press et al., Train Short, Test Long (ALiBi): https://arxiv.org/abs/2108.12409
- Chen et al., Extending Context Window via Positional Interpolation: https://arxiv.org/abs/2306.15595
- Peng et al., YaRN: Efficient Context Window Extension: https://arxiv.org/abs/2309.00071
