---
title: RoPE (rotary position embeddings) from scratch in NumPy
category: concepts
tags: [rope, rotary-embeddings, positional-encoding, relative-position, attention, transformers, long-context, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement rotary position embeddings for q and k in NumPy and check their properties numerically"
  - "show that a RoPE attention score depends only on the relative offset between tokens"
  - "understand the base / theta frequency and why context-extension tricks (PI, NTK, YaRN) change it"
  - "explain RoPE vs sinusoidal vs learned vs ALiBi positions in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/2104.09864
  - https://arxiv.org/abs/2306.15595
  - https://arxiv.org/abs/2309.00071
  - https://huggingface.co/docs/transformers/main/en/model_doc/llama
---

# RoPE (rotary position embeddings) from scratch in NumPy

## Summary
Rotary position embedding (Su et al., RoFormer, 2021) is the position scheme in Llama, Mistral, Qwen, Gemma and most other open decoder LLMs. Instead of adding a position vector to the token embedding, it rotates the query and key vectors inside each attention layer. Each consecutive pair of dimensions is treated as a 2-D point and rotated by `position × θ_i`, with a different frequency `θ_i` per pair. Since rotating `q` by angle `mθ` and `k` by `nθ` and taking the dot product gives the same result as rotating only by `(m − n)θ`, the attention score depends on the relative offset, not on absolute positions, while the code stays a cheap element-wise operation. This file implements RoPE in 15 lines and checks four properties numerically: norms are unchanged, `q·k` at offset 3 is identical at positions 5, 105 and 1005, similarity decays with distance (1.00 at 0, 0.44 at 64, 0.26 at 1024), and a whole attention matrix is unchanged when every position shifts by 50.

## Key concepts
- **Pairs as complex numbers.** Split a `d`-dim vector into `d/2` pairs `(x_{2i}, x_{2i+1})`. Rotating a pair by angle `φ` is multiplying the complex number `x_{2i} + i·x_{2i+1}` by `e^{iφ}`.
- **Frequencies.** Pair `i` rotates at `θ_i = base^(−2i/d)`, `base = 10000` in the paper. Early pairs spin fast (one radian per token) and resolve nearby order; late pairs spin very slowly and carry long-range position.
- **Relative by construction.** `⟨R(mθ)q, R(nθ)k⟩ = ⟨q, R((n − m)θ)k⟩` because rotations are orthogonal and compose by adding angles. No relative-position bias table is needed.
- **Only q and k.** Values are not rotated; position only affects which tokens attend to which, not what is copied.
- **Decay with distance.** For a fixed vector, the sum of `cos((m − n)θ_i)` terms over many frequencies averages toward zero as the offset grows, giving a mild built-in preference for nearby tokens.
- **Interleaved vs split halves.** The paper pairs adjacent dimensions (`0::2`, `1::2`). GPT-NeoX and Hugging Face Llama pair dimension `j` with `j + d/2` (`rotate_half`). Both are valid; the checkpoint fixes which one you must use.

## When to use / scenarios
- Learning: the position encoding inside every modern open LLM; needed to read `apply_rotary_pos_emb` in Hugging Face, vLLM or llama.cpp.
- Interviews: "how does RoPE encode relative position", "why rotate only q and k", "what does the rope base / theta do", "how do PI, NTK-aware scaling and YaRN extend context".
- Debugging: wrong pairing convention, wrong position ids with left padding or a KV cache, or a mismatched `rope_theta` / `rope_scaling` config all produce a model that loads and then emits garbage.
- Not for: choosing an encoding for a new model from first principles. See [[positional-encodings]] for the comparison with sinusoidal, learned and ALiBi, and [[long-context]] for context-extension in practice.

## Setup & code
`pip install numpy`. Runs in about a second.

```python
import numpy as np


def rope(x, pos, base=10000.0):
    """Rotate consecutive pairs (x0,x1), (x2,x3), ... of x[..., d] by angle pos * base^(-2i/d)."""
    d = x.shape[-1]
    inv_freq = base ** (-np.arange(0, d, 2) / d)            # (d/2,)
    ang = np.asarray(pos)[..., None] * inv_freq             # (..., d/2)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


rng = np.random.default_rng(0)
d = 64
q, k = rng.normal(size=d), rng.normal(size=d)

# 1) Rotation keeps the norm.
print("norm before/after:", round(np.linalg.norm(q), 4), round(np.linalg.norm(rope(q, 37)), 4))

# 2) The score depends only on the offset m - n, not on absolute positions.
pairs = [(5, 2), (105, 102), (1005, 1002)]
print("q.k at offset 3:", [round(float(rope(q, m) @ rope(k, n)), 4) for m, n in pairs])

# 3) Long-range decay: the average score of a vector with itself shrinks as distance grows.
keys = rng.normal(size=(2000, d))
for dist in [0, 1, 4, 16, 64, 256, 1024]:
    s = np.mean([rope(v, dist) @ rope(v, 0) for v in keys]) / d
    print(f"self-similarity at distance {dist:5d}: {s:+.3f}")

# 4) Batched use inside attention: rotate q and k, never v.
T = 6
Q, K = rng.normal(size=(T, d)), rng.normal(size=(T, d))
pos = np.arange(T)
S1 = rope(Q, pos) @ rope(K, pos).T
S2 = rope(Q, pos + 50) @ rope(K, pos + 50).T          # same sequence shifted by 50
print("scores unchanged when the whole sequence shifts:", np.allclose(S1, S2))
```

Output (numpy 2.5):
```
norm before/after: 7.3153 7.3153
q.k at offset 3: [-8.4462, -8.4462, -8.4462]
self-similarity at distance     0: +1.003
self-similarity at distance     1: +0.969
self-similarity at distance     4: +0.752
self-similarity at distance    16: +0.606
self-similarity at distance    64: +0.436
self-similarity at distance   256: +0.356
self-similarity at distance  1024: +0.262
scores unchanged when the whole sequence shifts: True
```

The norm is untouched, so RoPE never changes a vector's magnitude, only its direction. The `q·k` score for an offset of 3 is the same to four decimals whether the tokens sit at 5/2, 105/102 or 1005/1002: absolute position has dropped out. The self-similarity column is `mean(cos((m − n)θ_i))` over the 32 frequencies. It falls quickly over the first few tokens (the fast pairs decorrelate) and then slowly (the slow pairs still agree), so distant tokens are not cut off, only down-weighted. Shifting a whole 6-token sequence by 50 leaves every attention score unchanged, which is why a KV cache can store keys already rotated at their absolute position.

## Choosing / trade-offs
- **RoPE vs learned absolute.** Learned embeddings do not extrapolate past the trained length at all. RoPE does not extrapolate well either (unseen angle combinations appear past the trained length), but it can be stretched cheaply.
- **RoPE vs ALiBi.** ALiBi adds a fixed linear distance penalty to the scores and extrapolates more gracefully without tuning, but most strong open models chose RoPE plus context-extension fine-tuning.
- **Context extension.** Position interpolation (PI) divides positions by the extension factor. NTK-aware scaling raises the base instead, which stretches the slow frequencies more than the fast ones. YaRN mixes both per frequency band and adds an attention temperature. All need the model's config (`rope_scaling`) to match at inference.
- **Base / theta.** Larger bases (Llama 3 uses 500000) slow the lowest frequencies so long contexts do not wrap around. Changing it after pre-training needs fine-tuning.
- **Partial rotary.** Some models (GPT-NeoX, Phi) rotate only a fraction of each head's dimensions and leave the rest position-free.

## Gotchas
- Pairing convention must match the checkpoint: interleaved (`0::2`/`1::2`) and split-half (`rotate_half`) layouts are both "RoPE" but are not interchangeable. Converting weights between them requires permuting the q/k projection rows.
- Position ids, not array indices. With left padding or a KV cache the new token's position is the cached length (minus padding), not 0. A wrong position id silently degrades output.
- Apply RoPE after splitting into heads and per head dimension `d_head`, not over the full model width.
- Compute angles in float32 even in bf16 inference: `pos × θ` at positions in the hundreds of thousands loses precision in half precision. Implementations cache `cos`/`sin` tables in float32.
- RoPE goes on q and k only. Rotating v, or adding RoPE to the input embeddings like a sinusoidal encoding, is a common porting bug.
- The decay with distance is weak and depends on the vectors; it is not a substitute for a real locality bias.

## Related
- [[positional-encodings]] - sinusoidal, learned, relative, ALiBi and RoPE compared.
- [[self-attention-from-scratch-numpy]] - where the rotated q and k are used.
- [[transformer-block-from-scratch-numpy]] - a full block you can drop `rope` into.
- [[long-context]] - context-extension methods and their costs in practice.
- [[attention-variants-and-efficient-attention]] - GQA, sliding windows and FlashAttention, which combine with RoPE.

## References
- Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding": https://arxiv.org/abs/2104.09864
- Chen et al. (2023), "Extending Context Window of Large Language Models via Positional Interpolation": https://arxiv.org/abs/2306.15595
- Peng et al. (2023), "YaRN: Efficient Context Window Extension of Large Language Models": https://arxiv.org/abs/2309.00071
- Hugging Face Transformers, Llama model docs (`rope_theta`, `rope_scaling`): https://huggingface.co/docs/transformers/main/en/model_doc/llama
