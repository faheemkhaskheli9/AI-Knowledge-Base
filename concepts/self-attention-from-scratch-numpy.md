---
title: Self-attention from scratch in NumPy (scaled dot-product, multi-head, causal mask, checked against PyTorch)
category: concepts
tags: [attention, self-attention, multi-head-attention, scaled-dot-product, causal-mask, softmax, transformer, numpy, pytorch, from-scratch, deep-learning-basics]
use_cases:
  - "implement scaled dot-product and multi-head self-attention from scratch"
  - "understand the shapes inside a transformer attention layer and how heads are split and merged"
  - "see why attention scores are divided by sqrt(d) and how a causal mask works"
  - "check a hand-written attention layer against torch.nn.MultiheadAttention"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1706.03762
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
---

# Self-attention from scratch in NumPy (scaled dot-product, multi-head, causal mask, checked against PyTorch)

## Summary
Self-attention lets every token in a sequence build its new representation as a weighted average of all tokens' value vectors, with weights from query-key dot products. A multi-head version is about 15 lines of NumPy: three projections, a reshape into heads, `softmax(QKᵀ/√d)V`, and an output projection. Copying the weights from `torch.nn.MultiheadAttention` shows the hand-written layer matches PyTorch to float32 precision, with and without a causal mask.

## Key concepts
- **Q, K, V.** Each token `x` is projected three ways: a query (what am I looking for), a key (what do I offer), a value (what I pass on). `Q = XW_qᵀ + b_q`, and likewise for `K` and `V`.
- **Scaled dot-product.** `Attention(Q, K, V) = softmax(QKᵀ / √d_h) V`. Row `i` of the softmax is a probability distribution over the tokens that token `i` attends to.
- **Why `√d_h`.** For random vectors with unit-variance entries, `q·k` has standard deviation `√d_h`. Without the scale, scores grow with the head size and the softmax saturates to near one-hot, which also kills its gradients.
- **Multi-head.** Split the embedding `E` into `H` heads of size `d_h = E/H`, attend in each head independently, concatenate, and mix with an output projection `W_o`. Same cost as one big head, but each head can learn a different pattern.
- **Causal mask.** For autoregressive models, token `i` must not see tokens `j > i`. Set those scores to `−∞` before the softmax so their weights are exactly zero.
- **No sense of order.** Attention is permutation-equivariant: shuffle the input tokens and the outputs shuffle the same way. Order comes only from positional encodings ([[positional-encodings]]).
- **Cost.** The `T × T` score matrix makes time and memory quadratic in sequence length, which is what efficient-attention methods attack ([[attention-variants-and-efficient-attention]]).

## When to use / scenarios
- Learning: the one layer that defines a transformer, with every shape visible ([[transformers-and-attention]]).
- Interviews: "implement attention", "why divide by √d", "what does the causal mask do", "how do heads work".
- Debugging a custom transformer: a NumPy reference to compare a fused or modified kernel against.
- In real models use `torch.nn.functional.scaled_dot_product_attention`, which dispatches to FlashAttention or memory-efficient kernels.

## Setup & code
`pip install numpy torch`. Runs in about a second on CPU.

```python
import numpy as np
import torch


def softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True)              # stability; -inf from the mask stays -inf
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def multi_head_attention(x, Wq, Wk, Wv, Wo, bq, bk, bv, bo, n_heads, causal=False):
    """x: (T, E). W*: (E, E), applied as x @ W.T + b like nn.Linear. Returns (T, E), weights."""
    T, E = x.shape
    dh = E // n_heads
    split = lambda m: m.reshape(T, n_heads, dh).transpose(1, 0, 2)    # (H, T, dh)
    q, k, v = (split(x @ W.T + b) for W, b in ((Wq, bq), (Wk, bk), (Wv, bv)))
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(dh)                  # (H, T, T)
    if causal:
        scores = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, scores)
    A = softmax(scores)                                              # rows sum to 1
    out = (A @ v).transpose(1, 0, 2).reshape(T, E)                   # concat heads
    return out @ Wo.T + bo, A


rng = np.random.default_rng(0)
T, E, H = 6, 16, 4
x = rng.standard_normal((T, E))

torch.manual_seed(0)
mha = torch.nn.MultiheadAttention(E, H, batch_first=True)
with torch.no_grad():
    mha.in_proj_bias.normal_()                      # default init is zero; test the bias path
Wq, Wk, Wv = mha.in_proj_weight.detach().numpy().reshape(3, E, E)
bq, bk, bv = mha.in_proj_bias.detach().numpy().reshape(3, E)
Wo, bo = mha.out_proj.weight.detach().numpy(), mha.out_proj.bias.detach().numpy()

for causal in (False, True):
    ours, A = multi_head_attention(x, Wq, Wk, Wv, Wo, bq, bk, bv, bo, H, causal)
    xt = torch.tensor(x, dtype=torch.float32)[None]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1) if causal else None
    ref, _ = mha(xt, xt, xt, attn_mask=mask)
    print(f"causal={causal!s:<5} max |ours - torch| = {np.abs(ours - ref[0].detach().numpy()).max():.1e}")
print("causal weights, head 0, row 2:", A[0, 2].round(3))

# Permutation equivariance: without positions, attention cannot see order.
perm = rng.permutation(T)
out, _ = multi_head_attention(x, Wq, Wk, Wv, Wo, bq, bk, bv, bo, H)
out_p, _ = multi_head_attention(x[perm], Wq, Wk, Wv, Wo, bq, bk, bv, bo, H)
print("permute input == permute output:", np.allclose(out[perm], out_p))

# Why divide by sqrt(d): unscaled dot products of random d-dim vectors have std ~ sqrt(d),
# so softmax saturates to near one-hot as d grows.
for d in (16, 64, 256, 1024):
    q, k = rng.standard_normal((512, d)), rng.standard_normal((512, d))
    for name, s in (("unscaled", q @ k.T), ("scaled", q @ k.T / np.sqrt(d))):
        print(f"d={d:<5} {name:<8} mean max attention weight = {softmax(s).max(1).mean():.3f}")
```

Output (numpy 2.5, torch 2.13 CPU):
```
causal=False max |ours - torch| = 2.2e-07
causal=True  max |ours - torch| = 1.6e-07
causal weights, head 0, row 2: [0.5   0.2   0.301 0.    0.    0.   ]
permute input == permute output: True
d=16    unscaled mean max attention weight = 0.447
d=16    scaled   mean max attention weight = 0.027
d=64    unscaled mean max attention weight = 0.716
d=64    scaled   mean max attention weight = 0.027
d=256   unscaled mean max attention weight = 0.875
d=256   scaled   mean max attention weight = 0.028
d=1024  unscaled mean max attention weight = 0.931
d=1024  scaled   mean max attention weight = 0.026
```

The NumPy layer matches PyTorch to about `2e-7`, which is float32 rounding (PyTorch runs in float32, NumPy here in float64). With the causal mask, token 2 spreads its weight over tokens 0 to 2 and gives exactly zero to the future. The last block shows the reason for the scale: over 512 keys, the largest unscaled attention weight averages 0.45 at `d = 16` and 0.93 at `d = 1024`, so the softmax is close to a hard argmax. With the `1/√d` scale it stays near 0.027 regardless of `d`.

## Choosing / trade-offs
- **Fused projection vs three matrices.** PyTorch stores `W_q, W_k, W_v` as one `(3E, E)` matrix and does one matmul. Same maths, better speed. Split it with `reshape(3, E, E)` as above when porting weights.
- **Multi-head vs grouped-query / multi-query attention.** Standard MHA gives each head its own K and V. GQA and MQA share K/V across heads to shrink the KV cache at inference time, at a small quality cost ([[attention-variants-and-efficient-attention]]).
- **Masking style.** A boolean mask (`True` = blocked, PyTorch's convention for `attn_mask`) or an additive float mask with `−inf`. Both work; be consistent about which value means "blocked".
- **Reference implementation vs fused kernel.** Write the plain version to understand and test. Ship `scaled_dot_product_attention`, which avoids materialising the `T × T` matrix.

## Gotchas
- PyTorch's boolean `attn_mask` uses `True` for "not allowed". `scaled_dot_product_attention` uses the opposite convention (`True` = take part in attention). Mixing them up silently produces a reversed mask.
- Subtract the row max before `exp`. With the max trick, the `−inf` entries become `exp(−inf) = 0` safely. A row that is fully masked gives `0/0 = NaN`, which happens with padding masks on all-padding rows.
- `nn.MultiheadAttention` defaults to `batch_first=False`, expecting `(T, B, E)`. Passing `(B, T, E)` without `batch_first=True` runs without error and mixes up batch and time.
- `reshape(T, H, dh)` must split the *last* axis into heads. Reshaping to `(H, T, dh)` directly scrambles tokens across heads.
- The scale uses the head size `d_h`, not the full embedding size `E`.
- Its default bias initialisation is zero, so a test with default weights does not check that your bias handling is right. Randomise the biases, as above.
- Attention weights are not explanations. High weight on a token does not prove the output depends on it ([[saliency-maps-and-neural-attribution]]).

## Related
- [[transformers-and-attention]] - the full transformer block built around this layer.
- [[attention-variants-and-efficient-attention]] - GQA, MQA, FlashAttention and linear attention.
- [[positional-encodings]] - how order gets into a permutation-equivariant layer.
- [[tensor-shapes-broadcasting-and-einsum]] - the reshape and transpose moves used here.
- [[rnn-from-scratch-numpy]] - the recurrent alternative attention replaced.
- [[neural-network-from-scratch-numpy]] - the from-scratch MLP and softmax this builds on.

## References
- Vaswani et al., "Attention Is All You Need": https://arxiv.org/abs/1706.03762
- PyTorch `nn.MultiheadAttention`: https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- PyTorch `scaled_dot_product_attention`: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
