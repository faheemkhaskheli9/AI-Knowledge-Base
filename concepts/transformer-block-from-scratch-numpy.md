---
title: Transformer block from scratch in NumPy (layer norm, causal multi-head attention, GELU MLP, pre-LN residuals)
category: concepts
tags: [transformer, transformer-block, layer-norm, multi-head-attention, causal-mask, gelu, mlp, residual-stream, pre-ln, numpy, pytorch, from-scratch, deep-learning-basics]
use_cases:
  - "implement a full transformer block from scratch and match PyTorch"
  - "understand layer norm, multi-head attention, the MLP and residual connections together"
  - "check that a causal mask really stops information flowing from the future"
  - "count the parameters of a transformer layer"
status: draft
last_verified: 2026-10-04
sources:
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/2002.04745
---

# Transformer block from scratch in NumPy (layer norm, causal multi-head attention, GELU MLP, pre-LN residuals)

## Summary
A transformer is a stack of identical blocks. A pre-LN block (the GPT-2 style used by most modern LLMs) does two things, each wrapped in a residual connection: `x = x + Attention(LayerNorm(x))`, then `x = x + MLP(LayerNorm(x))`. Attention mixes information **between** positions; the MLP transforms each position **on its own**. About 40 lines of NumPy reproduce PyTorch's `nn.TransformerEncoderLayer(norm_first=True)` with a causal mask to `7e-16`, have exactly `12d² + 13d` parameters, and pass a causality test: changing the last token leaves every earlier output bit-for-bit unchanged.

## Key concepts
- **Layer norm.** Normalise each token's vector over its `d` features to mean 0 and variance 1, then scale and shift by learned `γ`, `β`. Unlike batch norm it does not depend on other examples in the batch ([[normalization-layers]], [[batchnorm-and-dropout-from-scratch-numpy]]).
- **Multi-head attention.** Project `x` to queries, keys and values (one `3d × d` matrix), split `d` into `h` heads of size `d/h`, compute `softmax(QKᵀ/√(d/h))V` per head, concatenate, and apply an output projection ([[self-attention-from-scratch-numpy]]).
- **Causal mask.** Set scores for future positions (`j > i`) to `−∞` before the softmax, so token `i` only attends to tokens `0…i`. That is what makes a block usable for next-token prediction.
- **MLP.** `W₂ · GELU(W₁ · x + b₁) + b₂` with a hidden width of `4d`, applied to each position independently. It holds about two thirds of the block's parameters.
- **Residual stream.** Each sub-layer **adds** to `x` rather than replacing it, so a deep stack is an identity path plus small updates. Gradients flow straight down that path ([[residual-and-skip-connections]]).
- **Pre-LN vs post-LN.** The original transformer applied layer norm after the residual add (post-LN), which needs learning-rate warm-up to train deep stacks. Pre-LN normalises the sub-layer input instead, keeps the residual path clean, and trains more stably (Xiong et al., 2020). It needs a final layer norm after the last block.
- **Parameter count.** Attention `4d² + 4d`, MLP `8d² + 5d`, two layer norms `4d`: `12d² + 13d` per block.

## When to use / scenarios
- Learning: how the pieces covered separately (attention, layer norm, residuals, GELU) fit into one layer, before reading nanoGPT or a model's `modeling_*.py`.
- Interviews: "write a transformer block", "pre-LN vs post-LN", "how many parameters in a layer", "how does the causal mask work".
- Debugging or porting a model: compare a hand-written block to a reference layer on the same weights, as below.
- Production: use PyTorch modules with fused attention (`F.scaled_dot_product_attention`) or a library implementation; a NumPy block is for understanding only ([[transformers-and-attention]], [[attention-variants-and-efficient-attention]]).
- Not for: training anything real. There is no backward pass here; use autograd ([[autograd-engine-from-scratch]]).

## Setup & code
`pip install numpy scipy torch`. Runs on CPU in a few seconds.

```python
import numpy as np
import torch
from scipy.special import erf


def layer_norm(x, gamma, beta, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)                      # biased variance, like PyTorch
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))           # exact GELU


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def mha(x, Wqkv, bqkv, Wo, bo, n_heads, causal=True):
    T, d = x.shape
    hd = d // n_heads
    q, k, v = np.split(x @ Wqkv.T + bqkv, 3, axis=-1)                # each (T, d)
    q, k, v = (m.reshape(T, n_heads, hd).transpose(1, 0, 2) for m in (q, k, v))  # (h, T, hd)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd)                  # (h, T, T)
    if causal:
        scores = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, scores)
    out = softmax(scores) @ v                                        # (h, T, hd)
    return out.transpose(1, 0, 2).reshape(T, d) @ Wo.T + bo         # concat heads, project


def block(x, p, n_heads):
    """Pre-LN transformer block: x + MHA(LN(x)), then x + MLP(LN(x))."""
    x = x + mha(layer_norm(x, *p["ln1"]), *p["attn"], n_heads)
    h = gelu(layer_norm(x, *p["ln2"]) @ p["W1"].T + p["b1"])
    return x + h @ p["W2"].T + p["b2"]


d, n_heads, T = 32, 4, 10
torch.manual_seed(0)
ref = torch.nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4 * d, dropout=0.0,
                                       activation="gelu", batch_first=True,
                                       norm_first=True).double().eval()
g = lambda t: t.detach().numpy()
p = {"ln1": (g(ref.norm1.weight), g(ref.norm1.bias)),
     "ln2": (g(ref.norm2.weight), g(ref.norm2.bias)),
     "attn": (g(ref.self_attn.in_proj_weight), g(ref.self_attn.in_proj_bias),
              g(ref.self_attn.out_proj.weight), g(ref.self_attn.out_proj.bias)),
     "W1": g(ref.linear1.weight), "b1": g(ref.linear1.bias),
     "W2": g(ref.linear2.weight), "b2": g(ref.linear2.bias)}
print("parameters:", sum(t.numel() for t in ref.parameters()), "= 12*d^2 + 13*d =", 12 * d * d + 13 * d)

rng = np.random.default_rng(0)
x = rng.normal(size=(T, d))
ours = block(x, p, n_heads)
mask = torch.nn.Transformer.generate_square_subsequent_mask(T, dtype=torch.float64)
theirs = ref(torch.tensor(x)[None], src_mask=mask, is_causal=True)[0].detach().numpy()
print(f"max |ours - torch| = {np.abs(ours - theirs).max():.1e}")

# Causality: changing the last token must not change any earlier output.
x2 = x.copy(); x2[-1] += 10
diff = np.abs(block(x2, p, n_heads) - ours).max(-1)
print(f"output change at positions 0..{T - 2}: {diff[:-1].max():.1e}, at position {T - 1}: {diff[-1]:.1f}")

# Pre-LN keeps a clean residual path: stacking 12 identical blocks, compare norms.
h = x
for _ in range(12):
    h = block(h, p, n_heads)
print(f"input norm per token {np.linalg.norm(x, axis=-1).mean():.1f}, "
      f"after 12 blocks {np.linalg.norm(h, axis=-1).mean():.1f}")
```

Output (numpy 2.5, scipy, torch 2.13):
```
parameters: 12704 = 12*d^2 + 13*d = 12704
max |ours - torch| = 6.7e-16
output change at positions 0..8: 0.0e+00, at position 9: 10.0
input norm per token 5.8, after 12 blocks 19.5
```

The block equals PyTorch's to float64 rounding, which confirms the head split, the scaling by `√(d/h)`, the masking and the exact (erf) GELU. Perturbing the last token changes only the last output, so no information leaks backwards from the future. Stacking the block 12 times grows the average token norm from 5.8 to 19.5, because every block adds to the residual stream and nothing in a pre-LN block normalises the stream itself. That is why GPT-style models end with a final layer norm before the output layer.

## Choosing / trade-offs
- **Pre-LN vs post-LN.** Pre-LN is the default for LLMs: stable without long warm-up. Post-LN can reach slightly better quality when it trains, and is what BERT and the original transformer used. Some newer models add normalisation on both sides of a sub-layer.
- **LayerNorm vs RMSNorm.** RMSNorm drops the mean subtraction and `β`, which is cheaper and works as well; Llama-family models use it ([[normalization-layers]]).
- **GELU vs SwiGLU.** Most recent LLMs replace the GELU MLP with a gated SwiGLU MLP, with the hidden width reduced to about `8d/3` to keep the parameter count similar ([[activation-functions]]).
- **Positional information.** This block has none: without positional encodings, attention is permutation-equivariant. Add learned or sinusoidal embeddings at the input, or rotate `q` and `k` with RoPE inside attention ([[positional-encodings]]).
- **Multi-head vs grouped-query attention.** Sharing key/value heads across query heads (GQA, MQA) shrinks the KV cache at inference with little quality loss ([[attention-variants-and-efficient-attention]]).

## Gotchas
- `x.var()` in NumPy is the biased (population) variance, which is what layer norm uses. `torch.var` defaults to the unbiased one; using it inside a hand-written layer norm gives a small mismatch.
- `nn.GELU()` and `activation="gelu"` are the exact erf version; `approximate="tanh"` is GPT-2's. Mixing them up gives differences around `1e-3`, enough to fail a port.
- Use `-inf` (or a very large negative number in float16) for masked scores, and mask **before** the softmax. A fully masked row (possible with padding masks) produces `nan`.
- Scale by `√(head_dim)`, not `√d`. With 4 heads of size 8 the wrong scale is 2× off and attention becomes too flat.
- PyTorch's `in_proj_weight` stacks `Wq, Wk, Wv` along dimension 0 and multiplies as `x @ Wᵀ`. Splitting the output along the wrong axis or forgetting the transpose silently mixes queries and keys.
- `nn.TransformerEncoderLayer` defaults to `batch_first=False` and `dropout=0.1`. Comparisons must set `dropout=0.0`, call `.eval()`, and use a consistent layout.
- In real models, keep layer norm and softmax in float32 even when the rest runs in bfloat16 or float16 ([[efficient-training-mixed-precision]]).

## Related
- [[self-attention-from-scratch-numpy]] - single-head attention and its backward pass in detail.
- [[transformers-and-attention]] - the architecture, encoder vs decoder, and how blocks are stacked.
- [[positional-encodings]] - the position information this block deliberately leaves out.
- [[normalization-layers]] - LayerNorm, RMSNorm and where they go.
- [[residual-and-skip-connections]] - why the residual stream makes deep stacks trainable.
- [[lstm-from-scratch-numpy]] - the recurrent architecture transformers replaced.

## References
- PyTorch `torch.nn.TransformerEncoderLayer`: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
- PyTorch `torch.nn.LayerNorm`: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- Vaswani et al., Attention Is All You Need (2017): https://arxiv.org/abs/1706.03762
- Xiong et al., On Layer Normalization in the Transformer Architecture (2020): https://arxiv.org/abs/2002.04745
