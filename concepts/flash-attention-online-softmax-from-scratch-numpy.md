---
title: FlashAttention tiling and online softmax from scratch in NumPy
category: concepts
tags: [flash-attention, online-softmax, tiling, attention, memory, numerical-stability, long-context, numpy, from-scratch]
use_cases:
  - "understand how FlashAttention computes exact attention without the N x N score matrix"
  - "compute a numerically safe softmax in one streaming pass"
  - "explain why attention memory is O(N) with FlashAttention and O(N^2) without it"
  - "answer 'how does FlashAttention work' in an interview"
status: stable
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/2205.14135
  - https://arxiv.org/abs/2307.08691
  - https://arxiv.org/abs/1805.02867
  - https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
---

# FlashAttention tiling and online softmax from scratch in NumPy

## Summary
Standard attention builds the full `N × N` score matrix `QKᵀ`, applies softmax row by row, then multiplies by `V`. For long sequences that matrix is the memory and bandwidth bottleneck. FlashAttention computes the **same exact output** one `B × B` tile at a time, using the *online softmax* trick: keep a running max and a running sum for each row, and rescale the partial result whenever the max grows. This file builds online softmax and tiled attention in NumPy. On `N = 1024`, `d = 64` with `128`-row tiles, the tiled result matches naive attention to `2e-16`, while the largest score buffer drops from 8.4 MB to 0.13 MB.

## Key concepts
- **Safe softmax.** `softmax(x) = exp(x − max x) / Σ exp(x − max x)`. Subtracting the max avoids overflow. Without it, `exp(1000)` is `inf` and the result is `nan`.
- **Online softmax.** Stream through `x` keeping `m` (max so far) and `l` (sum of `exp(x − m)` so far). When a new max `m'` arrives, the old sum is rescaled by `exp(m − m')`. One pass, no need to see the whole row first.
- **Tiling.** Split `Q` into row blocks and `K`, `V` into column blocks. For each `Q` block, loop over `K`/`V` blocks, compute a tile of scores, and fold it into a running output `acc` with the same rescaling as `l`. Divide by `l` once at the end.
- **Why it is faster on GPU.** The arithmetic is the same (slightly more, in fact). The win is memory traffic: tiles live in on-chip SRAM, and the `N × N` matrix is never written to or read from HBM. Attention is limited by memory bandwidth, not FLOPs.
- **Backward pass.** FlashAttention stores only `m` and `l` per row and recomputes the score tiles during backprop instead of saving them. Recomputing is cheaper than reading `N²` values back from memory.

## When to use / scenarios
- Learning: the algorithm behind `torch.nn.functional.scaled_dot_product_attention`, `flash_attn`, and the attention kernels in vLLM and TensorRT-LLM.
- Practice: you never write this yourself in production. Call SDPA in PyTorch (it picks a flash kernel when it can) or set `attn_implementation="flash_attention_2"` / `"sdpa"` in Hugging Face Transformers. Knowing the algorithm explains its constraints (supported dtypes, head dims, masks).
- Online softmax alone is useful anywhere you need softmax or log-sum-exp over a stream: large-vocabulary losses, merging partial results across GPUs (ring attention), split-K decoding.
- Not for: reducing the `O(N²)` compute. FlashAttention is exact attention. To change the complexity, see sparse, linear or sliding-window attention in [[attention-variants-and-efficient-attention]].

## Setup & code
NumPy only. Runs in about a second.

```python
import numpy as np

rng = np.random.default_rng(0)
N, d, B = 1024, 64, 128                     # sequence length, head dim, tile size
Q, K, V = (rng.normal(0, 1, (N, d)) for _ in range(3))


def naive_attention(Q, K, V):
    S = Q @ K.T / np.sqrt(Q.shape[1])        # N x N scores: the memory problem
    P = np.exp(S - S.max(axis=1, keepdims=True))
    return (P / P.sum(axis=1, keepdims=True)) @ V


def online_softmax(x):
    """One pass: running max m and running sum l, rescaled when m grows."""
    m, l = -np.inf, 0.0
    for v in x:
        m_new = max(m, v)
        l = l * np.exp(m - m_new) + np.exp(v - m_new)
        m = m_new
    return np.exp(x - m) / l


def flash_attention(Q, K, V, B):
    """Tiled attention: never builds the N x N matrix, only B x B tiles."""
    N, d = Q.shape
    O = np.zeros_like(Q)
    for i in range(0, N, B):
        q = Q[i:i + B]
        m = np.full(len(q), -np.inf)          # running row max
        l = np.zeros(len(q))                  # running row sum of exp
        acc = np.zeros((len(q), d))           # running unnormalised output
        for j in range(0, N, B):
            s = q @ K[j:j + B].T / np.sqrt(d)  # B x B tile of scores
            m_new = np.maximum(m, s.max(axis=1))
            p = np.exp(s - m_new[:, None])
            scale = np.exp(m - m_new)          # rescale what we had so far
            l = l * scale + p.sum(axis=1)
            acc = acc * scale[:, None] + p @ V[j:j + B]
            m = m_new
        O[i:i + B] = acc / l[:, None]          # normalise once at the end
    return O


x = rng.normal(0, 5, 10)
ref = np.exp(x - x.max()) / np.exp(x - x.max()).sum()
print(f"online softmax max abs diff {np.abs(online_softmax(x) - ref).max():.1e}")

O_ref = naive_attention(Q, K, V)
O_fa = flash_attention(Q, K, V, B)
print(f"flash vs naive max abs diff {np.abs(O_fa - O_ref).max():.1e}")
print(f"score memory  naive {N * N * 8 / 1e6:.1f} MB   tiled {B * B * 8 / 1e6:.3f} MB per tile")

big = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    print("unsafe softmax", np.exp(big) / np.exp(big).sum())
print("online softmax", online_softmax(big).round(4))
```

Output (Python 3.14, NumPy 2.5):
```
online softmax max abs diff 2.2e-16
flash vs naive max abs diff 2.2e-16
score memory  naive 8.4 MB   tiled 0.131 MB per tile
unsafe softmax [nan nan nan]
online softmax [0.09   0.2447 0.6652]
```

The tiled version is exact, not an approximation: the difference is floating-point rounding. The score buffer is `B²` instead of `N²`, so memory grows linearly with sequence length (the `O`, `m` and `l` arrays are `O(N)`). At `N = 128k` the naive score matrix for one head in fp16 would be 32 GB. In NumPy the tiled loop is slower than one big matmul. The speedup only appears in a fused GPU kernel where tiles stay in SRAM.

## Choosing / trade-offs
- **Tile size.** Bigger tiles mean fewer rescaling steps and better matmul efficiency, but they must fit in on-chip SRAM (about 100-250 KB per streaming multiprocessor). Kernels choose `B` from the head dim and GPU, which is why some head dims are unsupported.
- **FlashAttention 1 vs 2 vs 3.** v2 moved the outer loop to `Q` blocks (as here) for better parallelism and fewer non-matmul operations. v3 targets Hopper GPUs with asynchronous copies and FP8. Same math in all three.
- **Masks.** A causal mask lets the kernel skip tiles entirely above the diagonal, which roughly halves the work. Arbitrary dense masks block that and often force a slower path.
- **Decoding.** With one query token, there is little to tile over `Q`. Decode kernels (FlashDecoding, split-K) instead split the `K`/`V` sequence across thread blocks and merge partial `(m, l, acc)` results with the same rescaling rule.
- **Which to use.** PyTorch SDPA covers most cases with no extra install. The `flash-attn` package adds features such as variable-length batches without padding. Serving engines ship their own kernels, see [[inference-servers-vllm]].

## Gotchas
- Initialise the running max to `-inf`, not `0`. With `0` the first rescale is wrong whenever all scores are negative.
- A row that is fully masked has `l = 0` and divides by zero. Real kernels special-case it. Padding tokens in a batch are the usual source.
- Accumulate `m`, `l` and `acc` in fp32 even when `Q`, `K`, `V` are fp16 or bf16. Low-precision accumulators drift over long rows.
- FlashAttention saves memory for the score matrix only. The KV cache still grows linearly with context and dominates at long lengths, see [[kv-cache-from-scratch-numpy]].
- Benchmarks of "attention speedup" depend on sequence length. At a few hundred tokens the score matrix is small and the gain is modest.
- Getting attention weights back (`output_attentions=True`) forces the materialised path, since the fused kernel never stores them.

## Related
- [[self-attention-from-scratch-numpy]] - the naive attention this file reorganises.
- [[attention-variants-and-efficient-attention]] - sparse, linear and grouped-query attention, which change the cost instead of the memory layout.
- [[kv-cache-from-scratch-numpy]] - the other memory cost of long contexts.
- [[long-context]] - where tiled attention is required in practice.
- [[decoding-strategies-from-scratch-numpy]] - another place where softmax is computed over a large axis.

## References
- Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness": https://arxiv.org/abs/2205.14135
- Dao (2023), "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning": https://arxiv.org/abs/2307.08691
- Milakov and Gimelshein (2018), "Online normalizer calculation for softmax": https://arxiv.org/abs/1805.02867
- PyTorch `scaled_dot_product_attention`: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
