---
title: Attention variants and efficient attention (MHA, MQA, GQA, FlashAttention, sliding window)
category: concepts
tags: [attention, multi-head-attention, gqa, mqa, flash-attention, sdpa, sliding-window, kv-cache, transformer, pytorch]
use_cases:
  - "write a transformer attention layer that uses fast fused kernels"
  - "cut KV-cache memory so a model serves longer contexts or bigger batches"
  - "understand what GQA / MQA / sliding window mean on a model card"
  - "make attention fit in GPU memory for long sequences during training"
  - "add causal, padding or local masks without breaking the fast path"
status: draft
last_verified: 2026-10-04
sources:
  - https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
  - https://arxiv.org/abs/2205.14135
  - https://arxiv.org/abs/2305.13245
  - https://arxiv.org/abs/1911.02150
  - https://github.com/Dao-AILab/flash-attention
---

# Attention variants and efficient attention (MHA, MQA, GQA, FlashAttention, sliding window)

## Summary
Standard attention compares every token with every other token, so time and memory grow with the square of sequence length, and at inference the stored keys and values (the KV cache) often take more GPU memory than the weights. Two families of fixes exist. **Kernel-level** (FlashAttention, memory-efficient attention) computes exact attention without storing the n×n matrix. **Architecture-level** (multi-query and grouped-query attention, sliding-window and sparse attention) changes what is computed to shrink the KV cache or the number of comparisons. [[transformers-and-attention]] explains attention itself; this file is the menu of variants and how to use them from PyTorch.

## Key concepts
- **Multi-head attention (MHA).** H heads, each with its own Q, K and V projections. KV cache per token = 2 × layers × H × head_dim values.
- **Multi-query attention (MQA).** All query heads share **one** K/V head. KV cache shrinks H times; quality drops a little and training can be less stable.
- **Grouped-query attention (GQA).** G groups of query heads, each sharing one K/V head (1 < G < H). Most current open LLMs use it, with quality close to MHA and a KV cache H/G times smaller. Existing MHA checkpoints can be converted by mean-pooling K/V heads and briefly uptraining.
- **FlashAttention.** Exact attention computed in tiles that fit in on-chip SRAM, with an online softmax, so the n×n score matrix is never written to GPU memory. Memory is linear in length and it runs faster because attention is memory-bound. Same result as naive attention up to float rounding.
- **SDPA.** `torch.nn.functional.scaled_dot_product_attention` picks the fastest available backend (FlashAttention, memory-efficient, cuDNN, or plain math) for your hardware, dtype and mask. Use it instead of writing `softmax(QKᵀ/√d)V` by hand.
- **Sliding-window (local) attention.** Each token attends only to the last W tokens. Cost O(n·W). Stacking L layers still gives a receptive field of about L·W. Often mixed with some global-attention layers.
- **Sparse / linear attention.** Fixed sparsity patterns (strided, global tokens) or kernelized approximations with O(n) cost. Mostly superseded in LLMs by FlashAttention + GQA + sliding window, or by [[state-space-models]].
- **Multi-head latent attention (MLA).** Stores a low-rank compressed latent instead of full K/V and reconstructs them per head (used in DeepSeek-V2/V3). An alternative to GQA with an even smaller cache.

## When to use / scenarios
- Writing any transformer in PyTorch: call SDPA. It is the single biggest free speed-up.
- Serving LLMs at long context or high concurrency: GQA/MQA models and paged KV caches decide how many requests fit on a GPU ([[inference-servers-vllm]]).
- Training on long sequences (documents, genomics, audio frames): FlashAttention plus gradient checkpointing makes 8k-32k tokens fit ([[long-context]], [[efficient-training-mixed-precision]]).
- Streaming or very long inputs where far context matters less: sliding-window attention bounds memory per token.
- NOT worth changing the architecture of a small model on short inputs (≤512 tokens): attention is a small share of compute there; the MLP dominates.

## Setup & code
```bash
pip install "torch>=2.5"
```
SDPA vs naive attention, GQA, KV-cache sizes for MHA/GQA/MQA, a sliding-window mask, and timing:
```python
import math
import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)
B, T, H, D = 2, 1024, 8, 64


def naive_attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(-1) @ v


q, k, v = (torch.randn(B, H, T, D) for _ in range(3))
causal = torch.ones(T, T, dtype=torch.bool).tril()
ref = naive_attention(q, k, v, causal)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(f"SDPA vs naive causal max abs diff: {(out - ref).abs().max():.2e}")

# Grouped-query attention: 8 query heads share 2 key/value heads
H_kv = 2
k_g, v_g = torch.randn(B, H_kv, T, D), torch.randn(B, H_kv, T, D)
rep = H // H_kv
out_gqa = F.scaled_dot_product_attention(
    q, k_g.repeat_interleave(rep, dim=1), v_g.repeat_interleave(rep, dim=1), is_causal=True
)
print(f"GQA output shape: {tuple(out_gqa.shape)}")


def kv_cache_mb(layers, kv_heads, head_dim, seq, batch=1, bytes_per=2):
    return 2 * layers * kv_heads * head_dim * seq * batch * bytes_per / 2**20


for name, kvh in [("MHA (32 kv heads)", 32), ("GQA (8 kv heads)", 8), ("MQA (1 kv head)", 1)]:
    print(f"{name:18s} KV cache @ 32 layers, 8k tokens, fp16: {kv_cache_mb(32, kvh, 128, 8192):7.0f} MiB")

# Sliding-window (local) attention via a boolean mask
W = 128
idx = torch.arange(T)
window = (idx[None, :] <= idx[:, None]) & (idx[:, None] - idx[None, :] < W)
out_sw = F.scaled_dot_product_attention(q, k, v, attn_mask=window)
print(f"sliding window keeps {window.float().mean():.3f} of the score matrix")

# Speed: naive vs SDPA (CPU here; on GPU SDPA dispatches to Flash/mem-efficient kernels)
for fn, name in [(lambda: naive_attention(q, k, v, causal), "naive"),
                 (lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), "sdpa")]:
    fn()
    t = time.perf_counter()
    for _ in range(10):
        fn()
    print(f"{name:5s} {1000 * (time.perf_counter() - t) / 10:.1f} ms/call")
```
Output (PyTorch 2.13.0, CPU):
```
SDPA vs naive causal max abs diff: 7.15e-07
GQA output shape: (2, 8, 1024, 64)
MHA (32 kv heads)  KV cache @ 32 layers, 8k tokens, fp16:    4096 MiB
GQA (8 kv heads)   KV cache @ 32 layers, 8k tokens, fp16:    1024 MiB
MQA (1 kv head)    KV cache @ 32 layers, 8k tokens, fp16:     128 MiB
sliding window keeps 0.117 of the score matrix
naive 36.7 ms/call
sdpa  8.7 ms/call
```
SDPA matches naive attention to float precision and is ~4x faster even on CPU. For a 7B-class shape (32 layers, head_dim 128) one 8k-token sequence needs 4 GiB of KV cache with MHA, 1 GiB with 8-group GQA, 128 MiB with MQA. That ratio is how many more concurrent requests fit on the same GPU.

Since PyTorch 2.5, SDPA also accepts `enable_gqa=True` with fewer K/V heads than query heads, which avoids materializing the repeated tensors. For arbitrary masks (sliding window, document masking, ALiBi-style biases) without losing the fused kernel, use `torch.nn.attention.flex_attention`. Hugging Face models select the backend with `attn_implementation="sdpa"` or `"flash_attention_2"` (the latter needs the `flash-attn` package and a supported NVIDIA GPU).

## Choosing / trade-offs
- **MHA vs GQA vs MQA (designing or picking a model).** MHA: best quality, biggest cache. GQA with 4-8 K/V heads: the usual sweet spot. MQA: smallest cache, slight quality loss; good for latency-critical serving.
- **Full vs sliding window.** Full attention keeps all context and quality; sliding window caps memory but cannot directly look up a token beyond L·W. Hybrid models interleave local and global layers.
- **SDPA vs `flash-attn` package.** SDPA works everywhere and needs nothing extra. The `flash-attn` package often has newer kernels first (variable-length batches without padding, newer GPU architectures) but needs a matching CUDA build and GPU.
- **Kernel vs architecture fix.** FlashAttention speeds up training and prefill but does not shrink the KV cache. GQA/MQA/MLA and KV-cache quantization shrink the cache ([[quantization]]).

## Gotchas
- A custom `attn_mask` can force SDPA off the FlashAttention backend onto a slower one. Prefer `is_causal=True` when causal is all you need, and FlexAttention for structured masks.
- Mask conventions differ: in SDPA a **boolean** mask uses True = "may attend"; a **float** mask is **added** to the scores (use -inf to block). Many hand-written codebases use the opposite boolean convention.
- `is_causal=True` with q and k of different lengths (decoding with a KV cache) aligns the mask to the top-left corner, which is wrong for incremental decoding. Pass an explicit mask, or use the `torch.nn.attention.bias.causal_lower_right` helper.
- A row that is fully masked (e.g. all padding) gives NaN after softmax. Ensure every query can attend to at least one key, or zero those rows afterwards.
- FlashAttention kernels need fp16/bf16 on GPU; fp32 inputs fall back to slower paths. Check which backend ran with `torch.nn.attention.sdpa_kernel` to force one in tests.
- Converting an MHA checkpoint to GQA without uptraining degrades quality noticeably; the GQA paper uptrains on ~5% of the original pretraining compute.
- KV-cache math is per sequence: multiply by batch size and context length before promising capacity.

## Related
- [[transformers-and-attention]] - attention, heads and the KV cache from first principles.
- [[positional-encodings]] - RoPE/ALiBi interact with masks and window length.
- [[long-context]] - context extension, where efficient attention is required.
- [[inference-servers-vllm]] - paged KV cache and batching built on these variants.
- [[efficient-training-mixed-precision]] - bf16, checkpointing and fused kernels in training.
- [[state-space-models]] - linear-time alternatives to attention.
- [[quantization]] - shrinking weights and KV cache further.

## References
- PyTorch `scaled_dot_product_attention`: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch FlexAttention blog: https://pytorch.org/blog/flexattention/
- Dao et al., "FlashAttention", NeurIPS 2022: https://arxiv.org/abs/2205.14135
- Dao, "FlashAttention-2", 2023: https://arxiv.org/abs/2307.08691
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (MQA), 2019: https://arxiv.org/abs/1911.02150
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", 2023: https://arxiv.org/abs/2305.13245
- Beltagy et al., "Longformer" (sliding-window + global attention), 2020: https://arxiv.org/abs/2004.05150
- DeepSeek-AI, "DeepSeek-V2" (multi-head latent attention), 2024: https://arxiv.org/abs/2405.04434
- flash-attn package: https://github.com/Dao-AILab/flash-attention
