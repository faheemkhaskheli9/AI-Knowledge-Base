---
title: Tensor shapes, broadcasting and einsum
category: concepts
tags: [tensors, shapes, broadcasting, einsum, reshape, permute, view, contiguous, numpy, pytorch, deep-learning]
use_cases:
  - "understand the shape errors I get when writing a PyTorch or NumPy model"
  - "split a hidden dimension into attention heads and put them back"
  - "write a batched matrix product or pairwise distance without Python loops"
  - "find a silent broadcasting bug that makes the loss look fine but trains wrong"
status: draft
last_verified: 2026-10-04
sources:
  - https://pytorch.org/docs/stable/notes/broadcasting.html
  - https://pytorch.org/docs/stable/generated/torch.einsum.html
  - https://pytorch.org/docs/stable/tensor_view.html
  - https://numpy.org/doc/stable/user/basics.broadcasting.html
---

# Tensor shapes, broadcasting and einsum

## Summary
Deep-learning code is mostly bookkeeping on tensor shapes: which axis is the batch, which is time, which holds features or heads. Three tools cover nearly all of it: **broadcasting** (element-wise ops between tensors of different shapes), **reshape/permute** (moving data between layouts), and **einsum** (any product or reduction written as named axes). Most shape bugs raise an error. The dangerous ones broadcast silently into a wrong but valid shape, and the model trains on garbage.

## Key concepts
- **Name your axes.** Keep a convention and write it in comments: `(B, T, C)` for sequences, `(B, C, H, W)` for PyTorch images, `(B, H, T, D)` for attention heads. Most bugs come from mixing two conventions, e.g. `(B, H, W, C)` from a NumPy/PIL image fed into a `(B, C, H, W)` conv.
- **Broadcasting rule.** Shapes are aligned from the **right**. Two dims are compatible if they are equal or one of them is 1. A missing leading dim counts as 1. The size-1 dim is stretched without copying memory. `(32, 1, 64) - (32, 10, 64)` works. `(10,) + (10, 1)` gives `(10, 10)`.
- **`keepdim=True`.** Reductions (`mean`, `sum`, `max`) drop the reduced axis unless you keep it. Keeping it as size 1 lets the result broadcast back against the input (normalising, centering, softmax).
- **Adding axes.** `x[:, None]`, `x.unsqueeze(1)` and NumPy's `np.newaxis` all insert a size-1 axis so two tensors line up for an outer operation (pairwise differences, masks).
- **view vs reshape.** `view` reinterprets the same memory and requires a compatible (usually contiguous) layout. `reshape` does the same when it can and copies when it must. `permute`/`transpose` only change strides, so the result is non-contiguous and a following `view` raises. Use `reshape`, or `.contiguous().view(...)`.
- **Splitting and merging dims.** Reshape changes how a dim is grouped, never which values sit next to each other in memory. To turn `(B, T, H*D)` into heads, go to `(B, T, H, D)` and then **permute** to `(B, H, T, D)`. Reshaping straight to `(B, H, T, D)` gives a valid shape with scrambled data.
- **einsum.** Give every axis of every input a letter. Letters in the output are kept. Letters missing from the output are summed over. `"bhid,bhjd->bhij"` is attention scores, `"ij,jk->ik"` is matmul, `"ii->"` is the trace. It is readable and the backend picks an efficient contraction.

## When to use / scenarios
- Writing any custom layer: attention, pooling over a mask, per-channel normalisation, loss functions over batches.
- Porting a model between NumPy, PyTorch and JAX. All three share the broadcasting rules, so the same shape reasoning works in each.
- Replacing Python loops over rows with one vectorised op (pairwise distances, batched matmuls, gathering per-sample values).
- Debugging a loss that falls but a model that does not learn: check shapes of `pred` and `target` first.
- Not a substitute for library layers. If `nn.MultiheadAttention`, `F.scaled_dot_product_attention` or `torch.cdist` already does the job, use it ([[attention-variants-and-efficient-attention]]).

## Setup & code
```bash
pip install torch numpy
```
```python
import torch

x = torch.randn(32, 10, 64)            # (batch, seq, features)

# Broadcasting: trailing dims align; size-1 or missing dims stretch.
mean = x.mean(dim=1, keepdim=True)     # (32, 1, 64) - keepdim keeps it broadcastable
centered = x - mean                    # (32, 10, 64)

# view vs reshape vs permute
h = x.view(32, 10, 8, 8)               # split features into 8 heads x 8 dims (needs contiguous)
h = h.permute(0, 2, 1, 3)              # (32, 8, 10, 8) - now non-contiguous
flat = h.reshape(32, 8, 80)            # reshape copies if needed; .view would raise here
print(h.is_contiguous(), flat.shape)   # False torch.Size([32, 8, 80])

# einsum: name every axis, sum over the ones missing from the output
q = torch.randn(32, 8, 10, 8)          # (b, h, i, d)
k = torch.randn(32, 8, 10, 8)          # (b, h, j, d)
scores = torch.einsum("bhid,bhjd->bhij", q, k)
assert torch.allclose(scores, q @ k.transpose(-2, -1), atol=1e-5)

# The classic silent bug: (N,1) - (N,) broadcasts to (N,N)
y = torch.randn(100)
pred = torch.randn(100, 1)
print((pred - y).shape)                 # torch.Size([100, 100]) - wrong!
print((pred.squeeze(-1) - y).shape)     # torch.Size([100])

# None / unsqueeze to line up an outer operation
a, b = torch.randn(5, 3), torch.randn(7, 3)
d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1).sqrt()   # (5, 7)
assert torch.allclose(torch.cdist(a, b), d2, atol=1e-5)
```
Tested with torch 2.13 (CPU). The same code works in NumPy with `np.einsum`, `axis=` instead of `dim=` and `keepdims=` instead of `keepdim=`.

## Choosing / trade-offs
- **einsum vs `@`/`matmul`.** `@` is shortest for plain (batched) matrix products. einsum wins once axes need reordering or summing in the same step, and it documents the shapes. For three or more operands, `opt_einsum` (used automatically by `torch.einsum` when installed) picks the contraction order.
- **Broadcast vs `expand`/`repeat`.** Broadcasting and `expand` cost no memory. `repeat`/`tile` copy data. Only copy when a later in-place op or a kernel needs a real, contiguous tensor.
- **Explicit reshape vs einops.** `einops.rearrange(x, "b t (h d) -> b h t d", h=8)` says the same thing as reshape plus permute in one readable, checked line. It is worth adding to a codebase with many layout changes. Plain `reshape`/`permute` need no extra dependency.
- **Big outer ops.** `a[:, None] - b[None]` materialises an `(N, M, D)` tensor. For large N and M, use `torch.cdist`, chunking, or the `‖a‖² + ‖b‖² - 2a·b` expansion.

## Gotchas
- **Silent `(N, 1)` vs `(N,)` broadcast.** `MSELoss` on `(N, 1)` predictions and `(N,)` targets warns in PyTorch, but a hand-written `((pred - y) ** 2).mean()` gives a plausible number on an `(N, N)` matrix. Assert shapes in loss code.
- **Reshape without permute** produces correct shapes and wrong data (see "Splitting and merging dims"). Test a layout change by round-tripping it and comparing with the original.
- **Channels-last images.** PIL/OpenCV/NumPy arrays are `(H, W, C)`. PyTorch convs expect `(C, H, W)`. `x.permute(2, 0, 1)`, or `torchvision.transforms.ToTensor`, does the swap. A `reshape` does not.
- **Squeeze with no argument** removes every size-1 dim, including a batch of size 1 at inference. Always pass the dim: `x.squeeze(-1)`.
- **In-place ops on expanded tensors** fail or corrupt values, because many positions share one memory cell. Clone first.
- **Reducing over the wrong axis.** `softmax(dim=0)` over the batch instead of the classes still sums to 1 per column and trains badly. Name the axis you mean (`dim=-1` for classes in `(B, C)` logits).

## Related
- [[pytorch-basics]] - tensors, autograd and the training loop these ops live in.
- [[math-for-machine-learning]] - the linear algebra behind matmul, norms and outer products.
- [[transformers-and-attention]] - the layer where head splitting and einsum appear most.
- [[attention-variants-and-efficient-attention]] - fused kernels that replace hand-written attention einsums.
- [[debugging-neural-network-training]] - shape asserts and overfit-one-batch checks catch these bugs early.
- [[jax-and-flax]] - same broadcasting and einsum semantics, plus `vmap` to avoid manual batch axes.

## References
- PyTorch broadcasting semantics: https://pytorch.org/docs/stable/notes/broadcasting.html
- `torch.einsum`: https://pytorch.org/docs/stable/generated/torch.einsum.html
- Tensor views and contiguity: https://pytorch.org/docs/stable/tensor_view.html
- NumPy broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
- einops: https://einops.rocks/
