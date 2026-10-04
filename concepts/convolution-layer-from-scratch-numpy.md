---
title: Convolution layer from scratch (im2col forward and backward in NumPy, checked against PyTorch)
category: concepts
tags: [cnn, convolution, conv2d, im2col, col2im, backpropagation, stride, padding, numpy, pytorch, from-scratch, deep-learning-basics]
use_cases:
  - "implement a 2D convolution layer forward and backward pass from scratch"
  - "understand how frameworks turn convolution into one matrix multiply (im2col)"
  - "derive the gradients of conv2d with respect to input, weights and bias"
  - "verify a hand-written layer against torch.nn.functional.conv2d autograd"
status: draft
last_verified: 2026-10-04
sources:
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
  - https://cs231n.github.io/convolutional-networks/
---

# Convolution layer from scratch (im2col forward and backward in NumPy, checked against PyTorch)

## Summary
A 2D convolution layer slides `F` small filters over a `C`-channel image and takes a dot product at every position. The im2col trick copies every receptive field into one column of a matrix, so the whole layer becomes a single matrix multiply, and its backward pass becomes two more matrix multiplies plus the adjoint scatter (col2im). About 50 lines of NumPy, with stride and padding, match `torch.nn.functional.conv2d` and its autograd gradients to 1e-15. This is the CNN counterpart of [[neural-network-from-scratch-numpy]].

## Key concepts
- **Shapes.** Input `x: (N, C, H, W)`, weights `w: (F, C, kh, kw)`, bias `b: (F,)`, output `(N, F, OH, OW)` with `OH = (H + 2p − kh) // s + 1` ([[convolution-arithmetic-and-receptive-field]]).
- **It is cross-correlation.** Deep-learning "convolution" does not flip the kernel. Since the kernel is learned, the flip does not matter, but it matters when comparing with `scipy.signal.convolve`.
- **im2col.** Build `cols: (N, C·kh·kw, OH·OW)`: each column is one flattened receptive field. Loop over the `kh·kw` kernel offsets (9 for 3×3), not over output pixels, so each step is one strided slice of the whole batch.
- **Forward = one matmul.** `out = w.reshape(F, C·kh·kw) @ cols + b`, then reshape to `(N, F, OH, OW)`. This is how fast CPU/GPU libraries (GEMM-based convolution) work, with memory traded for speed.
- **Backward.** With upstream gradient `d: (N, F, OH·OW)`:
  - `db = d.sum over N and positions`, because the bias is added at every position.
  - `dw = Σₙ d @ colsᵀ`: the same "input × gradient" outer product as a dense layer, summed over positions (weight sharing).
  - `dcols = Wᵀ @ d`, then **col2im** scatters each column back to its pixels and *adds* overlapping contributions, giving `dx`.
- **col2im is the adjoint of im2col.** A pixel inside several receptive fields received copies in the forward pass, so it gets the sum of their gradients in the backward pass. Writing `=` instead of `+=` here is the classic bug.
- **Weight sharing and locality.** The same `F·C·kh·kw` weights are used at every position. That is why a conv layer has far fewer parameters than a dense layer on the same image, and why it is translation-equivariant ([[cnn-and-rnn-architectures]]).

## When to use / scenarios
- Learning: makes stride, padding, channels and the conv backward pass concrete before using `nn.Conv2d` ([[backpropagation-and-autograd]]).
- Interviews: output-shape arithmetic, parameter counts, and "why is conv backward a convolution too".
- Custom ops: a starting point for a layer the framework does not provide (a masked or deformable variant), checked against a framework reference the same way.
- Production: do NOT use this. `torch.nn.Conv2d` dispatches to cuDNN / oneDNN kernels (Winograd, FFT, implicit GEMM) that are orders of magnitude faster and use far less memory.

## Setup & code
`pip install numpy torch`. Runs on CPU in about a second.

```python
import numpy as np
import torch
import torch.nn.functional as F


def im2col(x, kh, kw, stride, pad):
    """(N, C, H, W) -> (N, C*kh*kw, OH*OW): every receptive field as a column."""
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    N, C, H, W = x.shape
    oh, ow = (H - kh) // stride + 1, (W - kw) // stride + 1
    cols = np.empty((N, C, kh, kw, oh, ow), x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = x[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride]
    return cols.reshape(N, C * kh * kw, oh * ow), oh, ow


def col2im(cols, x_shape, kh, kw, stride, pad, oh, ow):
    """Adjoint of im2col: scatter-add columns back (overlaps accumulate)."""
    N, C, H, W = x_shape
    dx = np.zeros((N, C, H + 2 * pad, W + 2 * pad), cols.dtype)
    cols = cols.reshape(N, C, kh, kw, oh, ow)
    for i in range(kh):
        for j in range(kw):
            dx[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride] += cols[:, :, i, j]
    return dx[:, :, pad:pad + H, pad:pad + W]


def conv_forward(x, w, b, stride=1, pad=0):
    F_, C, kh, kw = w.shape
    cols, oh, ow = im2col(x, kh, kw, stride, pad)
    out = w.reshape(F_, -1) @ cols + b[:, None]          # (N, F, OH*OW)
    return out.reshape(len(x), F_, oh, ow), (x.shape, cols, oh, ow)


def conv_backward(dout, w, cache, stride=1, pad=0):
    x_shape, cols, oh, ow = cache
    F_, C, kh, kw = w.shape
    d = dout.reshape(len(dout), F_, -1)                  # (N, F, OH*OW)
    dw = np.einsum("nfp,nkp->fk", d, cols).reshape(w.shape)
    db = d.sum((0, 2))
    dcols = w.reshape(F_, -1).T @ d                      # (N, C*kh*kw, OH*OW)
    dx = col2im(dcols, x_shape, kh, kw, stride, pad, oh, ow)
    return dx, dw, db


rng = np.random.default_rng(0)
x = rng.standard_normal((2, 3, 9, 9))
w = rng.standard_normal((4, 3, 3, 3))
b = rng.standard_normal(4)
stride, pad = 2, 1

out, cache = conv_forward(x, w, b, stride, pad)
dout = rng.standard_normal(out.shape)                # upstream gradient
dx, dw, db = conv_backward(dout, w, cache, stride, pad)

xt, wt, bt = (torch.tensor(a, requires_grad=True) for a in (x, w, b))
ot = F.conv2d(xt, wt, bt, stride=stride, padding=pad)
ot.backward(torch.tensor(dout))
print("output shape:", out.shape)
for name, ours, ref in [("out", out, ot), ("dx", dx, xt.grad),
                        ("dw", dw, wt.grad), ("db", db, bt.grad)]:
    print(f"{name}: max |ours - torch| = {np.abs(ours - ref.detach().numpy()).max():.1e}")
```

Output (numpy 2.5, torch 2.13 CPU):
```
output shape: (2, 4, 5, 5)
out: max |ours - torch| = 4.4e-16
dx: max |ours - torch| = 1.8e-15
dw: max |ours - torch| = 5.3e-15
db: max |ours - torch| = 8.9e-16
```

A 9×9 input with padding 1, a 3×3 kernel and stride 2 gives `(9 + 2 − 3) // 2 + 1 = 5`, as printed. All four tensors match PyTorch to float64 rounding. Passing a random `dout` to `backward` checks the full vector-Jacobian product, not just the gradient of a sum, so a transposed index anywhere would show up as an O(1) error.

## Choosing / trade-offs
- **im2col + GEMM vs direct loops.** im2col duplicates each input pixel up to `kh·kw` times (9× for 3×3, stride 1), but turns the work into one large, cache-friendly matmul. Direct convolution saves memory and is what specialised kernels do. For learning, im2col is the clearest correct version.
- **Loop over kernel offsets vs output positions.** Looping over the `kh·kw` offsets (9 iterations) vectorises over batch, channels and positions. Looping over `OH·OW` output pixels in Python is hundreds of times slower.
- **Hand-written backward vs autograd.** Writing the backward teaches where the gradients come from. In real code, define the forward with differentiable ops and let autograd do it, unless memory or speed forces a custom `autograd.Function`.
- **float64 vs float32.** Use float64 to verify (errors ~1e-15). In float32 expect ~1e-6 relative error, so compare with a tolerance (`torch.allclose(..., rtol=1e-5)`).

## Gotchas
- `=` instead of `+=` in col2im silently drops gradient wherever receptive fields overlap. It only shows up when stride < kernel size, so test with overlap.
- The slice end `i + stride*oh` matters: `x[:, :, i::stride]` without an end gives the wrong length when the padded size is not a multiple of the stride.
- Reshape order must match between `im2col` and `w.reshape(F, -1)`: both must flatten as `(C, kh, kw)`. A different order gives a valid-looking but wrong output.
- Padding must be removed from `dx` at the end, otherwise its shape does not match `x`.
- Kernel flipping: comparing with `scipy.signal.convolve2d` needs a flipped kernel; `scipy.signal.correlate2d` matches directly.
- im2col memory grows as `N·C·kh·kw·OH·OW`. On a real image batch this is gigabytes; it is a teaching and testing tool.
- A gradient comparison that only backpropagates `out.sum()` uses an all-ones `dout` and can hide transposition bugs. Use a random upstream gradient.

## Related
- [[neural-network-from-scratch-numpy]] - the dense-layer version of the same forward/backward/check workflow.
- [[convolution-arithmetic-and-receptive-field]] - output sizes, stride, padding, dilation and receptive fields.
- [[cnn-and-rnn-architectures]] - how conv layers are stacked into real networks.
- [[backpropagation-and-autograd]] - vector-Jacobian products and how autograd computes these gradients.
- [[tensor-shapes-broadcasting-and-einsum]] - the reshapes and the einsum used for `dw`.
- [[image-classification]] - using real conv nets for vision tasks.

## References
- PyTorch `torch.nn.functional.conv2d`: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
- PyTorch `torch.nn.Conv2d` (shape formula, cross-correlation note): https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- Stanford CS231n, Convolutional Neural Networks (im2col as matrix multiplication): https://cs231n.github.io/convolutional-networks/
