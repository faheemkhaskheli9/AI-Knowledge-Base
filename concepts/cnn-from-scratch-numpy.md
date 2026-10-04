---
title: CNN from scratch (max-pooling forward and backward, and an end-to-end conv net trained in NumPy)
category: concepts
tags: [cnn, max-pooling, pooling, convolution, im2col, col2im, backpropagation, softmax, sgd, numpy, pytorch, from-scratch, deep-learning-basics]
use_cases:
  - "implement max-pooling forward and backward from scratch"
  - "train a complete small convolutional network in pure NumPy, end to end"
  - "see how conv, ReLU, pooling and a dense softmax layer chain together in backprop"
  - "verify hand-written CNN layers against PyTorch autograd"
status: draft
last_verified: 2026-10-04
sources:
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
  - https://cs231n.github.io/convolutional-networks/
---

# CNN from scratch (max-pooling forward and backward, and an end-to-end conv net trained in NumPy)

## Summary
A minimal CNN is conv → ReLU → max-pool → dense → softmax. Max-pooling keeps the largest value in each window and routes the whole gradient back to that one position (a mask), so its backward pass has no weights. With a reshape-based pool, an im2col conv and the softmax gradient `p − onehot`, about 70 lines of NumPy train a 8-filter CNN to 96.5% on scikit-learn's 8×8 digits in a few seconds on CPU. Pooling and conv gradients both match PyTorch autograd to 1e-15. This builds on [[convolution-layer-from-scratch-numpy]], which derives the conv layer itself.

## Key concepts
- **Max-pool forward.** For non-overlapping `k×k` windows, reshape `(N, C, H, W)` to `(N, C, H/k, k, W/k, k)` and take the max over the two `k` axes. No loops, no copies.
- **Max-pool backward.** `y = max(window)` has gradient 1 for the arg-max element and 0 for the rest, so `dx = mask · dout` broadcast back into the window. Save the mask in the forward pass.
- **Ties.** If two elements in a window are equal (common after ReLU zeros), the gradient must go to exactly one. Keep the first max (`cumsum(mask) == 1`), as PyTorch does; otherwise the gradient is double-counted.
- **Pooling has no parameters.** It shrinks the spatial size (8×8 → 4×4 here, so the dense layer needs 4× fewer weights) and adds small translation invariance.
- **Backprop chain.** `dlogits = softmax − onehot` (over batch size) → dense: `dW2 = flatᵀ·dlogits`, `dflat = dlogits·W2ᵀ` → reshape → pool backward (mask) → ReLU backward (`· (z > 0)`) → conv: `dW1 = dzᵀ·cols`, `dx = col2im(dz·W1)`.
- **"Same" padding.** A 3×3 kernel with padding 1 keeps 8×8 → 8×8, so the pool divides evenly ([[convolution-arithmetic-and-receptive-field]]).
- **He init for ReLU.** Conv weights scaled by `√(2 / fan_in)` keep activations from vanishing in the first epochs ([[weight-initialization]]).

## When to use / scenarios
- Learning: the step from "I have a conv layer" to "I have a network that trains"; every piece of a real CNN pipeline is visible.
- Interviews: max-pool backward, the tie case, and why pooling reduces parameters.
- Debugging framework models: the same checks (forward error, gradient error against a reference) find bugs in custom layers.
- Production: do NOT use this. `torch.nn` layers run on optimised kernels and GPUs, and modern nets often replace max-pool with strided convs or global average pooling ([[cnn-and-rnn-architectures]]).

## Setup & code
`pip install numpy scikit-learn torch` (torch only for the checks). Runs in about 5 seconds on CPU.

```python
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def maxpool_forward(x, k=2):
    """x: (N, C, H, W) with H, W divisible by k. Returns output and argmax mask."""
    N, C, H, W = x.shape
    win = x.reshape(N, C, H // k, k, W // k, k)
    out = win.max(axis=(3, 5))
    mask = win == out[:, :, :, None, :, None]
    mask &= np.cumsum(np.cumsum(mask, 3), 5) == 1              # keep first max only (ties)
    return out, mask


def maxpool_backward(dout, mask):
    N, C, Ho, k, Wo, _ = mask.shape
    return (mask * dout[:, :, :, None, :, None]).reshape(N, C, Ho * k, Wo * k)


def im2col(x, kh, kw):                                         # stride 1, 'same' padding
    p = kh // 2
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    N, C, H, W = x.shape
    cols = np.lib.stride_tricks.sliding_window_view(xp, (kh, kw), axis=(2, 3))
    return cols.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * W, C * kh * kw)


def col2im(dcols, shape, kh, kw):
    N, C, H, W = shape
    p = kh // 2
    dxp = np.zeros((N, C, H + 2 * p, W + 2 * p))
    d = dcols.reshape(N, H, W, C, kh, kw)
    for i in range(kh):
        for j in range(kw):
            dxp[:, :, i:i + H, j:j + W] += d[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    return dxp[:, :, p:p + H, p:p + W]


# 1) Check max-pool against PyTorch autograd.
rng = np.random.default_rng(0)
x = rng.standard_normal((2, 3, 4, 4))
out, mask = maxpool_forward(x)
dout = rng.standard_normal(out.shape)
xt = torch.tensor(x, requires_grad=True)
ot = F.max_pool2d(xt, 2)
ot.backward(torch.tensor(dout))
print("pool fwd max err:", np.abs(out - ot.detach().numpy()).max())
print("pool bwd max err:", np.abs(maxpool_backward(dout, mask) - xt.grad.numpy()).max())

# 2) Train conv(8, 3x3) -> ReLU -> maxpool 2 -> dense(10) on 8x8 digits.
X, y = load_digits(return_X_y=True)
X = (X / 16.0).reshape(-1, 1, 8, 8)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

C, K = 8, 3
W1 = rng.standard_normal((C, 1 * K * K)) * np.sqrt(2 / (K * K))  # He init
b1 = np.zeros(C)
W2 = rng.standard_normal((C * 4 * 4, 10)) * np.sqrt(1 / (C * 16))
b2 = np.zeros(10)


def forward(xb):
    cols = im2col(xb, K, K)
    z = (cols @ W1.T + b1).reshape(len(xb), 8, 8, C).transpose(0, 3, 1, 2)
    a = np.maximum(z, 0)
    p, m = maxpool_forward(a)
    logits = p.reshape(len(xb), -1) @ W2 + b2
    return logits, (cols, z, p, m)


lr, bs = 0.1, 64
for epoch in range(15):
    perm = rng.permutation(len(Xtr))
    for s in range(0, len(perm), bs):
        idx = perm[s:s + bs]
        xb, yb = Xtr[idx], ytr[idx]
        logits, (cols, z, p, m) = forward(xb)
        prob = np.exp(logits - logits.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
        dlog = prob.copy(); dlog[np.arange(len(yb)), yb] -= 1; dlog /= len(yb)
        flat = p.reshape(len(xb), -1)
        dW2, db2 = flat.T @ dlog, dlog.sum(0)
        da = maxpool_backward((dlog @ W2.T).reshape(p.shape), m) * (z > 0)
        dz = da.transpose(0, 2, 3, 1).reshape(-1, C)
        dW1, db1 = dz.T @ cols, dz.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    if epoch % 5 == 4:
        loss = -np.log(prob[np.arange(len(yb)), yb]).mean()
        acc = (forward(Xte)[0].argmax(1) == yte).mean()
        print(f"epoch {epoch + 1}: last-batch loss={loss:.3f} test acc={acc:.3f}")

# col2im is the input gradient of the conv; check it against torch too.
xt = torch.tensor(Xte[:4], requires_grad=True)
zt = F.conv2d(xt, torch.tensor(W1.reshape(C, 1, K, K)), torch.tensor(b1), padding=1)
g = rng.standard_normal(zt.shape)
zt.backward(torch.tensor(g))
dx = col2im(g.transpose(0, 2, 3, 1).reshape(-1, C) @ W1, Xte[:4].shape, K, K)
print("conv fwd max err:", np.abs(forward(Xte[:4])[1][1] - zt.detach().numpy()).max())
print("conv input-grad max err:", np.abs(dx - xt.grad.numpy()).max())
```

Output (numpy 2.5, scikit-learn 1.9, torch 2.13 CPU):
```
pool fwd max err: 0.0
pool bwd max err: 0.0
epoch 5: last-batch loss=0.473 test acc=0.893
epoch 10: last-batch loss=0.242 test acc=0.922
epoch 15: last-batch loss=0.108 test acc=0.965
conv fwd max err: 4.440892098500626e-16
conv input-grad max err: 3.552713678800501e-15
```

Max-pooling matches `F.max_pool2d` exactly in both directions, and the conv forward and input gradient match `F.conv2d` to float64 rounding. The network (8 filters, 1,370 parameters) reaches 96.5% test accuracy after 15 epochs of plain SGD. The input gradient is not needed to train the first layer; it is computed here only to show that `col2im` is correct, which matters as soon as a second conv layer sits underneath.

## Choosing / trade-offs
- **Max vs average pooling.** Max keeps the strongest activation (edge or feature present anywhere in the window) and is the classic choice after early conv layers. Average pooling is smoother; global average pooling before the classifier replaces a large dense layer in modern CNNs.
- **Pooling vs strided convolution.** A stride-2 conv downsamples with learned weights; most recent architectures (ResNet stems aside) prefer it over max-pool. Pooling is free and parameter-less.
- **Reshape pool vs general pool.** The reshape trick only handles non-overlapping windows with sizes that divide `H` and `W`. Overlapping (3×3, stride 2) or padded pooling needs `sliding_window_view` plus a scatter-add backward, the same as im2col/col2im.
- **NumPy vs framework.** NumPy is for understanding and checks. Real training belongs in PyTorch ([[pytorch-basics]]), which also gives you GPU, autograd and optimisers.

## Gotchas
- `dlog = prob` then editing `dlog` in place also edits `prob`, which turns the logged loss into `log(negative) = nan` while training still works. Copy first, as in the code.
- Without the tie rule, a window of four ReLU zeros sends the gradient four times. Results look nearly right and the gradient check fails by small amounts.
- Pool, conv and dense shapes must agree: `(N, C, H, W)` for conv and pool, and the `(N, H, W, C)` order of im2col rows when reshaping `z`. A wrong transpose still runs and silently trains on scrambled features.
- Divide the softmax gradient by the batch size once. Dividing again in the conv layer makes the first layer learn far slower than the last.
- Forgetting `* (z > 0)` (ReLU backward) still lowers the loss for a while; the gradient check against autograd is what catches it ([[debugging-neural-network-training]]).
- `X / 16.0` scales the digit pixels to `[0, 1]`; on raw 0-16 values the same learning rate diverges.

## Related
- [[convolution-layer-from-scratch-numpy]] - the conv layer with stride and padding, derived step by step.
- [[neural-network-from-scratch-numpy]] - the dense-layer version of the same training loop.
- [[softmax-regression-from-scratch]] - the softmax and cross-entropy gradient used at the output.
- [[convolution-arithmetic-and-receptive-field]] - output sizes for conv and pooling.
- [[cnn-and-rnn-architectures]] - how these blocks are stacked in real CNNs.
- [[image-classification]] - production image classifiers.

## References
- PyTorch `torch.nn.functional.max_pool2d`: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html
- PyTorch `torch.nn.functional.conv2d`: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
- Stanford CS231n, Convolutional Neural Networks: https://cs231n.github.io/convolutional-networks/
