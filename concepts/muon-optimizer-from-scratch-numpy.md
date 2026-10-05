---
title: Muon optimizer from scratch (Newton-Schulz orthogonalisation, momentum, which parameters get it)
category: concepts
tags: [muon, optimizer, newton-schulz, orthogonalization, momentum, adamw, spectral-norm, pretraining, numpy, from-scratch]
use_cases:
  - "understand what Muon does differently from AdamW before trying it on a pretraining run"
  - "decide which parameters go to Muon and which stay on AdamW (embeddings, heads, biases, norms)"
  - "implement the Newton-Schulz orthogonalisation step and check how close it gets to U V^T"
  - "benchmark a new optimizer fairly against a tuned Adam baseline"
status: draft
last_verified: 2026-10-05
sources:
  - https://kellerjordan.github.io/posts/muon/
  - https://arxiv.org/abs/2502.16982
---

# Muon optimizer from scratch (Newton-Schulz orthogonalisation, momentum, which parameters get it)

## Summary
Muon ("MomentUm Orthogonalized by Newton-Schulz", Keller Jordan 2024) is an optimizer for the 2D weight matrices of a network. It takes the SGD-momentum update for a matrix and replaces it with the nearest semi-orthogonal matrix `U Vᵀ` (from the update's SVD `U S Vᵀ`), so every singular direction moves by the same amount instead of a few dominant directions taking almost all the step. The orthogonalisation uses a few Newton-Schulz matrix iterations instead of an SVD, so it runs in bf16 on GPU at small cost. This file implements it in NumPy, checks how close five iterations get to `U Vᵀ`, and trains a small MLP against SGD-momentum and Adam with a learning-rate grid for each.

## Key concepts
- **Orthogonalised update.** For gradient-momentum `M = U S Vᵀ`, Muon steps along `U Vᵀ`: same singular vectors, all singular values set to 1. Gradients of weight matrices are usually close to low rank, so plain momentum spends the step on a few directions; orthogonalising boosts the rare directions that are still useful.
- **Newton-Schulz iteration.** Normalise `X = M / ‖M‖_F` (all singular values in (0, 1]), then repeat `X ← aX + (b·XXᵀ + c·(XXᵀ)²)X`. Each step maps every singular value `s` to `a s + b s³ + c s⁵` without touching the singular vectors. Jordan's coefficients `(3.4445, −4.7750, 2.0315)` push small values up fast and leave all of them in roughly [0.7, 1.2] instead of exactly 1, which works as well in practice and needs only 5 steps.
- **Momentum first, then orthogonalise.** Muon keeps a normal momentum buffer (β = 0.95, Nesterov form) and orthogonalises the buffer, not the raw gradient.
- **Shape scaling.** The update is scaled by `√max(1, rows/cols)` in the original code. Moonshot's large-scale version scales by `0.2·√max(rows, cols)` so the update RMS matches AdamW's and AdamW learning rates and weight decay transfer (Liu et al. 2025).
- **Only hidden matrices.** Embeddings, the output head, biases and norm gains stay on AdamW. Embedding and head rows are looked up one token at a time, so orthogonalising the whole matrix makes no sense for them. Conv kernels are flattened to 2D first.
- **Steepest descent under the spectral norm.** The `U Vᵀ` direction is the steepest-descent step when step size is measured by the spectral norm, which bounds how much a layer's output can change. That is the usual explanation for why it suits weight matrices better than Adam's per-coordinate scaling.

## When to use / scenarios
- Pretraining or long training of transformers and MLPs where optimizer efficiency matters: speedrun-style results (NanoGPT, CIFAR-10) and Moonshot's reported ~2× compute efficiency over AdamW for LLM pretraining.
- Experiments where AdamW is already well tuned and you want the next cheap gain without changing the model.
- Not for fine-tuning a model pretrained with AdamW unless you test it: Liu et al. report the best results when pretraining and fine-tuning use the same optimizer.
- Not for scalar or vector parameters, embeddings or heads. Use AdamW for those inside the same run.
- Not as a default to recommend blindly. Like other new optimizers, benchmark against a tuned AdamW baseline on your own setup ([[optimizers]]).

## Setup & code
NumPy only, about 20 seconds on a laptop CPU. Part 1 checks the Newton-Schulz output on an ill-conditioned matrix. Part 2 trains a 3-layer tanh MLP on a teacher network with ill-conditioned inputs, trying four learning rates per optimizer with the same seed, batch order and linear decay. Muon updates the two hidden matrices and Adam updates the output layer, as in real use.

```python
import numpy as np


def newton_schulz5(G, steps=5, eps=1e-7):
    """Approximate the orthogonal factor U V^T of G = U S V^T with a quintic iteration (no SVD)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (np.linalg.norm(G) + eps)          # Frobenius norm <= 1 puts every singular value in (0, 1]
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T                                # iterate on the wide side: X @ X.T is the smaller Gram matrix
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X    # each singular value s -> a s + b s^3 + c s^5
    return X.T if tall else X


rng = np.random.default_rng(0)
G = rng.standard_normal((64, 32)) @ np.diag(np.logspace(0, -3, 32)) @ rng.standard_normal((32, 32))
s_in = np.linalg.svd(G, compute_uv=False)
print(f"input singular values: max {s_in.max():.2f}, min {s_in.min():.4f}, condition {s_in.max() / s_in.min():.0f}")
for k in (1, 3, 5, 10):
    s = np.linalg.svd(newton_schulz5(G, steps=k), compute_uv=False)
    print(f"  NS steps {k:2d}: singular values in [{s.min():.3f}, {s.max():.3f}]")
U, _, Vt = np.linalg.svd(G, full_matrices=False)
O = newton_schulz5(G)
cos = (O * (U @ Vt)).sum() / np.linalg.norm(O) / np.linalg.norm(U @ Vt)
print(f"  cosine(NS5 output, exact U V^T) = {cos:.3f}")

# Train a 3-layer tanh MLP on a teacher network; hidden matrices with each optimizer
d_in, h, n = 32, 128, 2048
X = rng.standard_normal((n, d_in)) @ np.diag(np.logspace(0, -1.5, d_in))   # ill-conditioned inputs
T1, T2 = rng.standard_normal((d_in, 64)) / np.sqrt(d_in) * 3, rng.standard_normal((64, 1)) / 8
Y = np.tanh(X @ T1) @ T2


def init(seed):
    r = np.random.default_rng(seed)
    return [r.standard_normal((d_in, h)) / np.sqrt(d_in), r.standard_normal((h, h)) / np.sqrt(h),
            r.standard_normal((h, 1)) / np.sqrt(h)]


def loss_grad(W, xb, yb):
    h1 = np.tanh(xb @ W[0]); h2 = np.tanh(h1 @ W[1]); out = h2 @ W[2]
    err = out - yb
    loss = (err ** 2).mean()
    g_out = 2 * err / len(xb)
    g2 = h2.T @ g_out
    d2 = (g_out @ W[2].T) * (1 - h2 ** 2)
    g1 = h1.T @ d2
    d1 = (d2 @ W[1].T) * (1 - h1 ** 2)
    g0 = xb.T @ d1
    return loss, [g0, g1, g2]


def train(kind, lr, steps=600, bs=128, seed=1):
    W = init(seed)
    r = np.random.default_rng(seed)
    m = [np.zeros_like(w) for w in W]; v = [np.zeros_like(w) for w in W]
    for t in range(1, steps + 1):
        idx = r.integers(0, n, bs)
        _, g = loss_grad(W, X[idx], Y[idx])
        lr_t = lr * min(1.0, (steps - t) / (0.3 * steps) + 1e-3)    # linear decay over the last 30%
        for i in range(3):
            # Muon only for the two hidden matrices; the output layer uses Adam (standard practice)
            if kind == "muon" and i < 2:
                m[i] = 0.95 * m[i] + g[i]
                upd = newton_schulz5(g[i] + 0.95 * m[i])                     # Nesterov momentum, then orthogonalise
                W[i] -= lr_t * max(1.0, W[i].shape[0] / W[i].shape[1]) ** 0.5 * upd
            elif kind == "sgd":
                m[i] = 0.9 * m[i] + g[i]
                W[i] -= lr_t * m[i]
            else:                                                            # Adam (also Muon's output layer)
                a_lr = lr_t if kind == "adam" else 3e-3 * lr_t / lr
                m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
                mh, vh = m[i] / (1 - 0.9 ** t), v[i] / (1 - 0.999 ** t)
                W[i] -= a_lr * mh / (np.sqrt(vh) + 1e-8)
    return loss_grad(W, X, Y)[0]


base = (Y ** 2).mean()
print(f"\nteacher-student MLP, full-data MSE / mean(y^2) after 600 steps, lr grid (nan = diverged):")
for kind, grid in [("sgd", [0.03, 0.1, 0.3, 1.0]), ("adam", [3e-3, 1e-2, 3e-2, 1e-1]), ("muon", [0.003, 0.01, 0.03, 0.1])]:
    with np.errstate(all="ignore"):
        res = {lr: train(kind, lr) / base for lr in grid}
    best = min(res, key=lambda k: np.nan_to_num(res[k], nan=np.inf))
    print(f"  {kind:5s} " + "  ".join(f"lr {lr:g}: {r:.4f}" for lr, r in res.items()) + f"   -> best {res[best]:.4f}")
```

Output (Python 3.14, NumPy 2.5):
```
input singular values: max 40.14, min 0.0030, condition 13310
  NS steps  1: singular values in [0.000, 1.190]
  NS steps  3: singular values in [0.002, 1.173]
  NS steps  5: singular values in [0.022, 1.199]
  NS steps 10: singular values in [0.682, 1.134]
  cosine(NS5 output, exact U V^T) = 0.906

teacher-student MLP, full-data MSE / mean(y^2) after 600 steps, lr grid (nan = diverged):
  sgd   lr 0.03: 0.0410  lr 0.1: 0.0256  lr 0.3: 0.0217  lr 1: nan   -> best 0.0217
  adam  lr 0.003: 0.0118  lr 0.01: 0.0082  lr 0.03: 0.0068  lr 0.1: 0.0321   -> best 0.0068
  muon  lr 0.003: 0.0056  lr 0.01: 0.0002  lr 0.03: 0.0005  lr 0.1: 0.0012   -> best 0.0002
```

How to read it:
- The test matrix has condition number 13,310. Five Newton-Schulz steps bring the top singular values to about 1 but leave the smallest at 0.022; ten steps put all of them in [0.68, 1.13]. Even so, five steps already point within cosine 0.906 of the exact `U Vᵀ`. Real gradients are not this ill-conditioned after momentum, and the tiny directions are mostly noise anyway, which is why 5 steps is the default.
- The quintic deliberately overshoots (max 1.19 rather than 1.0). The coefficients trade exactness for speed of lifting small singular values.
- On the MLP, Muon's best run reaches 0.0002 relative MSE against 0.0068 for Adam and 0.0217 for SGD-momentum, each at its own best learning rate. Every Muon learning rate in the grid beats the best Adam run, so the gain here is not a lucky tuning point.
- This is one toy task with 600 steps. It shows the mechanism and that the code is correct, not the size of the gain on a real model; the published claims are about LLM pretraining at scale.

## Choosing / trade-offs
- **Muon vs AdamW.** Muon stores one buffer per matrix instead of Adam's two, and its extra cost is a few matrix multiplications per step (small next to the forward and backward pass for large batches). AdamW is still the safe default and the one every recipe, schedule and fine-tuning guide assumes.
- **Hybrid parameter groups.** Real setups run two optimizers: Muon for 2D hidden weights (attention projections, MLP matrices), AdamW for embeddings, head, norms and biases. Getting the split wrong (e.g. Muon on the embedding) is a common source of worse results.
- **Learning rate transfer.** With the original scaling, Muon learning rates (~0.02) are not comparable to AdamW's. With Moonshot's RMS-matching scale, AdamW's learning rate and weight decay can be reused; that version also adds weight decay, which they found necessary at scale.
- **Distributed cost.** Newton-Schulz needs the full matrix, so with sharded (ZeRO/FSDP) parameters the gradient has to be gathered for the update. Library implementations handle this; a hand-rolled version will not.
- **Steps of Newton-Schulz.** 5 is standard. More steps get closer to exact `U Vᵀ` at more cost; fewer steps leave small directions under-scaled.

## Gotchas
- Normalise by the Frobenius norm before iterating. Without it singular values above ~1.5 diverge under the quintic.
- Iterate on the wide orientation (transpose tall matrices) so `XXᵀ` is the smaller Gram matrix.
- Orthogonalise the momentum (or the Nesterov combination), not the raw gradient; orthogonalising the gradient and then adding momentum is a different optimizer.
- A zero gradient (frozen or unused rows) gives a zero matrix; the `eps` in the normaliser keeps it from producing NaNs.
- Do not reuse an AdamW learning rate with the original `√max(1, rows/cols)` scaling. The update RMS is different by an order of magnitude.
- bf16 is fine for the iteration on GPU (the original is written for it); fp16 can overflow in `A @ A` for unnormalised inputs.

## Related
- [[optimizers]] - AdamW, schedules and where newer optimizers fit.
- [[optimizers-from-scratch-numpy]] - SGD, momentum and Adam implemented the same way.
- [[pretraining-and-scaling-laws]] - the setting where Muon's reported gains apply.
- [[transformer-block-from-scratch-numpy]] - which matrices in a block would go to Muon.
- [[lora-from-scratch-numpy]] - another place where the low-rank structure of weight updates matters.

## References
- Keller Jordan (2024), "Muon: An optimizer for hidden layers in neural networks": https://kellerjordan.github.io/posts/muon/
- Liu et al. (2025), "Muon is Scalable for LLM Training" (Moonshot AI): https://arxiv.org/abs/2502.16982
