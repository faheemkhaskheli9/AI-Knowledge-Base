---
title: Optimizers from scratch in NumPy (SGD, momentum, RMSProp, Adam/AdamW, checked against torch.optim)
category: concepts
tags: [optimizers, sgd, momentum, rmsprop, adam, adamw, bias-correction, weight-decay, rosenbrock, numpy, pytorch, from-scratch, deep-learning-basics]
use_cases:
  - "implement SGD, momentum, RMSProp and Adam update rules from scratch"
  - "understand Adam's bias correction and how AdamW's decoupled weight decay differs"
  - "verify a hand-written optimizer step for step against torch.optim"
  - "compare how optimizers handle an ill-conditioned valley like the Rosenbrock function"
status: draft
last_verified: 2026-10-04
sources:
  - https://docs.pytorch.org/docs/stable/optim.html
  - https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
  - https://arxiv.org/abs/1412.6980
  - https://arxiv.org/abs/1711.05101
---

# Optimizers from scratch in NumPy (SGD, momentum, RMSProp, Adam/AdamW, checked against torch.optim)

## Summary
Every deep-learning optimizer is a small rule that turns the current gradient (plus a little state) into a parameter update. SGD, momentum, RMSProp and Adam/AdamW each fit in under ten lines of NumPy. Run on the Rosenbrock function, the hand-written versions match `torch.optim` to machine precision after 200 steps, and a race over 5000 steps shows why adaptive methods are the default for hard, badly scaled problems.

## Key concepts
- **SGD.** `p ← p − η·g`. One learning rate for every parameter, so it crawls along flat directions and bounces in steep ones.
- **Momentum.** Keep a velocity `v ← μ·v + g` and step `p ← p − η·v`. Consistent gradient directions build up speed (about `1/(1−μ)` times, 10× for `μ = 0.9`); oscillating ones cancel out. This is PyTorch's form; some texts scale `g` by `η` or `(1−μ)` instead.
- **RMSProp.** Track a running mean of squared gradients `s ← α·s + (1−α)·g²` and divide: `p ← p − η·g/(√s + ε)`. Each parameter gets its own effective step size, large where gradients are small.
- **Adam.** Momentum on the gradient (`m`, first moment) plus RMSProp on its square (`v`, second moment): `p ← p − η·m̂/(√v̂ + ε)`.
- **Bias correction.** `m` and `v` start at zero, so early on they underestimate the true averages. Dividing by `1 − βᵗ` fixes this; without it the first steps are far too small (with `β₂ = 0.999`, `v` is only 0.1% of its target after one step).
- **AdamW (decoupled weight decay).** Adding `λ·p` to the gradient (L2) gets rescaled by Adam's per-parameter divisor, so heavily-updated weights are barely regularised. AdamW instead shrinks the weights directly, `p ← p·(1 − η·λ)`, before the Adam step ([[regularization-in-deep-learning]]).

## When to use / scenarios
- Learning: see exactly what `optimizer.step()` does and what state it stores.
- Interviews: "write Adam", "why bias correction", "Adam vs AdamW", "why momentum helps".
- Debugging: a reference to compare a custom or fused optimizer against, or to explain why training diverged after an optimizer change ([[debugging-neural-network-training]]).
- Real training: use `torch.optim` (or `optax` in JAX). AdamW is the default for transformers; SGD with momentum is still strong for CNNs on vision ([[optimizers]]).

## Setup & code
`pip install numpy torch`. Runs in about a second on CPU.

```python
import numpy as np
import torch


class SGD:
    def __init__(self, lr, momentum=0.0):
        self.lr, self.mu, self.v = lr, momentum, 0.0

    def step(self, p, g):
        self.v = self.mu * self.v + g                # PyTorch-style momentum buffer
        return p - self.lr * self.v


class RMSProp:
    def __init__(self, lr, alpha=0.99, eps=1e-8):
        self.lr, self.a, self.eps, self.s = lr, alpha, eps, 0.0

    def step(self, p, g):
        self.s = self.a * self.s + (1 - self.a) * g ** 2
        return p - self.lr * g / (np.sqrt(self.s) + self.eps)


class Adam:
    def __init__(self, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, weight_decay
        self.m = self.v = 0.0
        self.t = 0

    def step(self, p, g):
        self.t += 1
        if self.wd:
            p = p * (1 - self.lr * self.wd)          # decoupled decay = AdamW
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)     # bias correction: m, v start at 0
        v_hat = self.v / (1 - self.b2 ** self.t)
        return p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def rosenbrock(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)])


# 1) Step-for-step check against torch.optim
for name, ours, ref in (
    ("SGD+momentum", SGD(1e-3, 0.9), lambda q: torch.optim.SGD(q, lr=1e-3, momentum=0.9)),
    ("RMSprop", RMSProp(1e-3), lambda q: torch.optim.RMSprop(q, lr=1e-3)),
    ("AdamW", Adam(1e-2, weight_decay=0.1), lambda q: torch.optim.AdamW(q, lr=1e-2, weight_decay=0.1)),
):
    p = np.array([-1.5, 2.0])
    q = torch.tensor(p.copy(), requires_grad=True)
    opt = ref([q])
    for _ in range(200):
        p = ours.step(p, rosenbrock_grad(p))
        opt.zero_grad()
        x, y = q
        ((1 - x) ** 2 + 100 * (y - x ** 2) ** 2).backward()
        opt.step()
    print(f"{name:<13} max |ours - torch| after 200 steps: {np.abs(p - q.detach().numpy()).max():.1e}")

# 2) Race on Rosenbrock from (-1.5, 2); minimum is at (1, 1)
for name, opt in (("SGD", SGD(1e-4)), ("SGD+momentum", SGD(1e-4, 0.9)),
                  ("RMSProp", RMSProp(1e-2)), ("Adam", Adam(1e-2)), ("Adam lr=1e-1", Adam(1e-1))):
    p = np.array([-1.5, 2.0])
    for _ in range(5000):
        p = opt.step(p, rosenbrock_grad(p))
    print(f"{name:<13} after 5000 steps: p={p.round(3)}, loss={rosenbrock(p):.2e}")
```

Output (numpy 2.5, torch 2.13 CPU):
```
SGD+momentum  max |ours - torch| after 200 steps: 5.0e-16
RMSprop       max |ours - torch| after 200 steps: 0.0e+00
AdamW         max |ours - torch| after 200 steps: 0.0e+00
SGD           after 5000 steps: p=[-1.116  1.254], loss=4.48e+00
SGD+momentum  after 5000 steps: p=[0.92  0.846], loss=6.38e-03
RMSProp       after 5000 steps: p=[0.989 0.963], loss=2.22e-02
Adam          after 5000 steps: p=[1. 1.], loss=1.68e-17
Adam lr=1e-1  after 5000 steps: p=[1. 1.], loss=3.72e-08
```

All three hand-written rules agree with `torch.optim` to float64 precision, using PyTorch's default hyperparameters (`alpha = 0.99`, `betas = (0.9, 0.999)`, `eps = 1e-8`). The Rosenbrock valley is long, curved and badly scaled: the gradient across it is hundreds of times larger than along it. Plain SGD needs a tiny step (`1e-4`) to stay stable across the valley, so it is still far from the minimum after 5000 steps. Momentum with the same step gets close. RMSProp and Adam rescale each coordinate and reach the minimum; RMSProp at a fixed `1e-2` ends up hovering near it rather than settling, because without momentum it keeps taking full-size steps. A 2-D toy is not a neural network, so read this as intuition, not a benchmark.

## Choosing / trade-offs
- **AdamW vs SGD+momentum.** AdamW is robust to the learning rate and to badly scaled parameters, and is the default for transformers and most new work. Well-tuned SGD+momentum with a schedule can generalise slightly better on some vision tasks, but needs more tuning ([[optimizers]]).
- **Learning rate first.** Every optimizer here is only as good as its learning rate and schedule. Tune `η` before anything else; add warmup and decay for long runs ([[learning-rate-schedules]]).
- **Memory.** SGD has no state, momentum stores one buffer per parameter, Adam stores two. For very large models that extra state is a big part of GPU memory, which is why 8-bit and factored optimizers exist.
- **ε.** Mostly irrelevant, but raise it (e.g. `1e-6`) under mixed precision, where `√v̂` can underflow and steps explode ([[efficient-training-mixed-precision]]).

## Gotchas
- Forgetting bias correction makes Adam take tiny steps for the first few hundred iterations, which looks like a learning rate set too low.
- `torch.optim.Adam(weight_decay=...)` is L2 regularisation through the gradient, not AdamW. Use `torch.optim.AdamW` for decoupled weight decay; the two are not interchangeable at the same `λ`.
- Momentum conventions differ between libraries (`v ← μv + g` vs `v ← μv + ηg` vs `v ← μv + (1−μ)g`). They are equivalent only after rescaling `η`, so a learning rate copied across frameworks may be wrong by 10×.
- Optimizer state belongs to the checkpoint. Resuming with fresh `m`, `v` or momentum buffers causes a loss spike ([[model-checkpointing-and-export]]).
- Do not apply weight decay to biases and normalisation parameters; standard recipes put them in a parameter group with `weight_decay=0`.
- Adam's `t` counter is per-optimizer. Rebuilding the optimizer mid-training resets bias correction too.
- A smooth 2-D function rewards adaptive methods. Mini-batch gradient noise changes the picture a lot ([[batch-size-and-gradient-noise]]).

## Related
- [[optimizers]] - practical choice and tuning of optimizers in real training.
- [[gradient-descent]] - plain gradient descent, step size and convergence.
- [[learning-rate-schedules]] - warmup and decay that go around every optimizer.
- [[loss-landscapes-and-flat-minima]] - the terrain these optimizers move through.
- [[regularization-in-deep-learning]] - weight decay and why AdamW decouples it.
- [[neural-network-from-scratch-numpy]] - plug these update rules into a from-scratch MLP.

## References
- PyTorch `torch.optim` (update rules for SGD, RMSprop, Adam, AdamW): https://docs.pytorch.org/docs/stable/optim.html
- PyTorch AdamW: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- Kingma and Ba, "Adam: A Method for Stochastic Optimization": https://arxiv.org/abs/1412.6980
- Loshchilov and Hutter, "Decoupled Weight Decay Regularization": https://arxiv.org/abs/1711.05101
