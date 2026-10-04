---
title: Batch normalization and dropout from scratch (forward, backward and train/eval modes in NumPy)
category: concepts
tags: [batch-normalization, batchnorm, dropout, inverted-dropout, running-statistics, train-eval-mode, regularization, numpy, pytorch, from-scratch, dl-basics]
use_cases:
  - "implement batch norm forward and backward from scratch and check it against PyTorch"
  - "understand why model.eval() changes the output of a network"
  - "explain inverted dropout and why nothing is scaled at test time"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1502.03167
  - https://jmlr.org/papers/v15/srivastava14a.html
  - https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
  - https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
---

# Batch normalization and dropout from scratch (forward, backward and train/eval modes in NumPy)

## Summary
Batch normalization and dropout are the two common layers that behave differently in training and inference. Batch norm normalises each feature with the statistics of the current mini-batch while training and with running averages at test time; dropout zeroes random activations while training and does nothing at test time. Writing both in NumPy, including the batch-norm backward pass, makes that mode switch concrete. The implementation below matches PyTorch's `BatchNorm1d` forward, input gradient, `gamma` gradient and running variance to within `1e-14`.

## Key concepts
- **Batch norm forward (training).** For each feature: `x̂ = (x − μ_B) / √(σ²_B + ε)`, then `y = γ·x̂ + β`. `μ_B` and `σ²_B` are the mean and the *biased* variance over the batch; `γ` and `β` are learned so the layer can undo the normalisation if that helps.
- **Running statistics.** During training the layer keeps exponential moving averages `run = (1 − m)·run + m·batch_stat` with `m = 0.1` in PyTorch. PyTorch stores the *unbiased* variance (`× n/(n−1)`) in `running_var`. At eval time those averages replace the batch statistics, so the output for one example no longer depends on the rest of the batch.
- **Batch norm backward.** `μ_B` and `σ²_B` depend on every row, so the gradient for one row has three terms. With `g = dL/dx̂ = dout·γ`: `dx = (1/n)·(1/√(σ²+ε))·(n·g − Σg − x̂·Σ(g·x̂))`. Also `dγ = Σ dout·x̂` and `dβ = Σ dout`.
- **Inverted dropout.** In training, keep each unit with probability `1 − p` and divide survivors by `1 − p`, so the expected activation is unchanged. At test time the layer is the identity. (The original paper scaled weights at test time instead; every modern framework uses the inverted form.)
- **Dropout backward.** Multiply the upstream gradient by the same mask; dropped units get zero gradient.
- **Train/eval mode.** Both layers read a `training` flag. Forgetting `model.eval()` before inference gives noisy predictions (dropout still active) that also depend on batch composition (batch norm still using batch stats).

## When to use / scenarios
- Learning: the clearest examples of layers with state and mode-dependent behaviour.
- Interviews: "derive the batch-norm backward", "what happens at test time", "why scale by `1/(1−p)`".
- Debugging: a model that scores well in training mode but badly after `eval()` (or the reverse) usually has a batch-norm statistics problem ([[debugging-neural-network-training]]).
- Not for: production layers. Use `torch.nn.BatchNorm*d` / `nn.Dropout`. For transformers and small or variable batches, prefer LayerNorm or RMSNorm ([[normalization-layers]]).

## Setup & code
`pip install numpy torch` (torch only for the comparison). Runs in about a second.

```python
import numpy as np
import torch


class BatchNorm1d:
    def __init__(self, d, momentum=0.1, eps=1e-5):
        self.gamma, self.beta = np.ones(d), np.zeros(d)
        self.run_mean, self.run_var = np.zeros(d), np.ones(d)
        self.momentum, self.eps, self.training = momentum, eps, True

    def forward(self, x):
        if self.training:
            n = len(x)
            if n < 2:                                        # same check PyTorch does
                raise ValueError("BatchNorm needs more than 1 value per channel in training")
            mu, var = x.mean(0), x.var(0)                    # biased var for normalising
            m = self.momentum
            self.run_mean = (1 - m) * self.run_mean + m * mu
            self.run_var = (1 - m) * self.run_var + m * var * n / (n - 1)   # unbiased for running stats
        else:
            mu, var = self.run_mean, self.run_var
        self.xhat = (x - mu) / np.sqrt(var + self.eps)
        self.std_inv = 1 / np.sqrt(var + self.eps)
        return self.gamma * self.xhat + self.beta

    def backward(self, dout):                                # training-mode backward
        n = len(dout)
        self.dgamma = (dout * self.xhat).sum(0)
        self.dbeta = dout.sum(0)
        dxhat = dout * self.gamma
        # compact form: the batch mean and variance both depend on every x
        return self.std_inv / n * (n * dxhat - dxhat.sum(0) - self.xhat * (dxhat * self.xhat).sum(0))


class Dropout:
    def __init__(self, p=0.5, seed=0):
        self.p, self.training = p, True
        self.rng = np.random.default_rng(seed)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x                                         # inverted dropout: nothing to do at test time
        self.mask = (self.rng.random(x.shape) >= self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


# 1) BatchNorm forward/backward and running stats vs PyTorch
rng = np.random.default_rng(0)
x = rng.normal(3.0, 2.0, (32, 4))
dout = rng.normal(size=(32, 4))
bn = BatchNorm1d(4)
bn.gamma = rng.normal(size=4)
y = bn.forward(x)
dx = bn.backward(dout)

tbn = torch.nn.BatchNorm1d(4).double()
with torch.no_grad():
    tbn.weight.copy_(torch.from_numpy(bn.gamma))
tx = torch.from_numpy(x).requires_grad_()
ty = tbn(tx)
ty.backward(torch.from_numpy(dout))
print("forward max|diff|  ", f"{np.abs(y - ty.detach().numpy()).max():.1e}")
print("dx max|diff|       ", f"{np.abs(dx - tx.grad.numpy()).max():.1e}")
print("dgamma max|diff|   ", f"{np.abs(bn.dgamma - tbn.weight.grad.numpy()).max():.1e}")
print("running_var diff   ", f"{np.abs(bn.run_var - tbn.running_var.numpy()).max():.1e}")
print("train-mode out mean/std", np.round(bn.xhat.mean(0), 6)[:2], np.round(bn.xhat.std(0), 4)[:2])

# 2) Train vs eval mode: a single example
bn.training = False
print("eval single row ok:", bn.forward(x[:1]).shape)
try:
    BatchNorm1d(4).forward(x[:1])
except ValueError as e:
    print("train single row:", e)

# 3) Dropout keeps the expected activation
d = Dropout(p=0.5)
h = np.ones((10000, 100))
out = d.forward(h)
print(f"dropout p=0.5 train mean={out.mean():.3f} zeros={np.mean(out == 0):.3f}")
d.training = False
print(f"dropout eval mean={d.forward(h).mean():.3f}")
```

Output (numpy 2.5, torch 2.13):
```
forward max|diff|   8.9e-16
dx max|diff|        4.4e-16
dgamma max|diff|    4.4e-15
running_var diff    0.0e+00
train-mode out mean/std [0. 0.] [1. 1.]
eval single row ok: (1, 4)
train single row: BatchNorm needs more than 1 value per channel in training
dropout p=0.5 train mean=1.000 zeros=0.500
dropout eval mean=1.000
```

Every batch-norm quantity agrees with PyTorch to floating-point rounding, which confirms both the three-term backward and the unbiased running variance. A single row works in eval mode but is rejected in training mode, because a batch of one has zero variance. Dropout zeroes half the units yet keeps the mean activation at 1.0, so the next layer sees the same scale in training and at test time.

## Choosing / trade-offs
- **BatchNorm vs LayerNorm/GroupNorm.** BatchNorm works well for CNNs with batches of 32 or more. With small batches (detection, segmentation, per-GPU batches of 2 to 8) its statistics get noisy; GroupNorm or LayerNorm do not depend on the batch ([[normalization-layers]]).
- **Momentum.** Lower momentum makes running stats smoother but slower to track a changing network. If eval accuracy lags training, recompute the statistics with a few forward passes in training mode over clean data (no gradient) before evaluating.
- **Dropout rate.** `0.1` to `0.3` for transformers and MLPs, up to `0.5` for large fully-connected heads. Modern CNNs often rely on batch norm, data augmentation and weight decay instead ([[regularization-in-deep-learning]]).
- **Dropout + BatchNorm.** Dropout before a batch-norm layer shifts the activation variance between training and eval. Put dropout after the last normalisation layer, or avoid combining them.

## Gotchas
- Forgetting `model.eval()` at inference, or forgetting `model.train()` after an evaluation loop inside training. Both silently change results.
- PyTorch's batch-norm `momentum` is the weight on the *new* value (`0.1`), the opposite of the usual "momentum" convention (`0.9` on the old value) in Keras and in optimizers.
- Normalising with the unbiased variance in training, or storing the biased one in `running_var`, gives outputs that differ slightly from PyTorch and from loaded checkpoints.
- In the backward pass, treating `μ` and `σ²` as constants (the "naive" gradient `dx = g/σ`) is wrong. A finite-difference or framework comparison, as above, catches it.
- Batch norm leaks information across the examples in a batch. In contrastive learning and some ranking setups this lets the model cheat; shuffle across devices or use another normalisation.
- Dropout masks must be resampled on every forward pass. Caching one mask turns dropout into a fixed sparse network.
- Monte Carlo dropout (dropout left on at test time to estimate uncertainty) is a deliberate exception; turn on only the dropout layers, not batch norm ([[bayesian-deep-learning-and-uncertainty]]).

## Related
- [[normalization-layers]] - BatchNorm vs LayerNorm vs RMSNorm vs GroupNorm and where to place them.
- [[regularization-in-deep-learning]] - dropout next to weight decay, augmentation and early stopping.
- [[neural-network-from-scratch-numpy]] - the MLP these layers slot into.
- [[autograd-engine-from-scratch]] - how frameworks get these backward passes automatically.
- [[debugging-neural-network-training]] - symptoms of train/eval mode mistakes.

## References
- Ioffe and Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", 2015: https://arxiv.org/abs/1502.03167
- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 2014: https://jmlr.org/papers/v15/srivastava14a.html
- PyTorch BatchNorm1d: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
- PyTorch Dropout: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
