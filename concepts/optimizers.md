---
title: Optimizers (SGD, momentum, RMSprop, Adam, AdamW)
category: concepts
tags: [deep-learning, optimizer, sgd, momentum, nesterov, rmsprop, adam, adamw, weight-decay, parameter-groups, pytorch]
use_cases:
  - "pick an optimizer and starting learning rate for a new neural network"
  - "understand why Adam and AdamW differ when weight decay is on"
  - "exclude biases and normalization weights from weight decay"
  - "switch a CNN from Adam to SGD with momentum for better final accuracy"
  - "reduce optimizer memory when fine-tuning a large model"
status: draft
last_verified: 2026-10-04
sources:
  - https://pytorch.org/docs/stable/optim.html
  - https://arxiv.org/abs/1412.6980
  - https://arxiv.org/abs/1711.05101
  - https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
  - https://proceedings.mlr.press/v28/sutskever13.html
---

# Optimizers (SGD, momentum, RMSprop, Adam, AdamW)

## Summary
An optimizer turns gradients into parameter updates. Plain SGD steps against the gradient; momentum adds a running average of past gradients so updates keep direction through noise and narrow valleys; RMSprop and Adam additionally divide each parameter's step by a running estimate of its gradient magnitude, so every parameter gets its own effective learning rate. AdamW is Adam with weight decay applied directly to the weights instead of through the gradient, and is the default for transformers and most fine-tuning. SGD with momentum stays competitive for CNNs trained from scratch.

## Key concepts
- **SGD.** `w -= lr * g`. One hyperparameter, but the right LR depends heavily on the model and data, and progress stalls in directions with small gradients.
- **Momentum (0.9 typical).** `v = μ·v + g; w -= lr·v`. Averages noisy mini-batch gradients and speeds up along consistent directions. **Nesterov** evaluates the gradient after the momentum step; usually a small win, never a loss.
- **RMSprop.** Keeps an exponential average of `g²` and divides the step by its square root, so parameters with large gradients take smaller steps.
- **Adam.** Momentum (first moment, β1 = 0.9) plus RMSprop scaling (second moment, β2 = 0.999), with bias correction for the early steps. `eps` (1e-8) guards the division.
- **L2 penalty vs decoupled weight decay.** Adding `λ·w` to the gradient (L2, what `Adam(weight_decay=...)` does) gets rescaled by Adam's per-parameter denominator, so weights with large gradients are barely decayed. **AdamW** subtracts `lr·λ·w` from the weights directly. For SGD the two are equivalent; for Adam they are not.
- **Parameter groups.** Each group can have its own `lr`, `weight_decay`, etc. Standard recipe: decay matrices/conv kernels, not biases or norm γ/β.
- **Optimizer state.** Adam keeps two extra tensors per parameter (fp32), so weights + gradients + state is roughly 4x the weight memory before activations. SGD with momentum keeps one.

## When to use / scenarios
- Transformers, LLM fine-tuning, most new architectures, anything you do not want to tune much: **AdamW**, LR 1e-3 to 3e-4 from scratch, 1e-5 to 5e-5 for fine-tuning a pretrained transformer, weight decay 0.01-0.1.
- Image classification CNN from scratch (ResNet-style) with time to tune: **SGD + Nesterov momentum 0.9**, LR around 0.1 for batch 256 (scale linearly with batch), weight decay 1e-4 to 5e-4, cosine schedule.
- RNNs, reinforcement learning: Adam or RMSprop; both tolerate non-stationary gradients.
- Very large models where optimizer memory dominates: 8-bit Adam (bitsandbytes), Adafactor, or sharded optimizer state ([[distributed-training]]).
- Classical ML (linear/logistic models on tabular data): you rarely pick an optimizer by hand; scikit-learn's solvers (lbfgs, saga) handle it ([[linear-models]]).

## Setup & code
```bash
pip install torch
```
Same MLP and data, five optimizers, 300 mini-batch steps each (torch 2.13 CPU):
```python
import torch
from torch import nn

torch.manual_seed(0)
X = torch.randn(2000, 20)
w_true = torch.randn(20, 1)
y = (X @ w_true + 0.1 * torch.randn(2000, 1) > 0).float()

def make():
    torch.manual_seed(1)
    return nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))

opts = {
    "SGD lr=0.1":            lambda p: torch.optim.SGD(p, lr=0.1),
    "SGD+momentum lr=0.1":   lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, nesterov=True),
    "RMSprop lr=1e-3":       lambda p: torch.optim.RMSprop(p, lr=1e-3),
    "Adam lr=1e-3":          lambda p: torch.optim.Adam(p, lr=1e-3),
    "AdamW lr=1e-3 wd=0.01": lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=0.01),
}
loss_fn = nn.BCEWithLogitsLoss()
for name, mk in opts.items():
    model = make()
    opt = mk(model.parameters())
    for step in range(300):
        idx = torch.randint(0, 2000, (64,))
        loss = loss_fn(model(X[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        print(f"{name:24s} full-data loss after 300 steps: {loss_fn(model(X), y).item():.3f}")
# SGD 0.102 | SGD+momentum 0.017 | RMSprop 0.089 | Adam 0.130 | AdamW 0.130
```
The ranking above says more about the chosen LRs than about the optimizers: each one has its own good LR range, so compare optimizers only after tuning LR for each.

Parameter groups with no decay on biases and norm weights (the usual AdamW recipe):
```python
model = nn.Sequential(nn.Linear(20, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 1))
decay, no_decay = [], []
for n, p in model.named_parameters():
    (no_decay if p.ndim < 2 else decay).append(p)   # 1-D tensors: biases, LayerNorm/BatchNorm γ/β
opt = torch.optim.AdamW([
    {"params": decay, "weight_decay": 0.01},
    {"params": no_decay, "weight_decay": 0.0},
], lr=1e-3)
print(len(decay), len(no_decay))   # 2 4
```
Different LRs for a pretrained backbone and a new head (fine-tuning):
```python
opt = torch.optim.AdamW([
    {"params": backbone.parameters(), "lr": 1e-5},
    {"params": head.parameters(), "lr": 1e-3},
], weight_decay=0.01)
```

## Choosing / trade-offs
- **AdamW vs SGD + momentum.** AdamW converges fast from a wide LR range and is the safe default. SGD + momentum needs LR and schedule tuning but often generalises slightly better on vision CNNs and uses less memory.
- **Adam vs AdamW.** With weight decay > 0, prefer AdamW. Plain `Adam(weight_decay=...)` is L2 regularisation, which interacts badly with adaptive scaling.
- **Memory.** SGD + momentum: 1 state tensor per parameter. Adam/AdamW: 2. 8-bit optimizers or Adafactor cut this when fine-tuning large models; see [[fine-tuning-and-peft]] and [[efficient-training-mixed-precision]].
- **Large batch.** Adaptive layer-wise methods (LARS for SGD, LAMB for Adam) exist for very large batch training; for typical batch sizes they are unnecessary.
- **Newer optimizers.** Several newer optimizers (e.g. Lion, Sophia, Muon, schedule-free variants) report gains in specific settings. Treat them as experiments to benchmark against a tuned AdamW baseline, not as defaults.

## Gotchas
- `opt.zero_grad()` is required every step (or every N steps with gradient accumulation); otherwise gradients sum across steps.
- Creating the optimizer before moving the model to GPU, or before replacing layers, leaves it holding stale parameters. Build the model fully, `.to(device)`, then create the optimizer.
- Freezing layers after creating the optimizer: PyTorch skips parameters whose `.grad` is `None`, but if a frozen parameter's `.grad` is a zero tensor (`zero_grad(set_to_none=False)`), AdamW still applies weight decay and leftover momentum to it. Leave frozen parameters out of the optimizer.
- Loading a checkpoint to resume training: save and load `opt.state_dict()` as well as the model, or Adam restarts with empty moments and the loss spikes.
- `eps=1e-8` can underflow in fp16. Mixed-precision training keeps optimizer state in fp32 (AMP does this); custom fp16 setups may need `eps=1e-6`.
- Changing batch size changes the best LR. A common starting rule: scale LR linearly with batch size for SGD, by about the square root for Adam, then re-tune.

## Related
- [[learning-rate-schedules]] - the LR is the hyperparameter that matters most; the schedule shapes it over time.
- [[deep-learning-training]] - the full training recipe these optimizers sit in.
- [[regularization-in-deep-learning]] - weight decay alongside dropout and early stopping.
- [[backpropagation-and-autograd]] - where the gradients come from.
- [[debugging-neural-network-training]] - LR finder and symptoms of a wrong LR.
- [[math-for-machine-learning]] - gradient descent and the optimization basics.

## References
- PyTorch `torch.optim`: https://pytorch.org/docs/stable/optim.html
- Kingma & Ba, Adam: A Method for Stochastic Optimization: https://arxiv.org/abs/1412.6980
- Loshchilov & Hutter, Decoupled Weight Decay Regularization (AdamW): https://arxiv.org/abs/1711.05101
- Hinton, RMSprop (Coursera lecture 6 slides): https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
- Sutskever et al., On the importance of initialization and momentum in deep learning: https://proceedings.mlr.press/v28/sutskever13.html
