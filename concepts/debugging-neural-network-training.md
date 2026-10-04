---
title: Debugging neural network training (initialization, NaNs, loss not decreasing)
category: concepts
tags: [deep-learning, debugging, weight-initialization, nan, exploding-gradients, vanishing-gradients, gradient-clipping, lr-finder, sanity-checks, pytorch]
use_cases:
  - "my loss is not decreasing / stuck at the chance level"
  - "training loss goes to NaN or inf after a few steps"
  - "the model trains but validation accuracy is far below what it should be"
  - "choose a weight initialization and learning rate for a new architecture"
  - "set up sanity checks before an expensive training run"
status: draft
last_verified: 2026-10-04
sources:
  - https://karpathy.github.io/2019/04/25/recipe/
  - https://pytorch.org/docs/stable/nn.init.html
  - https://pytorch.org/docs/stable/autograd.html#anomaly-detection
  - https://github.com/google-research/tuning_playbook
---

# Debugging neural network training (initialization, NaNs, loss not decreasing)

## Summary
Neural networks fail silently: buggy code usually still trains, just badly. A
fixed order of sanity checks (check the data, check the initial loss, overfit
one batch, then scale up) finds most bugs in minutes. This file covers that
order, weight initialization, and the usual causes of NaNs, exploding and
vanishing gradients and a loss that will not move.

## Key concepts
- **Expected initial loss.** With `C` balanced classes and a sensible init,
  cross-entropy starts near `ln(C)` (2.30 for 10 classes). Far higher means
  the init or the last layer is wrong; this is the first check.
- **Overfit one batch.** A correct model and loop drives the loss on a single
  small batch to almost zero. If it cannot, the bug is in the code (labels,
  loss, shapes, optimizer wiring), not in the data or model size.
- **Initialization.** Weights must keep activation variance stable across
  layers. He/Kaiming init for ReLU-family activations, Xavier/Glorot for
  tanh/sigmoid. PyTorch and Keras layers already default to sensible inits;
  custom layers and raw `nn.Parameter(torch.randn(...))` often do not.
  Initialize the final bias to the class prior (e.g. `log(p/(1-p))` for a
  rare positive class) to skip the first wasted epochs.
- **Gradient health.** Log the global gradient norm. Exploding (norm grows,
  then NaN) -> lower the LR, clip gradients, add normalization. Vanishing
  (early layers' grads near zero) -> ReLU/GELU, residual connections,
  normalization, better init.
- **Learning rate is the first hyperparameter.** Too high diverges or
  plateaus at a high loss; too low barely moves. An LR range test (raise LR
  exponentially over a few hundred steps, pick roughly 1/10 of where loss
  blows up) finds a starting value.
- **Train/val gap reading.** Both high: underfitting or a bug. Train low, val
  high: overfitting, or a val set drawn from a different distribution. Val better than train: dropout/augmentation
  active only in training, or leakage.

## When to use / scenarios
- Before any long or expensive run: the checks take minutes and save GPU-days.
- When a working model is ported to new data, a new framework or a new
  architecture and suddenly does worse.
- When a loss curve looks wrong: flat, NaN, spiky, or diverging after warm-up.
- NOT a substitute for evaluation design: if the metric or split is wrong,
  see [[model-evaluation-and-metrics]] first.

## Setup & code
```python
import math, torch
from torch import nn

torch.manual_seed(0)
X, y = torch.randn(512, 20), torch.randint(0, 10, (512,))
model = nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Linear(128, 10))
loss_fn = nn.CrossEntropyLoss()

# 1. Initial loss should be close to ln(num_classes).
with torch.no_grad():
    print("init loss", loss_fn(model(X), y).item(), "expected", math.log(10))

# 2. Overfit a single small batch: loss should approach 0.
xb, yb = X[:32], y[:32]
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(500):
    opt.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward()
    # 3. Watch gradient norm and clip it (returns the pre-clip norm).
    gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss at step {step}")
print("one-batch loss", loss.item(), "grad norm", gnorm.item())

# 4. Locate the op that produced a NaN (slow: debugging only).
# with torch.autograd.detect_anomaly(): loss.backward()
```

## Choosing / trade-offs
- **Gradient clipping** (norm 1.0 is a common default for transformers) is
  cheap insurance against spikes, but hides an LR that is simply too high;
  check how often it triggers.
- **Warm-up** of the LR over the first few hundred to few thousand steps
  stabilizes Adam and large batches; it adds one more knob.
- **Mixed precision** saves memory but fp16 overflows; prefer bfloat16, or
  use a gradient scaler with fp16 (see [[deep-learning-training]]).
- **How much to log.** Loss, LR, gradient norm and a few activation stats per
  layer catch almost everything; full histograms every step are slow.

## Gotchas
- Common NaN sources: LR too high, `log(0)` or division by zero in a custom
  loss, softmax on large logits done by hand (use the framework's fused
  `cross_entropy`/`log_softmax`), fp16 overflow, NaNs already in the input data.
- Applying softmax before `CrossEntropyLoss` (which expects logits) trains
  slowly with oddly capped accuracy.
- Shape broadcasting bugs: `(N,1)` predictions against `(N,)` targets in an
  MSE silently become an `N x N` loss.
- Forgetting `model.eval()` at validation keeps dropout on and batch-norm
  statistics updating; forgetting `zero_grad()` sums gradients across steps.
- Augmentation or label shuffling that breaks the input-label pairing still
  trains, to chance level; visualize a few decoded batches as the model sees them.
- Dead ReLUs (a large share of units always zero) after a too-high LR;
  check the fraction of zero activations.
- Only one seed: a single good or bad run is noise; compare across 2-3 seeds
  before concluding a change helped.

## Related
- [[neural-network-fundamentals]] - backprop, activations and losses.
- [[deep-learning-training]] - optimizers, schedules, normalization and mixed precision.
- [[pytorch-basics]] - the training loop these checks plug into.
- [[math-for-machine-learning]] - numerical stability (log-sum-exp) behind many NaNs.
- [[experiment-tracking]] - log curves and gradient norms per run.
- [[hyperparameter-tuning]] - searching LR and friends once the code is correct.

## References
- https://karpathy.github.io/2019/04/25/recipe/
- https://pytorch.org/docs/stable/nn.init.html
- https://pytorch.org/docs/stable/autograd.html#anomaly-detection
- https://github.com/google-research/tuning_playbook
