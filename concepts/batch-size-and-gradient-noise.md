---
title: Batch size, gradient noise and the learning-rate scaling rule
category: concepts
tags: [batch-size, mini-batch, gradient-noise, learning-rate, linear-scaling-rule, gradient-accumulation, large-batch-training, pytorch, deep-learning-basics]
use_cases:
  - "choose a batch size for training a neural network on one GPU"
  - "train with a larger effective batch than fits in GPU memory"
  - "adjust the learning rate after changing the batch size or the number of GPUs"
  - "understand why small batches sometimes generalise better"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1706.02677
  - https://arxiv.org/abs/1812.06162
  - https://arxiv.org/abs/1609.04836
  - https://arxiv.org/abs/1711.00489
---

# Batch size, gradient noise and the learning-rate scaling rule

## Summary
Each mini-batch gradient is a noisy estimate of the full-dataset gradient, and the noise shrinks roughly as `1/√batch_size`. Batch size therefore trades update quality against cost: small batches give many cheap, noisy steps; large batches give fewer, cleaner steps that use the hardware better. Because batch size and learning rate together set the noise level, changing one without the other changes training. The usual rule is to scale the learning rate with the batch size and add warmup.

## Key concepts
- **Gradient noise.** The standard error of a mini-batch gradient falls as `1/√B`. Going from 8 to 512 samples cuts it about 8×, not 64×.
- **Noise scale.** For SGD the effective "temperature" of training is roughly `lr / B`. Keeping that ratio fixed keeps training dynamics similar, which is where the linear scaling rule comes from.
- **Linear scaling rule (SGD).** Multiply the batch by k, multiply the learning rate by k, and warm up the learning rate over the first few epochs (Goyal et al. trained ImageNet ResNet-50 at batch 8192 this way). For Adam-family optimizers the square-root rule (`lr × √k`) is often a better start.
- **Critical batch size.** Up to a problem-dependent batch size, doubling the batch nearly halves the number of steps needed. Beyond it, extra batch buys little and just costs compute. It grows as training progresses and the loss falls.
- **Gradient accumulation.** Run several micro-batches, add up the gradients, step once. Gives a large effective batch on small memory, at the same compute per sample.
- **Generalization.** Very large batches without retuning tend to reach sharper minima and lower test accuracy. With a retuned learning rate, warmup and enough epochs, much of that gap closes.
- **Batch size and BatchNorm.** BatchNorm statistics get noisy below ~16 samples per device; use GroupNorm/LayerNorm or sync BN instead ([[normalization-layers]]).

## When to use / scenarios
- Starting a new model on one GPU: pick the largest power-of-two batch that fits comfortably (often 32 to 256 for vision, token-count based for LLMs), tune the learning rate for it, then leave it alone.
- Moving from 1 GPU to N GPUs with data parallelism: the global batch grows N×, so scale the learning rate and add warmup ([[distributed-training]]).
- Fine-tuning a large model that only fits batch 1 to 4 in memory: use gradient accumulation to reach an effective batch of 16 to 64.
- Do NOT grow the batch just to finish epochs faster when you are already past the critical batch size. Wall-clock per epoch drops, but steps-to-target barely improve and the hardware is wasted.

## Setup & code
`pip install torch`. The script measures gradient noise at three batch sizes, then shows correct gradient accumulation and the linear scaling rule.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
X = torch.randn(4096, 20)
y = (X[:, :5].sum(1, keepdim=True) > 0).float()
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
loss_fn = nn.BCEWithLogitsLoss()

# Gradient noise: spread of mini-batch gradients around the full-batch gradient
def flat_grad(xb, yb):
    model.zero_grad()
    loss_fn(model(xb), yb).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])

full = flat_grad(X, y)
for bs in [8, 64, 512]:
    errs = []
    for _ in range(50):
        idx = torch.randint(0, len(X), (bs,))
        errs.append((flat_grad(X[idx], y[idx]) - full).norm() / full.norm())
    print(f"batch={bs:4d}  relative gradient noise = {torch.stack(errs).mean():.3f}")

# Gradient accumulation: effective batch 256 from micro-batches of 64
opt = torch.optim.SGD(model.parameters(), lr=0.1)
accum_steps, micro = 4, 64
opt.zero_grad()
for i in range(accum_steps):
    xb, yb = X[i * micro:(i + 1) * micro], y[i * micro:(i + 1) * micro]
    (loss_fn(model(xb), yb) / accum_steps).backward()  # divide so grads average
opt.step()

# Linear scaling rule: lr grows with batch size (with warmup), from a tuned base
base_lr, base_bs = 0.1, 256
for bs in [256, 1024, 4096]:
    print(f"batch={bs:5d}  lr ~ {base_lr * bs / base_bs:.2f} (SGD, linear rule)")
```

Output (torch 2.13 CPU):
```
batch=   8  relative gradient noise = 2.737
batch=  64  relative gradient noise = 0.966
batch= 512  relative gradient noise = 0.367
batch=  256  lr ~ 0.10 (SGD, linear rule)
batch= 1024  lr ~ 0.40 (SGD, linear rule)
batch= 4096  lr ~ 1.60 (SGD, linear rule)
```
8× more samples cut the noise about 2.8× (√8 ≈ 2.83), as the `1/√B` rule predicts.

## Choosing / trade-offs
- **Small batch (8 to 64).** Fits small GPUs, adds regularising noise, needs a lower learning rate, uses the GPU less efficiently and is slower per epoch.
- **Large batch (1k+).** Best throughput and fewest optimizer steps, needs learning-rate scaling, warmup and sometimes layer-wise optimizers (LARS/LAMB) to keep quality.
- **Accumulation vs a bigger GPU.** Accumulation reproduces the maths of a large batch (except BatchNorm statistics) at the same speed per sample. More memory only helps if it also raises throughput.
- **Fix batch, tune lr** is the practical order: batch size is mostly a hardware decision, learning rate is the knob that adapts to it ([[learning-rate-schedules]], [[optimizers]]).

## Gotchas
- Forgetting to divide the loss by `accum_steps` makes the gradient k× larger, which silently acts as a k× learning rate.
- With `DistributedDataParallel` plus accumulation, wrap the non-final micro-batches in `model.no_sync()` or every micro-batch triggers an all-reduce.
- Learning-rate schedules count optimizer steps. After raising the batch, the same number of epochs means fewer steps, so warmup and decay lengths set in steps must shrink too.
- The last batch of an epoch can be tiny. With BatchNorm, set `drop_last=True` on the training `DataLoader` to avoid a noisy or failing step.
- Comparing two runs with different batch sizes at the same learning rate compares two different noise levels, not just two batch sizes.
- Doubling the batch on a GPU that is already saturated does not double throughput; measure samples/second before assuming speed-up.

## Related
- [[gradient-descent]] - batch vs mini-batch vs stochastic gradient descent from first principles.
- [[learning-rate-schedules]] - warmup and decay, which large-batch training depends on.
- [[optimizers]] - SGD vs Adam behaviour and LARS/LAMB for very large batches.
- [[distributed-training]] - global batch size under data parallelism.
- [[efficient-training-mixed-precision]] - memory savings that allow larger batches.
- [[generalization-in-deep-learning]] - sharp vs flat minima and implicit regularisation from SGD noise.

## References
- Goyal et al., Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour (2017): https://arxiv.org/abs/1706.02677
- McCandlish et al., An Empirical Model of Large-Batch Training (critical batch size, 2018): https://arxiv.org/abs/1812.06162
- Keskar et al., On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima (2016): https://arxiv.org/abs/1609.04836
- Smith et al., Don't Decay the Learning Rate, Increase the Batch Size (2017): https://arxiv.org/abs/1711.00489
