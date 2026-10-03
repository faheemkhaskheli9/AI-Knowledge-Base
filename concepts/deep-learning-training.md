---
title: Deep learning training (optimizers, regularization, normalization)
category: concepts
tags: [deep-learning, optimizer, adam, adamw, sgd, learning-rate, lr-schedule, dropout, batch-norm, layer-norm, weight-decay, early-stopping, pytorch]
use_cases:
  - "pick an optimizer and learning rate for a new neural network"
  - "fix a model whose training loss is stuck, oscillating or NaN"
  - "reduce overfitting when validation loss rises while training loss falls"
  - "set up warmup and cosine learning-rate decay for fine-tuning"
  - "choose batch norm vs layer norm and where dropout belongs"
status: draft
last_verified: 2026-10-03
sources:
  - https://pytorch.org/docs/stable/optim.html
  - https://pytorch.org/docs/stable/nn.html#normalization-layers
  - https://github.com/google-research/tuning_playbook
  - https://d2l.ai/chapter_optimization/index.html
  - https://arxiv.org/abs/1711.05101
---

# Deep learning training (optimizers, regularization, normalization)

## Summary
Knowing how backprop works ([[neural-network-fundamentals]]) is not enough to get a network to train well. Results depend on a handful of choices: the optimizer and learning rate (with warmup and decay), batch size, normalisation layers and residual connections that keep gradients healthy, and regularisation (weight decay, dropout, data augmentation, early stopping) that stops the model memorising. This file gives sensible defaults and a debugging order for each.

## Key concepts
- **Learning rate (LR)** is the most important hyperparameter. Too high: loss oscillates or goes NaN. Too low: training crawls or stalls in a poor region.
- **Optimizers.**
  - SGD + momentum (0.9): strong for CNNs on vision with a tuned schedule; often generalises well.
  - Adam: per-parameter adaptive steps; robust default, fast to converge.
  - AdamW: Adam with decoupled weight decay; the default for transformers and fine-tuning.
- **LR schedules.** Linear **warmup** over the first ~1-5% of steps avoids early instability (especially with Adam and large batches); then **cosine** or linear decay to near zero. `OneCycleLR` and step decay are common for CNNs. `ReduceLROnPlateau` when you do not know the step count.
- **Batch size.** Larger batches give smoother gradients and better GPU use but need a larger LR (roughly scaled with batch size) and may generalise slightly worse. Use gradient accumulation to reach an effective batch that does not fit in memory.
- **Normalisation.** BatchNorm (normalise over the batch; CNNs; behaves differently in train and eval), LayerNorm (normalise per sample; transformers, RNNs, small batches), RMSNorm (cheaper LayerNorm variant used in many LLMs), GroupNorm (small-batch vision).
- **Residual connections** (`x + f(x)`) let gradients flow through very deep networks; essential beyond ~20 layers.
- **Regularisation.**
  - Weight decay (L2-like shrinkage; typical 0.01-0.1 with AdamW, ~1e-4 with SGD).
  - Dropout (randomly zero activations; 0.1-0.5; off at eval).
  - Data augmentation (flips, crops, colour jitter, mixup/cutmix for images; noise, time-shift for audio): usually the strongest regulariser.
  - Early stopping on validation loss/metric; label smoothing for classification.
- **Gradient clipping** (`clip_grad_norm_`, max-norm ~1.0) prevents single bad batches from blowing up training; standard for RNNs and transformers.
- **Mixed precision** (bf16/fp16 autocast) speeds training and halves activation memory; see [[pytorch-basics]].

## When to use / scenarios
- Starting a new model: AdamW, LR around 1e-3 (from scratch) or 1e-5 to 5e-5 (fine-tuning a pretrained transformer), warmup + cosine, weight decay 0.01, gradient clipping 1.0.
- Vision CNN from scratch on a modest dataset: SGD momentum 0.9 or AdamW, strong augmentation, BatchNorm, cosine/one-cycle schedule.
- Validation loss rises while training loss falls (overfitting): more augmentation or data, more weight decay/dropout, early stopping, smaller model, or a pretrained backbone.
- Both losses high (underfitting): bigger model, train longer, raise LR, reduce regularisation, check the data pipeline.
- NOT relevant when only calling a hosted model API, or for tree models (their knobs are in [[gradient-boosting-tabular]]).

## Setup & code
```bash
pip install torch
```
AdamW with warmup + cosine decay, gradient clipping and early stopping:
```python
import math
import torch
from torch import nn

torch.manual_seed(0)
X = torch.randn(4000, 20)
y = (X[:, :5].sum(1) + 0.5 * torch.randn(4000) > 0).long()
X_tr, y_tr, X_va, y_va = X[:3200], y[:3200], X[3200:], y[3200:]

model = nn.Sequential(nn.Linear(20, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
                      nn.Linear(128, 2))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

epochs, bs = 50, 64
total = epochs * math.ceil(len(X_tr) / bs)
warmup = int(0.05 * total)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup
    else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (total - warmup))))
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

best, patience, bad = float("inf"), 5, 0
for epoch in range(epochs):
    model.train()
    perm = torch.randperm(len(X_tr))
    for i in range(0, len(X_tr), bs):
        idx = perm[i:i + bs]
        loss = loss_fn(model(X_tr[idx]), y_tr[idx])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()                                   # per step, not per epoch
    model.eval()
    with torch.no_grad():
        val = loss_fn(model(X_va), y_va).item()
    if val < best - 1e-4:
        best, bad = val, 0
        torch.save(model.state_dict(), "best.pt")
    else:
        bad += 1
        if bad >= patience:
            print(f"early stop at epoch {epoch}")
            break
print(f"best val loss {best:.3f}")
```
Hugging Face `transformers.get_cosine_schedule_with_warmup` provides the same schedule ready-made.

## Choosing / trade-offs
- **Adam(W) vs SGD.** Adam(W) converges fast with little tuning and is standard for transformers; SGD + momentum needs more LR tuning but remains competitive for CNNs.
- **Finding the LR.** Sweep on a log scale (1e-5 … 1e-1) for a short run and pick the largest LR that still trains stably; tune LR before anything else.
- **BatchNorm vs LayerNorm.** BatchNorm for CNNs with batches ≥ ~16; LayerNorm/RMSNorm for transformers, sequence models and tiny batches.
- **Regularise with data first.** Augmentation and more data beat stacking dropout and weight decay; pretrained backbones beat both on small datasets.
- **Train longer vs early stopping.** With a decaying schedule, set epochs up front; early stopping is a safety net, not a substitute for a schedule.

## Gotchas
- Debug order when nothing learns: overfit one small batch first; if that fails, check labels, loss/output pairing, input scaling and LR before changing architecture.
- Calling `scheduler.step()` per epoch for a schedule computed in steps (or vice versa) makes warmup/decay far too fast or slow.
- Applying weight decay to biases and normalisation weights can hurt; transformer recipes usually exclude them via parameter groups.
- BatchNorm with batch size 1-2 or a forgotten `model.eval()` gives garbage validation numbers.
- Changing batch size without adjusting LR changes training dynamics; re-tune or scale LR.
- Dropout plus heavy augmentation plus big weight decay on a small model leads to underfitting; remove regularisers one at a time.
- Random seeds alone do not make GPU training bit-for-bit reproducible; see PyTorch's reproducibility notes if you need it.

## Related
- [[neural-network-fundamentals]] - backprop, losses and gradient descent.
- [[pytorch-basics]] - DataLoader, GPU, mixed precision and checkpointing around this loop.
- [[cnn-and-rnn-architectures]] - where BatchNorm, LayerNorm and clipping are typically used.
- [[fine-tuning-and-peft]] - LR and schedule choices for adapting pretrained models.
- [[experiment-tracking]] - log LR curves and losses to compare runs.

## References
- PyTorch optimizers and LR schedulers: https://pytorch.org/docs/stable/optim.html
- PyTorch normalisation layers: https://pytorch.org/docs/stable/nn.html#normalization-layers
- Google Deep Learning Tuning Playbook: https://github.com/google-research/tuning_playbook
- *Dive into Deep Learning*, Optimization chapter: https://d2l.ai/chapter_optimization/index.html
- Loshchilov & Hutter, Decoupled Weight Decay Regularization (AdamW): https://arxiv.org/abs/1711.05101
