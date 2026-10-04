---
title: Learning rate schedules (warmup, cosine, one-cycle, WSD, plateau)
category: concepts
tags: [deep-learning, learning-rate, lr-schedule, warmup, cosine-decay, one-cycle, wsd, reduce-lr-on-plateau, step-decay, pytorch]
use_cases:
  - "choose a learning rate schedule for training a network from scratch or fine-tuning"
  - "add warmup to stop a transformer from diverging in the first steps"
  - "decay the learning rate when validation loss stops improving"
  - "train without fixing the total number of steps in advance"
  - "fix a schedule that decays far too fast because it is stepped per batch instead of per epoch"
status: draft
last_verified: 2026-10-04
sources:
  - https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  - https://arxiv.org/abs/1608.03983
  - https://arxiv.org/abs/1803.09820
  - https://arxiv.org/abs/1706.02677
  - https://arxiv.org/abs/2404.06395
---

# Learning rate schedules (warmup, cosine, one-cycle, WSD, plateau)

## Summary
A learning rate schedule changes the step size during training: usually a short **warmup** from near zero, then a **decay** toward a small value. A high LR early explores quickly; a low LR late settles into a good minimum, and most of the final loss improvement typically happens during the decay. Warmup + cosine decay is the common default for transformers and modern CNNs; one-cycle suits fixed short budgets; warmup-stable-decay (WSD) lets you keep training without committing to a total length; ReduceLROnPlateau reacts to the validation metric when you cannot plan the run.

## Key concepts
- **Warmup.** Linearly ramp the LR over the first ~1-5% of steps (hundreds to a few thousand). Adam's second-moment estimate is noisy at the start and large random-init gradients can blow up; warmup avoids early divergence, especially with large batches and post-norm transformers.
- **Step decay.** Multiply LR by 0.1 at fixed epochs (classic ResNet: epochs 30, 60, 90). Simple; needs milestone tuning.
- **Cosine decay.** `lr = min + (max − min)·(1 + cos(π·t/T))/2`. Smooth, one hyperparameter (T), robust. Requires knowing the total steps.
- **One-cycle.** Ramp up to `max_lr` for ~30% of training, then anneal far below the start. Often reaches a good result in fewer epochs for a fixed budget.
- **WSD (warmup-stable-decay, "trapezoidal").** Warmup, hold constant, then a short decay (often the last 10-20%). You can branch a decay off any checkpoint of the stable phase, so the total length need not be fixed upfront; popular for LLM pretraining.
- **ReduceLROnPlateau.** Cut LR by a factor when a monitored metric has not improved for `patience` epochs. Steps on a metric, not on a counter.
- **Per-step vs per-epoch.** Most schedules are written in optimizer steps. Stepping a step-based schedule once per epoch (or vice versa) silently stretches or compresses it.

## When to use / scenarios
- Transformer from scratch or fine-tuning: linear warmup (e.g. 3-10% of steps for fine-tuning) then cosine or linear decay to ~0. This is the Hugging Face `Trainer` default family.
- CNN from scratch with a known epoch budget: cosine (with a few epochs of warmup) or one-cycle.
- Long or open-ended pretraining where you may extend the run: WSD.
- Small tabular or legacy models, or a run whose length is unknown and you watch validation loss: ReduceLROnPlateau.
- Hyperparameter search where trials differ in length: cosine to each trial's own length, or WSD, so trials are comparable.
- NOT a fix for a wrong base LR: find the base LR first (LR range test in [[debugging-neural-network-training]]), then add the schedule.

## Setup & code
```bash
pip install torch
```
Warmup + cosine, one-cycle and WSD on the same 1000-step budget (torch 2.13 CPU; LR printed at steps 0, 50, 100, 500, 900, 999):
```python
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, OneCycleLR, LambdaLR

def lrs(make_sched, steps=1000):
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1e-3)
    sched = make_sched(opt)
    out = []
    for _ in range(steps):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()          # once per optimizer step, after opt.step()
    return out

total, warmup = 1000, 100
warm_cos = lrs(lambda o: SequentialLR(o, [
    LinearLR(o, start_factor=0.01, total_iters=warmup),
    CosineAnnealingLR(o, T_max=total - warmup, eta_min=1e-5),
], milestones=[warmup]))
one = lrs(lambda o: OneCycleLR(o, max_lr=1e-3, total_steps=total))

def wsd(step, warm=100, decay_start=800, total=1000):   # multiplier on the base LR
    if step < warm:
        return (step + 1) / warm
    if step < decay_start:
        return 1.0
    return max(0.0, (total - step) / (total - decay_start))
w = lrs(lambda o: LambdaLR(o, wsd))

for name, s in [("warmup+cosine", warm_cos), ("one-cycle", one), ("WSD", w)]:
    print(f"{name:14s}", " ".join(f"{s[i]:.1e}" for i in (0, 50, 100, 500, 900, 999)))
# warmup+cosine  1.0e-05 5.0e-04 1.0e-03 5.9e-04 4.0e-05 1.0e-05
# one-cycle      4.0e-05 1.0e-04 2.8e-04 8.1e-04 4.9e-05 4.0e-09
# WSD            1.0e-05 5.1e-04 1.0e-03 1.0e-03 5.0e-04 5.0e-06
```
ReduceLROnPlateau steps once per epoch with the validation metric:
```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
plateau = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
for epoch, val_loss in enumerate([1.0, 0.8, 0.79, 0.79, 0.79, 0.79, 0.6]):
    plateau.step(val_loss)
    print(epoch, opt.param_groups[0]["lr"])   # 0.1 until epoch 5, then 0.05
```
Hugging Face equivalent: `transformers.get_cosine_schedule_with_warmup(opt, num_warmup_steps, num_training_steps)`, or `lr_scheduler_type="cosine"` plus `warmup_ratio` in `TrainingArguments`.

## Choosing / trade-offs
- **Cosine vs linear decay.** Both work; differences are small next to getting the peak LR right. Pick one and tune the peak.
- **Cosine vs WSD.** Cosine needs the total length fixed; WSD can be extended and decayed from any checkpoint, at the cost of one more hyperparameter (decay fraction).
- **One-cycle vs warmup+cosine.** One-cycle spends a longer time ramping up and ends lower; good for short fixed-epoch vision runs. Warmup + cosine is the more common default elsewhere.
- **Scheduled vs reactive (plateau).** Scheduled runs are reproducible and comparable; plateau reacts to the data but depends on noisy validation and its patience setting.
- **Warmup length.** Too short risks early divergence; too long wastes steps at low LR. Larger batches and higher peak LRs need longer warmup.

## Gotchas
- Call `scheduler.step()` after `optimizer.step()`. Calling it first skips the first LR value and PyTorch warns about it.
- Using `OneCycleLR` or `CosineAnnealingLR` with a step count that does not match the real number of optimizer steps (gradient accumulation, `drop_last`, multiple GPUs) ends the schedule early or never finishes decaying. Compute `total_steps = epochs * ceil(len(loader) / accum_steps)`.
- Resuming from a checkpoint: save and load `scheduler.state_dict()`; restarting a scheduler from zero re-runs warmup at the wrong point.
- `OneCycleLR` overrides the optimizer's LR (it uses `max_lr`) and also cycles momentum by default (`cycle_momentum=True`); with Adam it changes β1.
- `ReduceLROnPlateau` must be stepped with the metric, and only once per evaluation; stepping it every batch with training loss makes it fire constantly.
- Per-group LRs (e.g. backbone vs head) are each scaled by the schedule; the ratio between groups is preserved.

## Related
- [[optimizers]] - the optimizer whose LR is being scheduled.
- [[deep-learning-training]] - full training loop with warmup + cosine and clipping.
- [[debugging-neural-network-training]] - LR range test to find the peak LR.
- [[fine-tuning-and-peft]] - typical warmup/decay settings for fine-tuning.
- [[pretraining-and-scaling-laws]] - why WSD is used for long pretraining runs.

## References
- PyTorch, How to adjust learning rate: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
- Loshchilov & Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts (cosine): https://arxiv.org/abs/1608.03983
- Smith, A disciplined approach to neural network hyper-parameters (one-cycle): https://arxiv.org/abs/1803.09820
- Goyal et al., Accurate, Large Minibatch SGD (linear scaling + warmup): https://arxiv.org/abs/1706.02677
- Hu et al., MiniCPM (warmup-stable-decay schedule): https://arxiv.org/abs/2404.06395
