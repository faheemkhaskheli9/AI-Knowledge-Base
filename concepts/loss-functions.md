---
title: Loss functions (regression, classification, ranking, distributions)
category: concepts
tags: [loss-function, cross-entropy, mse, mae, huber, quantile-loss, focal-loss, label-smoothing, bce, kl-divergence, contrastive-loss, pytorch]
use_cases:
  - "pick the right loss for a regression, binary, multi-class or multi-label model"
  - "make a regression model robust to outliers in the target"
  - "predict an interval or a percentile instead of the mean (quantile regression)"
  - "train a detector or classifier on heavily imbalanced classes with focal or weighted loss"
  - "match a model's output layer to the loss so training is numerically stable"
status: draft
last_verified: 2026-10-04
sources:
  - https://pytorch.org/docs/stable/nn.html#loss-functions
  - https://arxiv.org/abs/1708.02002
  - https://arxiv.org/abs/1512.00567
  - https://scikit-learn.org/stable/modules/model_evaluation.html
---

# Loss functions (regression, classification, ranking, distributions)

## Summary
The loss is the number gradient descent minimises, so it decides what the model actually learns: the mean (MSE), the median (MAE), a percentile (quantile/pinball), calibrated class probabilities (cross-entropy) or a ranking of pairs (contrastive, margin). Picking the loss means stating what a prediction error costs in the problem, then pairing the loss with the right output layer (raw logits, not softmax/sigmoid) so it stays numerically stable. The evaluation metric and the training loss can differ; the loss must be differentiable, the metric need not be.

## Key concepts
- **Loss = statistical estimate.** Minimising MSE predicts the conditional *mean*; MAE predicts the *median*; pinball loss at τ predicts the τ-*quantile*; cross-entropy predicts *probabilities*. Choose by which statistic you need.
- **Regression losses.**
  - `MSELoss` (L2): smooth, punishes large errors quadratically; sensitive to outlier targets.
  - `L1Loss` (MAE): robust to outliers; gradient has constant magnitude, so it converges less smoothly near zero.
  - `HuberLoss` / `SmoothL1Loss`: quadratic near zero, linear beyond `delta`; the usual robust default. Box regression in detectors uses Smooth L1.
  - Quantile (pinball) loss: `max(τ·e, (τ−1)·e)` with `e = y − ŷ`; train τ = 0.1 and 0.9 for an 80% interval.
  - Poisson / Tweedie / Gamma (`PoissonNLLLoss`, GBM objectives) for counts and non-negative skewed amounts.
- **Classification losses.**
  - Binary or multi-label: `BCEWithLogitsLoss` on raw logits (sigmoid inside, log-sum-exp stable). `pos_weight` up-weights the rare positive.
  - Multi-class (one label per sample): `CrossEntropyLoss` on raw logits with integer class targets. It already applies log-softmax.
  - `weight=` per class for imbalance; `label_smoothing=0.05-0.1` to reduce over-confidence.
  - **Focal loss** `−(1−p_t)^γ · log p_t` (γ≈2) down-weights easy examples; built for dense object detection with extreme foreground/background imbalance (`torchvision.ops.sigmoid_focal_loss`).
  - Hinge loss (SVMs) optimises the margin, not probabilities.
- **Distribution losses.** `KLDivLoss` (expects log-probs as input) for knowledge distillation and matching distributions; Gaussian NLL (`GaussianNLLLoss`) when the model predicts a mean *and* variance.
- **Similarity / ranking losses.** Contrastive, triplet (`TripletMarginLoss`) and InfoNCE learn embeddings (see [[metric-learning-and-few-shot]]); pairwise/listwise ranking losses in [[learning-to-rank]].
- **Reduction.** `reduction="mean"` (default) averages over the batch; use `"none"` to weight or mask per-sample losses (padding tokens, sample weights), then reduce yourself.

## When to use / scenarios
- House prices or demand with a few huge outliers: Huber, or MSE on `log1p(target)`.
- Delivery-time or stock-level planning that needs "90% of the time under X": quantile loss at τ=0.9.
- Insurance claim amounts or counts of events: Tweedie or Poisson objective.
- Fraud, churn, medical screening (rare positive): `BCEWithLogitsLoss(pos_weight=neg/pos)` or focal loss, then tune the threshold on validation (see [[imbalanced-data]]).
- Tagging an article with several topics: multi-label `BCEWithLogitsLoss`, one logit per tag (see [[multi-label-and-multi-task-learning]]).
- Distilling a big model into a small one: KL between temperature-softened teacher and student outputs (see [[knowledge-distillation-and-compression]]).
- NOT a custom loss when a reweighted standard one works; and NOT the loss to fix a bad metric choice; pick the metric first ([[model-evaluation-and-metrics]]).

## Setup & code
```bash
pip install torch
```
Same data, three regression losses, three different answers (mean, median, 90th percentile), plus the stable classification pairings:
```python
import torch
from torch import nn

torch.manual_seed(0)
# Skewed target: most values small, a few very large.
y = torch.cat([torch.randn(950) + 10, torch.randn(50) * 5 + 60])

def fit_constant(loss_fn, steps=3000, lr=0.5):
    c = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([c], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(c.expand_as(y), y).backward()
        opt.step()
    return c.item()

def pinball(tau):
    def f(pred, target):
        e = target - pred
        return torch.maximum(tau * e, (tau - 1) * e).mean()
    return f

print(f"MSE   -> {fit_constant(nn.MSELoss()):6.2f}  (mean   {y.mean():6.2f})")
print(f"L1    -> {fit_constant(nn.L1Loss()):6.2f}  (median {y.median():6.2f})")
print(f"Huber -> {fit_constant(nn.HuberLoss(delta=1.0)):6.2f}  (near median, outliers ignored)")
print(f"q0.9  -> {fit_constant(pinball(0.9)):6.2f}  (90th pct {y.quantile(0.9):6.2f})")

# Classification: always feed raw logits.
logits = torch.tensor([[2.0, -1.0, 0.5]])
target = torch.tensor([0])
print("CE  ", nn.CrossEntropyLoss(label_smoothing=0.1)(logits, target).item())

bin_logits = torch.tensor([3.0, -2.0, 0.1])
bin_target = torch.tensor([1.0, 0.0, 1.0])
print("BCE ", nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))(bin_logits, bin_target).item())

# Focal loss: easy examples (confident and right) contribute almost nothing.
from torchvision.ops import sigmoid_focal_loss
print("focal", sigmoid_focal_loss(bin_logits, bin_target, alpha=0.25, gamma=2.0, reduction="mean").item())
```
Expected shape of the output: MSE lands near the mean (~12.5), L1 and Huber near the median (~10), q0.9 near the 90th percentile.

## Choosing / trade-offs
- **MSE vs MAE vs Huber.** MSE when large errors really are much worse and targets are clean; MAE/Huber when targets have outliers or heavy tails. Huber's `delta` sets where "outlier" begins, in target units; scale targets or tune it.
- **Transform the target vs change the loss.** MSE on `log1p(y)` optimises relative error and is often simpler than a custom loss; remember it predicts the geometric, not arithmetic, mean when transformed back.
- **Class weights vs focal loss vs resampling.** Class weights are one line and usually enough; focal loss helps when easy negatives swamp the batch (detection, dense prediction); either way, the decision threshold matters more than the loss for the final metric.
- **Label smoothing.** Small gains in accuracy and calibration for large classifiers; hurts when you need sharp probabilities or do distillation from that model.
- **Point vs probabilistic output.** Gaussian NLL or quantile heads give uncertainty at a small cost; for guaranteed coverage add [[conformal-prediction-and-uncertainty]].

## Gotchas
- Applying `softmax` before `CrossEntropyLoss` or `sigmoid` before `BCEWithLogitsLoss` double-applies the activation: training is slow and gradients vanish. Pass raw logits.
- `CrossEntropyLoss` wants class indices of dtype `long` (or class probabilities of the same shape as logits); passing one-hot floats as indices silently errors or misbehaves.
- `BCELoss` (without logits) on probabilities is numerically unstable near 0/1; prefer `BCEWithLogitsLoss`.
- `KLDivLoss` expects `input` as log-probabilities and `target` as probabilities (unless `log_target=True`); swapping them gives a finite but meaningless number. Use `reduction="batchmean"` for the true KL.
- Sequence models: mask padding (`ignore_index=-100` in `CrossEntropyLoss`) or the model learns to predict padding.
- A falling loss with a flat metric usually means the loss and metric disagree (e.g. MSE vs MAPE, CE vs F1 at a fixed threshold); align them or tune the threshold.
- Summing vs averaging changes the effective learning rate when batch size changes.

## Related
- [[neural-network-fundamentals]] - where the loss sits in the forward/backward pass.
- [[model-evaluation-and-metrics]] - the metric the loss is a proxy for.
- [[imbalanced-data]] - weighting, resampling and threshold tuning around the loss.
- [[gradient-boosting-tabular]] - the same objectives (Huber, quantile, Tweedie) in GBM libraries.
- [[metric-learning-and-few-shot]] - contrastive and triplet losses for embeddings.
- [[knowledge-distillation-and-compression]] - KL loss on softened teacher outputs.

## References
- PyTorch loss functions: https://pytorch.org/docs/stable/nn.html#loss-functions
- Lin et al., Focal Loss for Dense Object Detection: https://arxiv.org/abs/1708.02002
- Szegedy et al., Rethinking the Inception Architecture (label smoothing): https://arxiv.org/abs/1512.00567
- scikit-learn metrics and scoring: https://scikit-learn.org/stable/modules/model_evaluation.html
