---
title: Focal loss from scratch in NumPy (alpha/gamma, hand-derived gradient, class imbalance)
category: concepts
tags: [focal-loss, class-imbalance, retinanet, object-detection, binary-cross-entropy, loss-function, numpy, from-scratch]
use_cases:
  - "train a detector or classifier where 99% of examples are easy background"
  - "understand what alpha and gamma in focal loss actually do before tuning them"
  - "decide between focal loss, class weights and resampling for an imbalanced dataset"
status: draft
last_verified: 2026-10-06
sources:
  - https://arxiv.org/abs/1708.02002
  - https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html
---

# Focal loss from scratch in NumPy (alpha/gamma, hand-derived gradient, class imbalance)

## Summary
Focal loss is binary cross-entropy multiplied by `(1 - p_t)^gamma`, so examples the model already gets right contribute almost nothing. It was introduced for RetinaNet (Lin et al. 2017). One-stage detectors score ~100k anchor boxes per image and almost all of them are easy background, so plain cross-entropy is dominated by those easy negatives. This file implements the loss and its gradient by hand, checks the gradient numerically, matches `torchvision.ops.sigmoid_focal_loss` exactly, and shows on a toy problem what the loss changes and what it does not.

## Key concepts
- **p_t.** The model's probability for the true class: `p` if `y = 1`, `1 - p` if `y = 0`. Cross-entropy is `-log(p_t)`.
- **Modulating factor `(1 - p_t)^gamma`.** At `gamma = 2` an example with `p_t = 0.9` is down-weighted 100x and one with `p_t = 0.99` 10,000x. Hard examples (`p_t` near 0) keep almost their full CE loss. `gamma = 0` gives plain BCE back.
- **alpha.** A fixed class weight: `alpha` for positives, `1 - alpha` for negatives. RetinaNet uses `alpha = 0.25, gamma = 2`. That looks backwards, since it down-weights the rare class, but gamma already removes so much negative loss that the positives need less help.
- **Prior bias init.** RetinaNet starts the final logit bias at `log(pi / (1 - pi))` with `pi = 0.01`, so every anchor starts as "probably background". Without it the huge number of negatives gives a large, unstable loss on the first iterations.
- **Gradient.** `dFL/dz = alpha_t * [-gamma (1-p_t)^(gamma-1) CE - (1-p_t)^gamma / p_t] * s * p(1-p)` with `s = +1` for positives and `-1` for negatives. Autograd does this for you. Deriving it once shows that easy examples also get a near-zero gradient, not just a near-zero loss.

## When to use / scenarios
- Dense object detection and segmentation with extreme foreground/background imbalance: RetinaNet, FCOS and many YOLO variants use focal loss or a relative of it ([[object-detection]], [[segmentation]]).
- Pixel-wise tasks with tiny positive regions (lesions, cracks, defects), often combined with Dice loss.
- Classifiers with many easy negatives *and* a model flexible enough that easy examples would otherwise drive training.
- Not the first fix for ordinary tabular imbalance. Use class weights, a tuned threshold and calibration first ([[imbalanced-data]], [[smote-from-scratch]]). The toy result below shows why.

## Setup & code
NumPy only. `torch`/`torchvision` are used only as an optional cross-check.

```python
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def focal_loss(logits, y, alpha=0.25, gamma=2.0):
    """Binary sigmoid focal loss (Lin et al. 2017), per element. y in {0,1}.
    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)."""
    p = sigmoid(logits)
    p_t = np.where(y == 1, p, 1 - p)
    a_t = np.where(y == 1, alpha, 1 - alpha) if alpha is not None else 1.0
    ce = np.logaddexp(0, -logits) * y + np.logaddexp(0, logits) * (1 - y)   # stable -log(p_t)
    return a_t * (1 - p_t) ** gamma * ce


def focal_grad(logits, y, alpha=0.25, gamma=2.0):
    """d FL / d logit, derived by hand."""
    p = sigmoid(logits)
    p_t = np.where(y == 1, p, 1 - p)
    a_t = np.where(y == 1, alpha, 1 - alpha) if alpha is not None else 1.0
    s = np.where(y == 1, 1.0, -1.0)                     # d p_t / d z = s * p * (1 - p)
    ce = -np.log(np.clip(p_t, 1e-12, None))
    dpt = -gamma * (1 - p_t) ** (gamma - 1) * ce - (1 - p_t) ** gamma / np.clip(p_t, 1e-12, None)
    return a_t * dpt * s * p * (1 - p)


# 1) Gradient check against central differences
rng = np.random.default_rng(0)
z, y = rng.normal(0, 3, 8), rng.integers(0, 2, 8)
eps = 1e-6
num = (focal_loss(z + eps, y) - focal_loss(z - eps, y)) / (2 * eps)
print(f"max |analytic - numeric| grad: {np.abs(focal_grad(z, y) - num).max():.2e}")

# 2) How much an easy example is down-weighted vs cross-entropy
for p_t in [0.5, 0.9, 0.99]:
    zz = np.log(p_t / (1 - p_t))
    ce = focal_loss(np.array([zz]), np.array([1]), alpha=None, gamma=0)[0]
    fl = focal_loss(np.array([zz]), np.array([1]), alpha=None, gamma=2)[0]
    print(f"p_t={p_t:<5} CE={ce:.4f}  FL(g=2)={fl:.6f}  ratio={ce / fl:,.0f}x")

# 3) Imbalanced toy problem: 1% positives, logistic model, BCE vs focal
n, d = 20000, 2
X = rng.normal(0, 1, (n, d))
y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.6, n) > 2.6).astype(float)
print(f"positives: {y.mean():.2%}")


def train(loss_grad, steps=2000, lr=0.5):
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        g = loss_grad(X @ w + b, y)
        w -= lr * X.T @ g / n
        b -= lr * g.mean()
    return w, b


def share_from_easy_negatives(loss_fn, w, b):
    z = X @ w + b
    per = loss_fn(z, y)
    easy = (y == 0) & (sigmoid(z) < 0.1)
    return per[easy].sum() / per.sum(), easy.mean()


bce = lambda z, y: focal_loss(z, y, alpha=None, gamma=0)
bce_g = lambda z, y: focal_grad(z, y, alpha=None, gamma=0)
fl = lambda z, y: focal_loss(z, y, alpha=0.25, gamma=2)
fl_g = lambda z, y: focal_grad(z, y, alpha=0.25, gamma=2)

w0, b0 = np.zeros(d), np.log(0.01 / 0.99)          # prior-init bias, as in RetinaNet
s_bce, frac = share_from_easy_negatives(bce, w0 + np.array([1.0, 0.5]), b0)
s_fl, _ = share_from_easy_negatives(fl, w0 + np.array([1.0, 0.5]), b0)
print(f"easy negatives are {frac:.0%} of data; share of total loss: BCE {s_bce:.0%}, focal {s_fl:.0%}")


def recall_at_precision(z, target=0.5):
    order = np.argsort(-z)
    tp = np.cumsum(y[order])
    prec, rec = tp / np.arange(1, n + 1), tp / y.sum()
    ok = prec >= target
    return rec[ok].max() if ok.any() else 0.0


for name, g in [("BCE", bce_g), ("focal", fl_g)]:
    w, b = train(g)
    z = X @ w + b
    print(f"{name:5} mean p(pos)={sigmoid(z)[y == 1].mean():.2f}  recall@precision>=0.5: {recall_at_precision(z):.2f}")

# 4) Cross-check against torchvision
try:
    import torch
    from torchvision.ops import sigmoid_focal_loss
    zt, yt = torch.tensor(z[:1000]), torch.tensor(y[:1000])
    ref = sigmoid_focal_loss(zt, yt, alpha=0.25, gamma=2, reduction="sum").item()
    print(f"ours {focal_loss(z[:1000], y[:1000]).sum():.6f}  torchvision {ref:.6f}")
except ImportError:
    pass

assert np.allclose(focal_loss(z, y, alpha=None, gamma=0), bce(z, y))   # gamma=0, no alpha = BCE
```

Output (Python 3.14, NumPy 2.5, torchvision 0.28):
```
max |analytic - numeric| grad: 3.98e-10
p_t=0.5   CE=0.6931  FL(g=2)=0.173287  ratio=4x
p_t=0.9   CE=0.1054  FL(g=2)=0.001054  ratio=100x
p_t=0.99  CE=0.0101  FL(g=2)=0.000001  ratio=10,000x
positives: 2.02%
easy negatives are 97% of data; share of total loss: BCE 23%, focal 0%
BCE   mean p(pos)=0.35  recall@precision>=0.5: 0.56
focal mean p(pos)=0.33  recall@precision>=0.5: 0.56
ours 3.960841  torchvision 3.960841
```

How to read it:
- The hand-derived gradient matches finite differences to 4e-10, and the loss matches torchvision to six decimals.
- `(1 - p_t)^2` turns a 10x drop in CE (p_t 0.5 to 0.99) into a 100,000x drop. That is the whole mechanism.
- With a reasonable model, easy negatives (97% of rows, `p < 0.1`) still produce 23% of the BCE loss. Under focal loss they produce under 1%, so the gradient budget goes to the hard and positive examples.
- **The ranking did not improve.** Both losses give the same recall at 50% precision. A 3-parameter logistic model has no spare capacity for easy examples to "waste", so reweighting them barely moves the decision boundary. Focal loss pays off when a large network trained on millions of easy anchors would otherwise spend its updates on them. That is the RetinaNet setting, not this one.
- Focal-trained probabilities are not calibrated. The loss deliberately stops pushing confident predictions further, so if you need probabilities, recalibrate afterwards ([[platt-and-isotonic-calibration-from-scratch]]).

## Choosing / trade-offs
- **Focal vs class-weighted BCE.** Class weights rescale whole classes. Focal reweights per example by difficulty, so it also handles easy positives and hard negatives. Try weighted BCE first. Switch to focal when there is a flood of easy negatives (dense detection, segmentation) and the model is large.
- **gamma.** 2 is the paper default; the RetinaNet ablation found 0.5–5 workable. Higher gamma ignores more of the data, so noisy labels (hard because they are wrong) get relatively more weight.
- **alpha.** Tune it together with gamma, not alone. With `gamma = 2`, `alpha = 0.25` worked best in the paper. Without gamma, `alpha` approaching the inverse class frequency is the usual class-weighting choice.
- **Variants.** Quality focal loss and varifocal loss (soft IoU targets for detection), and the multi-class softmax form `-(1-p_t)^gamma log p_t` applied to the true class. Use the version your detection framework ships rather than re-implementing it.

## Gotchas
- Normalise by the number of positive anchors, not by the total, as RetinaNet does. With `reduction="mean"` over 100k anchors the loss becomes tiny and the learning rate needs retuning.
- Forgetting the prior bias init gives a huge loss and divergence in the first iterations of a dense detector.
- Computing `log(sigmoid(z))` directly overflows for large `|z|`. Use `logaddexp` / `BCEWithLogits`-style stable forms as above.
- Focal loss changes the meaning of the output scores. Thresholds tuned under BCE do not carry over.
- Under label noise, focal loss amplifies mislabeled "hard" examples. Clean the labels or lower gamma.

## Related
- [[loss-functions]] - where focal loss sits among classification and regression losses.
- [[object-detection]] - one-stage detectors are the main users of focal loss.
- [[imbalanced-data]] - resampling, class weights and threshold tuning, the cheaper first options.
- [[nms-iou-and-map-from-scratch]] - how detectors trained with focal loss are evaluated.
- [[platt-and-isotonic-calibration-from-scratch]] - recalibrating the deliberately under-confident scores.

## References
- Lin, Goyal, Girshick, He, Dollár (2017), "Focal Loss for Dense Object Detection": https://arxiv.org/abs/1708.02002
- torchvision `sigmoid_focal_loss`: https://pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html
