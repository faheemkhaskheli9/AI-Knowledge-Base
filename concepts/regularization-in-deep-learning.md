---
title: Regularization in deep learning (dropout, weight decay, early stopping)
category: concepts
tags: [deep-learning, regularization, overfitting, dropout, weight-decay, l2, early-stopping, stochastic-depth, label-smoothing, pytorch]
use_cases:
  - "stop a neural network from overfitting a small training set"
  - "fix validation loss that rises while training loss keeps falling"
  - "add early stopping and keep the best checkpoint"
  - "choose dropout rate and weight decay for an MLP, CNN or transformer"
  - "understand why a model behaves differently in train() and eval() mode"
status: draft
last_verified: 2026-10-04
sources:
  - https://jmlr.org/papers/v15/srivastava14a.html
  - https://arxiv.org/abs/1711.05101
  - https://arxiv.org/abs/1603.09382
  - https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
  - https://www.deeplearningbook.org/contents/regularization.html
---

# Regularization in deep learning (dropout, weight decay, early stopping)

## Summary
Regularization is anything that trades a little training fit for better performance on unseen data. Neural networks have enough capacity to memorise small datasets, so some regularization is almost always on. The main tools are **more or augmented data** (the strongest), **weight decay** (shrink weights toward zero), **dropout** (randomly zero activations during training), **early stopping** (stop when validation stops improving and keep the best checkpoint), plus architecture-level variants such as stochastic depth and label smoothing. They stack, but each one adds a hyperparameter, so add them in response to a measured train/validation gap.

## Key concepts
- **Diagnose first.** Overfitting = training loss falls while validation loss rises (gap grows). Underfitting = both high. Regularization helps only the first; for the second, add capacity or train longer ([[ml-fundamentals]]).
- **Weight decay.** Each step shrinks weights by `lr·λ·w`. With AdamW it is decoupled from the gradient; with SGD it equals an L2 penalty ([[optimizers]]). Typical: 0.01-0.1 (AdamW), 1e-4 to 5e-4 (SGD). Usually not applied to biases or norm parameters.
- **Dropout.** In training, each activation is zeroed with probability `p` and survivors are scaled by `1/(1−p)` (inverted dropout), so in eval mode the layer is the identity. Typical `p`: 0.1 in transformers, 0.2-0.5 in MLP heads. Discourages co-adapted features; acts like averaging many thinned networks.
- **Early stopping.** Track a validation metric each epoch; stop after `patience` epochs without improvement and restore the best weights. Limits how far the weights move from initialization; often the cheapest, most effective regularizer for small data.
- **Data augmentation.** Label-preserving transforms enlarge the effective dataset ([[data-augmentation]]); for images and audio it usually beats any penalty.
- **Others.** Label smoothing (soften one-hot targets, [[loss-functions]]), stochastic depth / DropPath (randomly skip residual blocks, standard in ViTs and ConvNeXt), mixup/cutmix, gradient noise from small batches, and smaller models. BatchNorm also adds mild noise ([[normalization-layers]]).

## When to use / scenarios
- Small tabular or sensor dataset (hundreds to a few thousand rows) with an MLP: early stopping first, then dropout 0.1-0.3 and weight decay. Compare against gradient boosting, which is often better here ([[gradient-boosting-tabular]]).
- Image classifier with a few thousand images: transfer learning plus augmentation before adding dropout ([[transfer-learning-and-domain-adaptation]]).
- Fine-tuning a pretrained transformer: keep its built-in dropout (often 0.1), weight decay ~0.01, few epochs; early stopping on validation.
- Training large models on huge datasets (one pass or less): little or no dropout; weight decay remains common.
- NOT for underfitting or label noise: noisy labels need cleaning or robust losses ([[label-noise-and-data-cleaning]]).

## Setup & code
```bash
pip install torch
```
Overfit a 256-unit MLP on 200 noisy samples, then compare regularizers (torch 2.13 CPU; validation on 2000 held-out samples):
```python
import copy
import torch
from torch import nn

torch.manual_seed(0)
def data(n):
    X = torch.randn(n, 10)
    y = ((X[:, 0] * X[:, 1] + 0.5 * X[:, 2]) + 0.5 * torch.randn(n) > 0).float()
    return X, y
Xtr, ytr = data(200)
Xva, yva = data(2000)

def run(p_drop=0.0, wd=0.0, patience=None, epochs=300):
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(10, 256), nn.ReLU(), nn.Dropout(p_drop),
                          nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p_drop),
                          nn.Linear(256, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for i in range(0, 200, 32):
            loss = loss_fn(model(Xtr[i:i + 32]).squeeze(1), ytr[i:i + 32])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva).squeeze(1), yva).item()
        if patience is not None:                       # early stopping
            if val < best - 1e-4:
                best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    model.load_state_dict(best_state)  # restore the best epoch
                    break
    model.eval()
    with torch.no_grad():
        tr = ((model(Xtr).squeeze(1) > 0).float() == ytr).float().mean().item()
        logits = model(Xva).squeeze(1)
        va = ((logits > 0).float() == yva).float().mean().item()
    return tr, va, loss_fn(logits, yva).item(), epoch + 1

for name, kw in [("none", {}), ("dropout 0.3", {"p_drop": 0.3}),
                 ("weight decay 0.1", {"wd": 0.1}),
                 ("early stop (patience 10)", {"patience": 10}),
                 ("dropout + early stop", {"p_drop": 0.3, "patience": 10})]:
    print(name, run(**kw))
```
Observed:

| Setting | Train acc | Val acc | Val loss | Epochs |
|---|---|---|---|---|
| none | 1.00 | 0.76 | 1.71 | 300 |
| dropout 0.3 | 1.00 | 0.77 | 1.66 | 300 |
| weight decay 0.1 | 1.00 | 0.76 | 1.48 | 300 |
| early stop (patience 10) | 0.94 | 0.74 | 0.55 | 22 |
| dropout + early stop | 0.92 | 0.73 | 0.56 | 28 |

Accuracy barely moves, but the unregularized model's validation loss is three times higher: it is confidently wrong, which matters for any use of predicted probabilities. Early stopping fixed that here and trained 10x fewer epochs; dropout and weight decay alone at 300 epochs did not. Results depend on the data, so measure on yours.

Dropout is active only in training mode:
```python
d = nn.Dropout(0.5)
x = torch.ones(8)
d.train(); print(d(x))   # tensor([2., 0., 0., 2., 2., 2., 2., 0.])  survivors scaled by 1/(1-p)
d.eval();  print(d(x))   # tensor([1., 1., 1., 1., 1., 1., 1., 1.])
```
Stochastic depth for residual blocks is available as `torchvision.ops.StochasticDepth(p, mode="row")`; in `timm` models it is the `drop_path_rate` argument.

## Choosing / trade-offs
- **Order to try.** More data / augmentation, then early stopping, then weight decay, then dropout. Use transfer learning when a pretrained model exists.
- **Dropout vs BatchNorm.** Combining them can hurt because dropout shifts activation variance between train and eval. In CNNs with BatchNorm, dropout is usually only in the classifier head.
- **Early stopping vs fixed schedule.** Early stopping saves compute and needs no schedule tuning, but stops on a noisy signal; with a decaying LR schedule ([[learning-rate-schedules]]) many recipes instead train to the end and keep the best checkpoint.
- **Weight decay strength.** Too high underfits and slows training; tune on a log scale (1e-5 to 1e-1) together with the LR.
- **Monitoring metric.** Early-stop on validation loss to protect calibration; on the business metric (F1, AUC) when that is what matters, with a larger patience because it is noisier.

## Gotchas
- Forgetting `model.eval()` at validation and inference keeps dropout on: predictions become random and accuracy drops. Forgetting `model.train()` afterwards silently disables it.
- Early stopping on the test set leaks it into model selection; use a separate validation split ([[model-evaluation-and-metrics]]).
- Keeping only the last epoch's weights after early stopping defeats the purpose; restore (or save) the best checkpoint.
- `torch.optim.Adam(weight_decay=...)` is L2, not decoupled decay; use AdamW ([[optimizers]]).
- Dropout on the input layer of tabular data, or after the final layer, usually hurts. Put it between hidden layers.
- MC dropout (dropout on at inference to estimate uncertainty) is a deliberate exception; see [[bayesian-deep-learning-and-uncertainty]].

## Related
- [[ml-fundamentals]] - overfitting, bias-variance and train/validation/test splits.
- [[optimizers]] - weight decay as implemented by AdamW vs SGD.
- [[data-augmentation]] - the strongest regularizer when it applies.
- [[deep-learning-training]] - where regularization fits in the training recipe.
- [[loss-functions]] - label smoothing.
- [[normalization-layers]] - BatchNorm's regularizing noise and its interaction with dropout.

## References
- Srivastava et al., Dropout: A Simple Way to Prevent Neural Networks from Overfitting: https://jmlr.org/papers/v15/srivastava14a.html
- Loshchilov & Hutter, Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
- Huang et al., Deep Networks with Stochastic Depth: https://arxiv.org/abs/1603.09382
- PyTorch `nn.Dropout`: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
- Goodfellow, Bengio & Courville, Deep Learning, ch. 7 Regularization: https://www.deeplearningbook.org/contents/regularization.html
