---
title: Transfer learning, domain adaptation and continual learning
category: concepts
tags: [transfer-learning, domain-adaptation, distribution-shift, covariate-shift, continual-learning, catastrophic-forgetting, fine-tuning, feature-extraction, pytorch]
use_cases:
  - "train an accurate model with only a few hundred labelled examples by starting from a pretrained one"
  - "a model trained on one factory, hospital or region performs badly on another"
  - "adapt a model to a new domain where labels are scarce or missing"
  - "keep updating a model on new data without it forgetting old classes"
status: draft
last_verified: 2026-10-04
sources:
  - https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
  - https://docs.pytorch.org/vision/stable/models.html
  - https://arxiv.org/abs/1505.07818
  - https://arxiv.org/abs/1612.00796
---

# Transfer learning, domain adaptation and continual learning

## Summary
Transfer learning reuses a model trained on a large source task as the
starting point for your task, so a few hundred labels can be enough. Domain
adaptation handles the case where the target data differs from the training
data (new camera, new hospital, new country) and labels there are scarce.
Continual learning updates a model over time on new data or classes without
"catastrophic forgetting" of what it learned before.

## Key concepts
- Feature extraction: freeze the pretrained backbone, train only a new head.
  Fast, needs little data, cannot adapt low-level features.
- Fine-tuning: unfreeze some or all layers with a small learning rate (often
  lower for early layers: discriminative learning rates). Better with more data.
- Source/target similarity decides how much to unfreeze: similar domain and
  little data = freeze more; different domain and more data = unfreeze more.
- Distribution shift types: covariate shift (P(x) changes), label shift
  (class mix changes), concept shift (P(y|x) changes).
- Unsupervised domain adaptation: labelled source + unlabelled target.
  Methods align features (DANN adversarial training, MMD/CORAL losses),
  self-train on confident target pseudo-labels, or pretrain further on
  target data with self-supervision.
- Importance weighting for covariate shift: reweight source rows by
  P_target(x)/P_source(x), estimated with a domain classifier.
- Catastrophic forgetting: fine-tuning on new data overwrites old knowledge.
  Mitigations: replay a buffer of old examples, regularise important weights
  (EWC), freeze/expand parameters, or add adapters per task.

## When to use / scenarios
- Small labelled image, text or audio datasets: always start from a
  pretrained model ([[image-classification]], [[nlp-classic-tasks]]).
- Manufacturing defect detection moved to a new production line or camera;
  medical imaging model taken to another hospital's scanners; retail demand
  model rolled out to a new country.
- Sim-to-real: train on synthetic/rendered data, adapt to real images.
- Model must learn new product categories monthly while keeping old ones:
  continual learning with replay.
- NOT for: gradual drift in a live stream (see
  [[online-learning-and-concept-drift]]), or LLM behaviour changes (see
  [[fine-tuning-and-peft]] and [[rag-basics]]).

## Setup & code
```bash
pip install torch torchvision
```
Feature extraction then fine-tuning with torchvision (downloads ImageNet weights):
```python
import torch, torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
preprocess = weights.transforms()          # use the same preprocessing as pretraining

for p in model.parameters():               # stage 1: freeze the backbone
    p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 3)   # new head for 3 classes (trainable)
opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
# ... train the head for a few epochs ...

for p in model.layer4.parameters():        # stage 2: unfreeze the last block
    p.requires_grad = True
opt = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-4},   # smaller LR for pretrained layers
    {"params": model.fc.parameters(), "lr": 1e-3},
])

x = torch.randn(2, 3, 224, 224)
print(model(x).shape)                       # torch.Size([2, 3])
```
Detecting covariate shift with a domain classifier (any tabular features):
```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
X_src, X_tgt = rng.normal(0, 1, (2000, 5)), rng.normal(0.5, 1, (2000, 5))
X = np.vstack([X_src, X_tgt]); d = np.r_[np.zeros(2000), np.ones(2000)]
auc = cross_val_score(HistGradientBoostingClassifier(), X, d, cv=5, scoring="roc_auc").mean()
print(f"domain AUC {auc:.2f}")   # ~0.5 = no shift; well above 0.5 = shift
```
Fit the domain classifier on all rows, then weight source rows by p/(1-p)
(clipped) when retraining the task model.

## Choosing / trade-offs
- Few labels, similar domain: frozen backbone + linear head (or a
  [[metric-learning-and-few-shot]] approach).
- Moderate labels: fine-tune the top blocks, then everything with a low LR.
- Target unlabelled: label a small target sample first if at all possible;
  a few hundred target labels usually beat clever unsupervised adaptation.
  Otherwise self-supervised pretraining on target data or pseudo-labelling
  ([[semi-supervised-and-active-learning]]).
- Large model, limited GPU: parameter-efficient adapters/LoRA also work for
  vision and audio models ([[fine-tuning-and-peft]]).
- Continual learning: a replay buffer is the simple, strong baseline; EWC and
  architecture methods add complexity for modest gains in most products.

## Gotchas
- Using different preprocessing (resize, normalisation, tokenizer) than the
  pretrained model expects silently degrades accuracy.
- Too high a learning rate when unfreezing wipes out pretrained features in a
  few steps; use small LRs and warmup.
- BatchNorm with tiny batches during fine-tuning gives noisy statistics;
  keep BN layers in eval mode or freeze them.
- Validating on a random split of the source domain hides the shift; always
  hold out data from the actual target domain.
- Pretrained weight licences and dataset terms carry over to your model
  ([[model-licenses]]).
- Evaluating continual learning on new classes only hides forgetting; report
  accuracy on old and new tasks.

## Related
- [[neural-network-fundamentals]] - why learned representations transfer.
- [[image-classification]] - the standard transfer-learning recipe for images.
- [[fine-tuning-and-peft]] - LoRA and adapters for large models.
- [[autoencoders-and-self-supervised-learning]] - pretraining on unlabelled target data.
- [[online-learning-and-concept-drift]] - detecting and reacting to gradual drift.
- [[metric-learning-and-few-shot]] - very few labels per class.
- [[manufacturing-iot]], [[healthcare]] - common domain-shift settings.

## References
- PyTorch transfer learning tutorial: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- torchvision pretrained models: https://docs.pytorch.org/vision/stable/models.html
- Ganin et al., "Domain-Adversarial Training of Neural Networks": https://arxiv.org/abs/1505.07818
- Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (EWC): https://arxiv.org/abs/1612.00796
