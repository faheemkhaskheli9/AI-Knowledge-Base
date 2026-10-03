---
title: Image classification
category: cv
tags: [image-classification, timm, torchvision, transfer-learning, fine-tuning, clip, zero-shot]
use_cases:
  - "classify product photos into categories for an online store"
  - "detect defective vs good parts on a production line from images"
  - "classify medical or crop-disease images with a small labelled dataset"
  - "build an image classifier with only a few dozen examples per class"
  - "zero-shot tag images without any training data"
status: draft
last_verified: 2026-10-03
sources:
  - https://pytorch.org/vision/stable/models.html
  - https://huggingface.co/docs/timm/index
  - https://huggingface.co/docs/transformers/tasks/image_classification
---

# Image classification

## Summary
Image classification assigns one (or several, for multi-label) labels to a whole image. The standard recipe is transfer learning: start from a backbone pretrained on large data (ResNet/ConvNeXt/EfficientNet/ViT), replace the head, fine-tune on your labelled images. With no labels at all, CLIP-style zero-shot or a vision-language model can classify by text prompts.

## Key concepts
- Transfer learning: freeze the backbone and train the head (fast, few images), then optionally unfreeze and fine-tune with a small learning rate.
- Pretrained model libraries: `torchvision.models`, `timm` (large collection with consistent API), Hugging Face `transformers`.
- Preprocessing must match the pretrained model (resize, normalisation); use the model's own transform config.
- Augmentation (flips, crops, colour jitter, RandAugment) is the main regulariser on small data; keep it label-preserving for your domain.
- Single-label (softmax, cross-entropy) vs multi-label (sigmoid, BCE).
- Metrics: accuracy only when balanced; otherwise per-class precision/recall, confusion matrix; calibrate confidence.
- Zero-shot / few-shot: CLIP-like embeddings plus text prompts or a logistic regression on embeddings (see [[vision-language-models]]).

## When to use / scenarios
- Retail: product categorisation, attribute tagging. Agriculture: crop disease. Manufacturing: pass/fail visual inspection ([[manufacturing-iot]]).
- Healthcare imaging: triage support only, with clinical validation and regulatory review.
- Content moderation, document type routing (invoice vs ID vs receipt before [[ocr]]).
- Use [[object-detection]] if you need where objects are or several instances; [[segmentation]] for pixel-level shapes; [[anomaly-detection]] when defect examples are rare.

## Setup & code
```bash
pip install torch torchvision timm
```
```python
import timm, torch
from torch import nn

# num_classes replaces the head; pretrained=True downloads weights
model = timm.create_model("resnet50", pretrained=True, num_classes=5)
cfg = timm.data.resolve_data_config({}, model=model)
transform = timm.data.create_transform(**cfg)       # use for train/val images (add augmentation for train)

for p in model.parameters():                        # stage 1: train head only
    p.requires_grad = False
for p in model.get_classifier().parameters():
    p.requires_grad = True

opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
x = torch.randn(4, 3, 224, 224)
loss = nn.CrossEntropyLoss()(model(x), torch.tensor([0, 1, 2, 3]))
loss.backward(); opt.step()
```
Wrap with a standard loop (see [[pytorch-basics]]); then unfreeze all layers at lr around 1e-4 to 1e-5.

## Choosing / trade-offs
- Small data (under ~1k images): frozen backbone or embeddings + logistic regression, strong augmentation.
- Medium data: fine-tune a ConvNeXt/EfficientNet/ViT-small; CNNs are cheaper on edge devices.
- Latency/edge: MobileNet/EfficientNet-lite, quantisation, ONNX export ([[edge-on-device]]).
- Zero-shot VLM/CLIP: no training, but lower accuracy on fine-grained or domain-specific classes; use as a baseline or for bootstrapping labels.
- Image size: higher resolution helps small defects but costs compute quadratically.

## Gotchas
- Train/validation leakage from near-duplicate images or frames of the same scene; split by source (patient, product, session).
- Dataset shift between lab photos and real deployment (lighting, camera); validate on deployment-like data.
- Wrong normalisation or channel order silently ruins accuracy.
- Class imbalance: weighted loss or sampling; do not trust accuracy.
- Models learn shortcuts (watermarks, backgrounds, scanner artefacts); inspect with Grad-CAM.
- Softmax confidence is overconfident on out-of-distribution inputs; add a reject class or threshold.
- Check pretrained weight licences before commercial use.

## Related
- [[pytorch-basics]] - training loop.
- [[object-detection]] - when location matters.
- [[vision-language-models]] - zero-shot and captioning alternatives.
- [[data-labeling-and-synthetic-data]] - getting labels.
- [[edge-on-device]] - deploying small classifiers.

## References
- https://pytorch.org/vision/stable/models.html
- https://huggingface.co/docs/timm/index
- https://huggingface.co/docs/transformers/tasks/image_classification
