---
title: Data augmentation (images, text, audio, tabular, mixup, TTA)
category: concepts
tags: [data-augmentation, regularization, torchvision, transforms-v2, mixup, cutmix, randaugment, test-time-augmentation, specaugment, back-translation, deep-learning]
use_cases:
  - "improve an image classifier trained on only a few thousand photos"
  - "reduce overfitting when validation loss rises while training loss falls"
  - "augment images and their bounding boxes or masks consistently for detection/segmentation"
  - "make a speech or audio model robust to noise and recording conditions"
  - "squeeze extra accuracy at inference with test-time augmentation"
status: draft
last_verified: 2026-10-04
sources:
  - https://pytorch.org/vision/stable/transforms.html
  - https://pytorch.org/vision/stable/auto_examples/transforms/plot_cutmix_mixup.html
  - https://arxiv.org/abs/1710.09412
  - https://arxiv.org/abs/1905.04899
  - https://arxiv.org/abs/1909.13719
  - https://arxiv.org/abs/1904.08779
  - https://albumentations.ai/docs/
---

# Data augmentation (images, text, audio, tabular, mixup, TTA)

## Summary
Data augmentation creates new training examples by applying label-preserving changes (flip, crop, colour shift, noise, time-shift, paraphrase) to existing ones, so the model learns the invariances you care about instead of memorising the training set. It is usually the strongest single regulariser for vision and audio models and costs nothing at inference. Batch-level methods (mixup, cutmix) blend examples and their labels; test-time augmentation (TTA) averages predictions over augmented copies of the input.

## Key concepts
- **Label-preserving by domain.** An augmentation is valid only if a human would still give the same label. Horizontal flip is fine for cats, wrong for road signs with arrows or for text; 90° rotation is fine for satellite tiles, wrong for portraits.
- **Train only.** Random augmentation goes in the training pipeline; validation/test get only deterministic resize/normalise.
- **Image ops** (`torchvision.transforms.v2`): `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, `RandomRotation`, `RandomErasing`, `GaussianBlur`. Policy-based: `RandAugment` (2 params: number of ops, magnitude), `TrivialAugmentWide` (no params), `AutoAugment`.
- **Geometric targets.** For detection/segmentation the boxes, masks and keypoints must move with the image. `transforms.v2` does this when targets are wrapped as `tv_tensors.BoundingBoxes` / `Mask`; Albumentations does it with `bbox_params`.
- **Batch-level mixing.** Mixup blends two images and their one-hot labels with λ ~ Beta(α, α); CutMix pastes a patch from one image into another and mixes labels by area. Both need soft-label cross-entropy.
- **Audio.** Additive noise, time-shift, speed/pitch perturbation, room impulse responses; SpecAugment masks time and frequency bands on the spectrogram.
- **Text.** Back-translation, synonym swap, random deletion; for LLM-era work, paraphrasing with an LLM (see [[data-labeling-and-synthetic-data]]). Gains are smaller than in vision; pretrained models already carry most invariances.
- **Tabular.** Rarely helpful; SMOTE-style oversampling is resampling for imbalance ([[imbalanced-data]]), not invariance learning. Gaussian noise on features can regularise small neural nets.
- **TTA.** Predict on the original plus a few deterministic variants (flip, multi-crop) and average probabilities; typically a small gain for a k-fold inference cost.

## When to use / scenarios
- Small or medium image datasets (product photos, medical images, defect inspection, plant disease): flips, crops, colour jitter, then `TrivialAugmentWide` or `RandAugment`.
- Training from scratch or long fine-tunes on ImageNet-scale data: RandAugment + mixup/cutmix + random erasing is the modern recipe.
- Factory or medical images where orientation is arbitrary: add rotations and vertical flips; where colour is diagnostic (pathology stains, ripeness), keep colour jitter small.
- Speech recognition, keyword spotting, sound classification: noise + SpecAugment.
- Leaderboard or offline batch scoring where latency is not a concern: TTA.
- NOT when the dataset is huge relative to the model (augmentation then mainly slows convergence), and NOT ops that change the label (flipping digits 6/9, cropping out the defect being detected).

## Setup & code
```bash
pip install torch torchvision
```
Train vs eval pipelines, boxes that follow the image, mixup/cutmix on a batch, and TTA:
```python
import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.transforms import v2

train_tf = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.TrivialAugmentWide(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    v2.RandomErasing(p=0.25),
])
eval_tf = v2.Compose([
    v2.ToImage(),
    v2.Resize(256, antialias=True), v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

img = torch.randint(0, 256, (3, 300, 400), dtype=torch.uint8)   # stand-in for a PIL image
print(train_tf(img).shape, eval_tf(img).shape)

# Detection: boxes move with the image.
boxes = tv_tensors.BoundingBoxes([[50, 60, 150, 200]], format="XYXY", canvas_size=(300, 400))
flip = v2.RandomHorizontalFlip(p=1.0)
_, flipped = flip(tv_tensors.Image(img), boxes)
print("box after flip:", flipped.tolist())       # x coords mirrored: [[250, 60, 350, 200]]

# Mixup / CutMix on a batch: labels become soft, so use soft-target cross-entropy.
num_classes = 10
images = torch.randn(8, 3, 224, 224)
labels = torch.randint(0, num_classes, (8,))
mix = v2.RandomChoice([v2.MixUp(num_classes=num_classes), v2.CutMix(num_classes=num_classes)])
images, soft_labels = mix(images, labels)
print("soft labels:", soft_labels.shape, soft_labels[0].sum().item())   # (8, 10), rows sum to 1

model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, num_classes))
loss = nn.CrossEntropyLoss()(model(images), soft_labels)   # CE accepts class probabilities
print("loss", loss.item())

# Test-time augmentation: average probabilities over the image and its flip.
model.eval()
with torch.no_grad():
    x = images[:2]
    probs = (model(x).softmax(-1) + model(x.flip(-1)).softmax(-1)) / 2
print("TTA probs:", probs.shape)
```
Use mixup/cutmix *after* batching (in the training loop or a `collate_fn`), not inside the per-sample transform.

## Choosing / trade-offs
- **Hand-picked ops vs policies.** Start with crop + flip; add `TrivialAugmentWide` (zero tuning) next; `RandAugment` when you can tune magnitude. Learned policies (AutoAugment) cost search time for little extra.
- **Strength vs model size.** Small models and short schedules underfit with heavy augmentation; large models trained long need it. Increase strength until train accuracy drops but validation still rises.
- **Mixup/CutMix.** Help calibration and robustness on long training runs; usually skip for short fine-tunes of strong pretrained backbones and for detection.
- **torchvision v2 vs Albumentations.** torchvision v2 is built in and handles boxes/masks; Albumentations has more ops (weather, elastic, optical distortion) and is fast on NumPy images.
- **Offline vs on-the-fly.** Augment on the fly in the `Dataset` so every epoch sees new variants; pre-generating a fixed augmented copy gives far less diversity.
- **TTA cost.** k variants = k× inference compute; usually not worth it for real-time serving.

## Gotchas
- Augmenting the validation or test set inflates or deflates metrics and makes runs incomparable.
- Augmentation ops that change geometry without updating boxes/masks/keypoints silently corrupt labels; always draw a few augmented samples with their targets.
- Normalise after colour/photometric ops, and `RandomErasing` only works on tensors (place it after `ToDtype`/`Normalize`).
- With mixup/cutmix, training accuracy against hard labels looks low; track validation accuracy instead.
- Heavy CPU augmentation starves the GPU; raise `num_workers` or move ops to GPU (see [[efficient-training-mixed-precision]]).
- Synthetic or LLM-paraphrased text can leak test examples or drift the label; dedupe against the test set.
- Augmenting before the train/test split lets near-copies of a test image sit in training.

## Related
- [[deep-learning-training]] - augmentation alongside weight decay, dropout and early stopping.
- [[image-classification]] - where image augmentation is applied in practice.
- [[object-detection]] - box-aware augmentation (mosaic, flips, scale jitter).
- [[data-labeling-and-synthetic-data]] - generating new examples rather than transforming existing ones.
- [[imbalanced-data]] - oversampling is a different tool from invariance augmentation.

## References
- torchvision transforms v2: https://pytorch.org/vision/stable/transforms.html
- torchvision CutMix/MixUp example: https://pytorch.org/vision/stable/auto_examples/transforms/plot_cutmix_mixup.html
- Zhang et al., mixup: https://arxiv.org/abs/1710.09412
- Yun et al., CutMix: https://arxiv.org/abs/1905.04899
- Cubuk et al., RandAugment: https://arxiv.org/abs/1909.13719
- Park et al., SpecAugment: https://arxiv.org/abs/1904.08779
- Albumentations docs: https://albumentations.ai/docs/
