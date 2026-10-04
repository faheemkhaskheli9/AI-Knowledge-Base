---
title: Saliency maps and attribution for neural networks (gradients, Grad-CAM, Integrated Gradients)
category: concepts
tags: [explainability, saliency-maps, grad-cam, integrated-gradients, attribution, smoothgrad, captum, cnn, pytorch, xai]
use_cases:
  - "show which part of an image a CNN looked at to make its prediction"
  - "check that a medical or defect classifier uses the lesion, not a ruler or watermark"
  - "attribute a neural network's prediction to input pixels, tokens or features"
  - "add a heatmap explanation to a vision model's output in a review tool"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1610.02391
  - https://arxiv.org/abs/1703.01365
  - https://arxiv.org/abs/1810.03292
  - https://arxiv.org/abs/1706.03825
  - https://captum.ai/docs/algorithms
---

# Saliency maps and attribution for neural networks (gradients, Grad-CAM, Integrated Gradients)

## Summary
Attribution methods answer "which parts of this input drove this prediction?" for a single example. For deep networks, the main family uses gradients: **vanilla gradient saliency** (how much the class score changes per pixel), **Grad-CAM** (a coarse heatmap from the last convolutional layer), and **Integrated Gradients** (gradients averaged along a path from a baseline, with a completeness guarantee). They are cheap, need only a backward pass, and are the standard way to sanity-check vision models. They are also easy to over-trust. Some maps look plausible even for a randomly initialised network.

## Key concepts
- **Local, post-hoc explanation.** One heatmap explains one prediction of an already-trained model. It is not a global description of the model ([[model-interpretability]] covers global methods like permutation importance and SHAP).
- **Vanilla gradient.** `|∂ score_c / ∂ x|`, max over colour channels. Fine-grained but noisy, and it saturates: a feature that is already "fully on" can get zero gradient.
- **SmoothGrad.** Average the gradient over several noisy copies of the input. This reduces visual noise at the cost of N backward passes.
- **Grad-CAM.** Take the activations A of a late conv layer `(C, h, w)` and the gradients of the class score with respect to them. Weight each channel by its spatially averaged gradient, sum the channels, apply ReLU and upsample to the image size. It is class-discriminative and robust, but coarse (7x7 for ResNet's last block).
- **Integrated Gradients (IG).** `(x - x') · ∫₀¹ ∂f(x' + α(x - x'))/∂x dα`, approximated with a sum over m steps. Satisfies **completeness**: attributions sum to `f(x) - f(x')`, which is a built-in correctness check. The result depends on the baseline x' (black image, blurred image, zero embedding).
- **Baseline.** The "absence of signal" reference. A black image says "relative to black". For text, the usual choice is a pad- or zero-embedding. Different baselines give different maps, so pick one that means "no information" in your domain.
- **Sanity checks (Adebayo et al., 2018).** Randomise the model weights or the labels. A trustworthy method's map should change a lot. Guided Backprop and some other methods barely change, so they behave like edge detectors, not explanations.

## When to use / scenarios
- Medical imaging: confirm a classifier highlights the lesion, not a scanner tag, ruler or hospital marker (a known shortcut-learning failure).
- Manufacturing inspection: show the operator where the defect is, alongside the classification ([[image-classification]]).
- Model debugging: find spurious correlations (snow behind "wolf", watermark on "horse") before deployment.
- Text and tabular neural nets: IG over token embeddings or input features ranks what drove a score.
- Not a substitute for a localisation model. If you need reliable boxes or masks, train detection or segmentation ([[object-detection]], [[segmentation]]).
- Not proof of causality or of correctness. Treat maps as hypotheses to verify, e.g. by occluding the highlighted region and checking that the score drops.

## Setup & code
```bash
pip install torch torchvision
```
Plain PyTorch, no extra libraries:
```python
import torch
import torch.nn.functional as F
from torchvision.models import resnet18

model = resnet18(weights=None).eval()   # use weights="DEFAULT" for a real, pretrained model
x = torch.randn(1, 3, 224, 224, requires_grad=True)   # a normalised image in practice

# 1) Vanilla gradient saliency
logits = model(x)
cls = logits.argmax(1).item()
logits[0, cls].backward()
saliency = x.grad.abs().amax(dim=1)[0]           # (224, 224)

# 2) Grad-CAM on the last conv block
acts, grads = {}, {}
layer = model.layer4
h1 = layer.register_forward_hook(lambda m, i, o: acts.update(a=o))
h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.update(g=go[0]))
model.zero_grad()
model(x.detach())[0, cls].backward()
h1.remove(); h2.remove()
w = grads["g"].mean(dim=(2, 3), keepdim=True)                   # channel importance
cam = F.relu((w * acts["a"]).sum(1, keepdim=True)).detach()     # (1, 1, 7, 7)
cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)        # overlay on the image

# 3) Integrated Gradients with a black baseline
def integrated_gradients(model, x, cls, steps=32):
    baseline = torch.zeros_like(x)
    alphas = torch.linspace(0, 1, steps).view(-1, 1, 1, 1)
    path = (baseline + alphas * (x - baseline)).requires_grad_(True)
    model(path)[:, cls].sum().backward()
    return (x - baseline) * path.grad.mean(0, keepdim=True)

ig = integrated_gradients(model, x.detach(), cls)
with torch.no_grad():
    gap = model(x)[0, cls] - model(torch.zeros_like(x))[0, cls]
print(ig.sum().item(), gap.item())   # completeness: these should be close
```
Tested with torch 2.13 and torchvision 0.28 (CPU). In one run the IG sum was 2.415 vs a score gap of 2.493 with 32 steps. More steps close the gap.

For production use, **Captum** (`pip install captum`) implements these methods and more (`Saliency`, `LayerGradCam`, `IntegratedGradients`, `NoiseTunnel` for SmoothGrad, `Occlusion`) with batching and convergence checks. `pytorch-grad-cam` adds Grad-CAM++, ScoreCAM and support for ViTs.

## Choosing / trade-offs
| Method | Cost | Resolution | Notes |
|---|---|---|---|
| Vanilla gradient | 1 backward | Pixel | Noisy, saturates |
| SmoothGrad | N backward | Pixel | Cleaner maps, N x cost |
| Grad-CAM | 1 backward | Feature map (coarse) | Class-discriminative. Passes sanity checks. CNN default |
| Integrated Gradients | m backward (20-300) | Pixel/token | Completeness check. Baseline-dependent |
| Occlusion | One forward per patch | Patch | Model-agnostic, slow, easy to explain to non-experts |

- **Vision CNN, quick look:** Grad-CAM on the last conv block.
- **Need per-pixel or per-token scores that add up:** Integrated Gradients, checked with completeness.
- **Transformers (ViT, text):** IG on embeddings, or attention rollout. Raw attention weights are not reliable attributions on their own.
- **Model-agnostic or tabular:** SHAP or permutation methods in [[model-interpretability]].

## Gotchas
- **Model in train mode.** Dropout and BatchNorm in train mode change the output between passes. Call `model.eval()`. Gradients still flow in eval mode.
- **Explaining softmax probabilities** spreads attribution across competing classes. Attribute the **logit** of the target class.
- **Grad-CAM on the wrong layer.** Early layers give edge-like maps. The final linear layer has no spatial axes. Use the last spatial conv block (`layer4` in ResNet, `features[-1]` in many torchvision models).
- **Normalisation before display.** Per-image min-max scaling always produces a "hot" region, even when every attribution is tiny. Compare raw magnitudes across images when that matters.
- **Too few IG steps.** If the completeness gap is large, raise `steps`. Captum's `return_convergence_delta=True` reports it.
- **Plausible ≠ faithful.** Run at least the model-randomisation sanity check, or an occlusion test, before showing maps to clinicians or customers.
- **In-place ReLUs and backward hooks.** `register_full_backward_hook` can fail on modules whose outputs are modified in place. Use forward hooks that call `tensor.register_hook`, or Captum, if this happens.

## Related
- [[model-interpretability]] - SHAP, permutation importance and global explanations, mostly for tabular models.
- [[image-classification]] - the CNN and ViT classifiers these maps are usually applied to.
- [[backpropagation-and-autograd]] - how the input gradients are computed.
- [[mechanistic-interpretability]] - explaining what circuits inside a network compute, instead of per-input heatmaps.
- [[adversarial-examples-and-robustness]] - uses the same input gradients to fool the model instead of explaining it.
- [[cnn-and-rnn-architectures]] - the conv layers and feature maps Grad-CAM reads.

## References
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization" (2017): https://arxiv.org/abs/1610.02391
- Sundararajan et al., "Axiomatic Attribution for Deep Networks" (Integrated Gradients, 2017): https://arxiv.org/abs/1703.01365
- Smilkov et al., "SmoothGrad: removing noise by adding noise" (2017): https://arxiv.org/abs/1706.03825
- Adebayo et al., "Sanity Checks for Saliency Maps" (NeurIPS 2018): https://arxiv.org/abs/1810.03292
- Simonyan et al., "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps" (2013): https://arxiv.org/abs/1312.6034
- Captum algorithms: https://captum.ai/docs/algorithms
