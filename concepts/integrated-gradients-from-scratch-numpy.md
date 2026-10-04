---
title: Integrated gradients from scratch in NumPy
category: concepts
tags: [integrated-gradients, attribution, explainability, interpretability, saliency, gradients, xai, numpy, from-scratch]
use_cases:
  - "explain which input features drove a neural network's prediction"
  - "get feature attributions that add up to the change in model output"
  - "understand why plain gradients fail on saturated networks"
  - "pick a baseline and step count for Captum's IntegratedGradients"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1703.01365
  - https://captum.ai/api/integrated_gradients.html
  - https://distill.pub/2020/attribution-baselines/
---

# Integrated gradients from scratch in NumPy

## Summary
Integrated gradients (IG) attributes a model's output to its input features by averaging the gradient along a straight path from a *baseline* input (for example all zeros, a black image or padding tokens) to the actual input, then multiplying by `(x − baseline)`. Unlike a raw gradient or gradient × input, the attributions satisfy *completeness*: they sum exactly to `f(x) − f(baseline)`. This file builds IG for a small tanh MLP in NumPy. The attributions sum to `9.1666` against a true output change of `9.1660`, the gap shrinks with more steps, and gradient × input gets both the total (`2.49`) and the sign of the largest feature wrong.

## Key concepts
- **Definition.** `IG_i(x) = (x_i − x'_i) · ∫₀¹ ∂f(x' + α(x − x')) / ∂x_i dα`, where `x'` is the baseline.
- **Completeness.** By the fundamental theorem of calculus for path integrals, `Σ_i IG_i = f(x) − f(x')`. This is the main reason to prefer IG over raw gradients: every unit of output change is assigned to some feature.
- **Riemann approximation.** The integral is approximated with `m` points along the path, so IG costs `m` forward and backward passes (typically 20-300). The midpoint rule used here converges faster than the left-endpoint sum.
- **Saturation.** When a unit saturates (tanh, sigmoid, ReLU off-region), the local gradient at `x` is near zero even though the feature changed the output a lot on the way there. IG integrates over the whole path, so it still credits that feature.
- **The baseline is a modelling choice.** Attributions answer "why `f(x)` rather than `f(baseline)`". Changing the baseline changes the answer.

## When to use / scenarios
- Explaining a single prediction of a differentiable model: credit scoring, medical imaging, text classification (attribution per token embedding), as a sanity check before shipping.
- Debugging: finding that a model relies on a watermark, a hospital ID or a padding token.
- Compliance or user-facing explanations where the attributions must add up to the score.
- Not for: tree models (use SHAP's TreeExplainer, see [[model-interpretability]]), non-differentiable pipelines, or global feature importance across a dataset (average many IG maps or use permutation importance). Attributions explain the model, not causality in the data.

## Setup & code
NumPy only. The gradient is written by hand for a one-hidden-layer MLP. With a framework, swap `grad_f` for autograd.

```python
import numpy as np

rng = np.random.default_rng(0)
D, H = 6, 32
W1, b1 = rng.normal(0, 1, (D, H)), rng.normal(0, 0.5, H)
w2 = rng.normal(0, 1, H)


def f(X):
    """Tiny MLP, scalar output per row: tanh hidden layer, linear head."""
    return np.tanh(X @ W1 + b1) @ w2


def grad_f(X):
    h = np.tanh(X @ W1 + b1)
    return ((1 - h ** 2) * w2) @ W1.T        # df/dx, one row per input


def integrated_gradients(x, baseline, steps=64):
    """Riemann (midpoint) sum of gradients along the straight path baseline -> x."""
    alphas = (np.arange(steps) + 0.5) / steps
    path = baseline + alphas[:, None] * (x - baseline)
    return (x - baseline) * grad_f(path).mean(axis=0)


x = rng.normal(0, 1, D)
base = np.zeros(D)

ig = integrated_gradients(x, base)
print("IG attributions     ", ig.round(3))
print(f"sum of IG {ig.sum():+.4f}   f(x) - f(base) {f(x[None])[0] - f(base[None])[0]:+.4f}")

gxi = x * grad_f(x[None])[0]
print("grad x input        ", gxi.round(3))
print(f"sum of grad*input {gxi.sum():+.4f}   (no completeness guarantee)")

for steps in (4, 16, 64, 256):
    gap = integrated_gradients(x, base, steps).sum() - (f(x[None])[0] - f(base[None])[0])
    print(f"steps {steps:4d}  completeness gap {gap:+.1e}")

# Saturation: scale the input so tanh units saturate, plain gradient goes ~0.
xs = 8 * x
print(f"saturated x: f(x)-f(base) {f(xs[None])[0] - f(base[None])[0]:+.3f}   "
      f"sum grad*input {(xs * grad_f(xs[None])[0]).sum():+.3f}   "
      f"sum IG {integrated_gradients(xs, base, 256).sum():+.3f}")
```

Output (Python 3.14, NumPy 2.5):
```
IG attributions      [-1.034 -0.522  0.756  5.707  1.746  2.514]
sum of IG +9.1666   f(x) - f(base) +9.1660
grad x input         [-0.082 -0.241  1.653 -0.369  0.005  1.526]
sum of grad*input +2.4919   (no completeness guarantee)
steps    4  completeness gap +2.3e-01
steps   16  completeness gap +1.0e-02
steps   64  completeness gap +6.2e-04
steps  256  completeness gap +3.9e-05
saturated x: f(x)-f(base) +9.380   sum grad*input -0.957   sum IG +9.383
```

IG says feature 3 is responsible for most of the output (`+5.7`). Gradient × input gives it `−0.37`, the wrong sign, because the tanh units it drives are already saturated at `x`, so the local slope says little about how the output got there. The completeness gap drops by about 16× for every 4× more steps, the expected rate for the midpoint rule, so the gap doubles as a convergence check. With inputs scaled 8× into full saturation, gradient × input explains `−0.96` of a `+9.38` change, and IG still explains all of it.

## Choosing / trade-offs
- **Baseline.** Zeros are common for images and tabular data after standardisation (zero means "average"). For text, use the padding or `[MASK]` embedding, not the zero vector. A blurred image or an average over random baselines (expected gradients, as in SHAP's GradientExplainer) reduces dependence on one arbitrary choice.
- **Steps.** Start with 50 and raise them until the completeness gap is below about 1% of `|f(x) − f(x')|`. Captum returns the gap as `convergence_delta`.
- **IG vs SHAP vs gradient × input.** Gradient × input is one backward pass but has no completeness guarantee. IG costs `m` passes and is exact for the chosen path. KernelSHAP is model-agnostic but much slower for high-dimensional inputs. For trees, TreeSHAP is exact and fast.
- **IG vs SmoothGrad.** SmoothGrad averages gradients over noisy copies to produce cleaner-looking image maps but does not add up to the output. They can be combined (SmoothGrad on IG).
- **Which output.** For a classifier, attribute the logit of the class of interest, not the softmax probability. The softmax couples classes and saturates.
- **Libraries.** Captum (`IntegratedGradients`, `LayerIntegratedGradients` for embeddings), `tf-explain` or the original TensorFlow implementation, and SHAP's `GradientExplainer`.

## Gotchas
- The model must be differentiable along the whole path. Argmax, rounding, embedding lookups by integer ID or a tree in the pipeline break it. For text, attribute at the embedding layer.
- Batch normalisation in training mode, dropout and other randomness make `f` stochastic. Call `model.eval()` first.
- A baseline equal to the input in some feature gives that feature zero attribution by construction, whatever the model does with it.
- Large step counts are memory-heavy if all path points go through the model in one batch. Use `internal_batch_size` in Captum.
- Attribution maps look plausible even for a randomly initialised model. Run the sanity check of randomising the weights and confirming the attributions change.
- Attributions are for one input and one baseline. Do not read a single map as the model's global behaviour.

## Related
- [[saliency-maps-and-neural-attribution]] - overview of gradient, CAM and perturbation attribution methods.
- [[model-interpretability]] - SHAP, permutation importance and partial dependence for tabular models.
- [[neural-network-from-scratch-numpy]] - the hand-written backprop the gradient here relies on.
- [[autograd-engine-from-scratch]] - how frameworks compute `grad_f` automatically.

## References
- Sundararajan, Taly and Yan (2017), "Axiomatic Attribution for Deep Networks": https://arxiv.org/abs/1703.01365
- Captum `IntegratedGradients` API: https://captum.ai/api/integrated_gradients.html
- Sturmfels, Lundberg and Lee (2020), "Visualizing the Impact of Feature Attribution Baselines", Distill: https://distill.pub/2020/attribution-baselines/
