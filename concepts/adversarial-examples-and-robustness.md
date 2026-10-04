---
title: Adversarial examples and model robustness
category: concepts
tags: [adversarial-examples, robustness, fgsm, pgd, adversarial-training, evasion-attack, data-poisoning, certified-robustness, pytorch, ml-security]
use_cases:
  - "test how easily a classifier can be fooled by small input perturbations"
  - "harden an image, malware or fraud model with adversarial training"
  - "assess robustness to natural corruptions (blur, noise, weather) before deployment"
  - "threat-model a deployed ML system: evasion, poisoning, model extraction"
  - "understand why accuracy on a clean test set does not mean a model is safe"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1312.6199
  - https://arxiv.org/abs/1412.6572
  - https://arxiv.org/abs/1706.06083
  - https://arxiv.org/abs/1903.12261
  - https://atlas.mitre.org/
  - https://github.com/Trusted-AI/adversarial-robustness-toolbox
---

# Adversarial examples and model robustness

## Summary
An adversarial example is an input changed by a small, often imperceptible amount that makes a model predict the wrong answer with high confidence. Neural networks (and linear models in high dimensions) are vulnerable because their decision boundaries lie close to most inputs. Attacks are measured against a threat model: how much the attacker may change (an L∞ or L2 budget ε), and what they know (white-box gradients vs black-box queries). The main practical defence is **adversarial training** (train on attacked inputs, e.g. PGD), which trades some clean accuracy for robustness; robustness to *natural* shifts (noise, blur, lighting) is a related but separate goal that augmentation addresses.

## Key concepts
- **Threat model.** Perturbation set (`‖δ‖∞ ≤ ε`, e.g. 8/255 for images; L2; patches; semantic edits), attacker knowledge (white-box, black-box, transfer), goal (untargeted: any wrong class; targeted: a chosen class). Robustness claims mean nothing without it.
- **FGSM** (Goodfellow 2014): one step, `x' = x + ε · sign(∇x L)`. Cheap; a weak attack, useful for quick checks and fast adversarial training.
- **PGD** (Madry 2017): many small FGSM steps, each projected back into the ε-ball, from a random start. The standard strong first-order attack and the basis of adversarial training.
- **Other attacks.** Carlini-Wagner (optimisation-based, minimal perturbation), AutoAttack (parameter-free ensemble used for benchmarking), black-box attacks via query-based gradient estimation or via **transfer** from a surrogate model.
- **Attack classes beyond evasion.** **Poisoning** (corrupting training data, backdoors/trojans triggered by a pattern), **model extraction** (stealing a model via queries), **membership inference** (was this record in training?). For LLMs the analogue is prompt injection / jailbreaks ([[prompt-injection]]).
- **Defences.**
  - Adversarial training (PGD-AT, TRADES): the only broadly reliable empirical defence; costs 3-10× training compute.
  - Certified defences (randomised smoothing, interval bound propagation): provable guarantees for a radius, at a large accuracy cost.
  - Input preprocessing (JPEG, denoising) and gradient masking: usually broken by adaptive attacks; do not rely on them.
- **Natural robustness.** Corruptions (ImageNet-C: noise, blur, weather, digital), distribution shift. Improved by augmentation, pretraining at scale, and monitoring ([[data-augmentation]], [[out-of-distribution-detection]]).

## When to use / scenarios
- Security-relevant classifiers where an adversary profits from evasion: malware detection, spam/fraud, content moderation, face recognition / liveness, CAPTCHA. Run attacks as part of evaluation.
- Safety-relevant perception (autonomous driving, medical imaging, industrial inspection): test against natural corruptions and physical perturbations (stickers, lighting), not only L∞ noise.
- Model audits and red-teaming before deployment; map threats with MITRE ATLAS ([[ai-security-privacy-compliance]]).
- Models trained on scraped or user-contributed data: consider poisoning and backdoor checks.
- NOT necessary (beyond basic robustness tests) for internal analytics or recommendation models with no adversary and no safety impact: adversarial training costs accuracy and compute.
- Tabular/fraud models: perturbations must respect feature constraints (can't change age by 0.3); use domain-aware attacks, not raw L∞ on all features.

## Setup & code
```bash
pip install torch scikit-learn
```
Train a small MLP on scikit-learn digits, attack it with FGSM and PGD, then adversarially train and compare:
```python
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32)      # pixels in [0, 1]
y = torch.tensor(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
loss_fn = nn.CrossEntropyLoss()

def pgd(model, x, y, eps, steps=10, alpha=None):
    """L-inf PGD from a random start; steps=1, alpha=eps, no random start = FGSM."""
    alpha = alpha or 2.5 * eps / steps
    delta = (torch.empty_like(x).uniform_(-eps, eps) if steps > 1 else torch.zeros_like(x))
    delta = (x + delta).clamp(0, 1) - x                # keep x + delta a valid image
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = loss_fn(model(x + delta), y)
        grad, = torch.autograd.grad(loss, delta)
        delta = (delta + alpha * grad.sign()).clamp(-eps, eps)   # project to the eps-ball
        delta = ((x + delta).clamp(0, 1) - x).detach()           # and to the valid range
    return x + delta

def train(adv_eps=0.0, epochs=60):
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for i in range(0, len(Xtr), 64):
            xb, yb = Xtr[i:i + 64], ytr[i:i + 64]
            if adv_eps:                                # adversarial training: fit on PGD inputs
                xb = pgd(model, xb, yb, adv_eps, steps=5)
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    return model

def acc(model, x):
    return (model(x).argmax(1) == yte).float().mean().item()

EPS = 0.1
for name, model in [("standard", train()), ("adv-trained", train(adv_eps=EPS))]:
    fgsm = pgd(model, Xte, yte, EPS, steps=1, alpha=EPS)
    strong = pgd(model, Xte, yte, EPS, steps=20)
    print(f"{name:12s} clean {acc(model, Xte):.3f}  FGSM {acc(model, fgsm):.3f}  PGD-20 {acc(model, strong):.3f}")
```
Expect the standard model's accuracy to fall sharply under FGSM and further under PGD, while the adversarially trained one keeps far more. On this toy task clean accuracy barely moves; on real image benchmarks adversarial training usually costs some clean accuracy. For real evaluations use maintained libraries: Adversarial Robustness Toolbox (ART), `torchattacks`, or AutoAttack.

## Choosing / trade-offs
- **Evaluate first.** Running PGD/AutoAttack against your model is cheap; adversarial training is not. Decide from the threat model whether the measured weakness matters.
- **ε choice.** Must match what an attacker can do without being noticed or what the domain allows; too-large ε makes inputs genuinely ambiguous.
- **Robust vs clean accuracy.** Adversarial training typically costs a few points of clean accuracy on images and multiplies training cost; TRADES exposes a knob for the trade-off.
- **Empirical vs certified.** Certified methods give guarantees only for small radii and lose much more accuracy; use them where an auditor needs a proof.
- **System-level defences** often beat model-level ones: rate limits and query monitoring (against extraction and black-box attacks), human review for high-stakes decisions, ensembles of different modalities, not exposing confidence scores.

## Gotchas
- Defences evaluated only against FGSM or non-adaptive attacks almost always look stronger than they are; test with PGD with random restarts, AutoAttack, and an attack designed for the defence.
- **Gradient masking**: if accuracy under a black-box/transfer attack is *lower* than under white-box PGD, your gradients are obfuscated, not robust.
- Project the *perturbation* so `x + δ` stays in the valid range; clamping only inside the forward pass zeroes the gradient on saturated pixels and silently weakens PGD (it can come out weaker than FGSM). Apply the same normalisation as in training; attacking normalised tensors with a pixel-space ε gives a different budget.
- Put the model in `eval()` mode when generating attacks for evaluation (dropout/batch-norm change gradients).
- Robustness to one norm (L∞) does not transfer to others (L2, patches, rotations).
- Adversarial training needs longer training and is prone to "robust overfitting"; early-stop on robust validation accuracy, not clean accuracy.

## Related
- [[out-of-distribution-detection]] - flagging inputs unlike the training data, the natural-shift side of robustness.
- [[data-augmentation]] - corruption/noise augmentation for natural robustness.
- [[ai-security-privacy-compliance]] - threat modelling, poisoning, model theft and privacy attacks at system level.
- [[prompt-injection]] - the adversarial-input problem for LLM applications.
- [[model-evaluation-and-metrics]] - adding robust accuracy to the evaluation suite.
- [[deep-learning-training]] - training-loop details adversarial training plugs into.

## References
- Szegedy et al., Intriguing Properties of Neural Networks: https://arxiv.org/abs/1312.6199
- Goodfellow et al., Explaining and Harnessing Adversarial Examples (FGSM): https://arxiv.org/abs/1412.6572
- Madry et al., Towards Deep Learning Models Resistant to Adversarial Attacks (PGD): https://arxiv.org/abs/1706.06083
- Hendrycks and Dietterich, Benchmarking Robustness to Common Corruptions (ImageNet-C): https://arxiv.org/abs/1903.12261
- MITRE ATLAS (adversarial threat landscape for AI): https://atlas.mitre.org/
- Adversarial Robustness Toolbox: https://github.com/Trusted-AI/adversarial-robustness-toolbox
