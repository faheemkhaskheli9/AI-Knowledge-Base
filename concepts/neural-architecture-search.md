---
title: Neural architecture search
category: concepts
tags: [neural-architecture-search, nas, darts, one-shot-nas, hardware-aware, efficientnet, optuna, model-design, pytorch]
use_cases:
  - "find a small network architecture that fits a latency or memory budget on a device"
  - "decide whether to search an architecture or just reuse a proven backbone"
  - "tune layer counts, widths and kernel sizes automatically instead of by hand"
  - "pick the best model variant for a phone, microcontroller or edge accelerator"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1808.05377
  - https://arxiv.org/abs/1806.09055
  - https://arxiv.org/abs/1905.11946
  - https://arxiv.org/abs/1908.09791
  - https://arxiv.org/abs/1902.07638
  - https://optuna.readthedocs.io/
---

# Neural architecture search

## Summary
Neural architecture search (NAS) automates the design of a network: which layers, how deep, how wide, which kernel sizes and connections. It is defined by three choices: a *search space* (what architectures are allowed), a *search strategy* (random, evolutionary, reinforcement learning, Bayesian, gradient-based) and a *performance estimate* (how to score a candidate without training it fully). Early NAS cost thousands of GPU-days; weight sharing and one-shot supernets cut that to GPU-hours. Today its main practical use is *hardware-aware* search for edge devices; for most projects, a proven pretrained backbone plus hyperparameter tuning beats a custom search.

## Key concepts
- **Search space.** Cell-based (search a small repeating block, stack it, as in NASNet/DARTS) or macro (per-stage depth, width, kernel, expansion ratio, as in MobileNetV3/EfficientNet). A good search space matters more than the search algorithm.
- **Search strategies.**
  - **Random search**: a strong baseline that many papers fail to beat (Li & Talwalkar 2019).
  - **Evolutionary / regularised evolution**: mutate the best architectures; robust and parallel.
  - **Reinforcement learning**: a controller proposes architectures (original NAS, very expensive).
  - **Bayesian optimisation / TPE**: model the score from past trials ([[hyperparameter-tuning]]).
  - **Gradient-based (DARTS)**: relax the discrete choice into a softmax over operations and learn it by gradient descent; fast but unstable (collapses to skip connections).
- **Performance estimation.** Full training per candidate is the bottleneck. Cheaper proxies: fewer epochs, smaller data, early stopping/successive halving, learning-curve extrapolation, and **weight sharing**.
- **One-shot / supernet NAS.** Train one large network that contains all candidates as subnetworks (Once-for-All); then evaluate or search subnetworks without retraining. Train once, deploy many sizes.
- **Hardware-aware NAS.** Objective includes measured latency, memory or energy on the target device (MnasNet, FBNet, ProxylessNAS); results in a Pareto front of accuracy vs latency.
- **Compound scaling** (EfficientNet). Search a small base network, then scale depth, width and resolution together with fixed ratios; the cheapest practical take-away from NAS.
- **Benchmarks.** NAS-Bench-101/201 tabulate trained results for whole search spaces, so search algorithms can be compared without GPU cost.

## When to use / scenarios
- Edge and embedded deployment: a vision or audio model must hit, for example, < 20 ms on a specific phone NPU or microcontroller ([[edge-on-device]]).
- A product that ships one model family in many sizes (cloud, laptop, phone): one-shot supernets give every size from one training run.
- Unusual input modalities (sensor arrays, spectrograms, tabular-plus-image) without a standard backbone, and a large enough budget.
- Constrained search (number of layers, widths, dropout) on a small model, using Optuna: this is just hyperparameter tuning with architecture parameters.
- NOT for standard vision, text or speech tasks with available pretrained models: fine-tuning a known backbone is cheaper and usually better ([[transfer-learning-and-domain-adaptation]]).
- NOT for tabular data: boosting rarely needs it ([[gradient-boosting-tabular]]); AutoML tools search models and preprocessing instead ([[automl]]).

## Setup & code
```bash
pip install torch scikit-learn optuna
```
A small, honest NAS: Optuna's TPE searches depth, width, activation and dropout for an MLP on digits, with pruning of weak trials and a parameter-count penalty (a stand-in for latency):
```python
import optuna
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_tr, X_va, y_tr, y_va = train_test_split(X / 16.0, y, test_size=0.3, random_state=0)
X_tr, X_va = (torch.tensor(a, dtype=torch.float32) for a in (X_tr, X_va))
y_tr, y_va = torch.tensor(y_tr), torch.tensor(y_va)

def build(trial):
    layers, width_in = [], 64
    act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[
        trial.suggest_categorical("act", ["relu", "gelu", "tanh"])]
    for i in range(trial.suggest_int("depth", 1, 4)):
        w = trial.suggest_int(f"width_{i}", 16, 256, log=True)
        layers += [nn.Linear(width_in, w), act(), nn.Dropout(trial.suggest_float(f"drop_{i}", 0.0, 0.5))]
        width_in = w
    return nn.Sequential(*layers, nn.Linear(width_in, 10))

def objective(trial):
    torch.manual_seed(0)
    model = build(trial)
    opt = torch.optim.Adam(model.parameters(), lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True))
    for epoch in range(30):
        model.train()
        for idx in torch.randperm(len(X_tr)).split(64):
            loss = nn.functional.cross_entropy(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X_va).argmax(1) == y_va).float().mean().item()
        trial.report(acc, epoch)
        if trial.should_prune():                       # successive-halving-style early stop
            raise optuna.TrialPruned()
    params = sum(p.numel() for p in model.parameters())
    trial.set_user_attr("params", params)
    return acc - 0.02 * params / 100_000                  # accuracy minus a size penalty

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0),
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=40)
best = study.best_trial
print(f"best score {best.value:.3f}, params {best.user_attrs['params']}")
print(best.params)
```
For real hardware-aware search, replace the parameter penalty with measured latency on the target device, and use multi-objective `optuna.create_study(directions=["maximize", "minimize"])` to get the accuracy-latency Pareto front. For CNN search spaces at scale, see Once-for-All, NNI or Keras Tuner rather than hand-rolling a supernet.

## Choosing / trade-offs
- **Reuse a backbone first.** EfficientNet, MobileNetV3, ConvNeXt, ViT families are already NAS-derived or heavily tuned; pick the variant that fits your budget ([[cnn-and-rnn-architectures]]).
- **Small search space + Optuna/random search** when the model is small and each trial takes minutes; simple, parallel, reproducible.
- **One-shot supernet** when you need many deployment sizes or the search space is large; higher engineering cost, ranking of subnetworks can be unreliable.
- **DARTS-style differentiable search** for research or cell design; needs regularisation tricks and careful evaluation.
- **Compress instead of search.** Pruning, quantization and distillation of a known model often reach the same latency target with less effort ([[quantization]], [[knowledge-distillation-and-compression]]).

## Gotchas
- Comparing against a weak baseline: random search on the same space and budget is the minimum comparison; many NAS gains vanish against it.
- Selecting the architecture on the test set; keep a separate validation split and retrain the final architecture from scratch before reporting.
- Proxy mismatch: rankings from 5-epoch training or a shared supernet do not always match rankings after full training.
- FLOPs and parameter counts are poor latency proxies; measure on the target hardware with the real runtime and batch size.
- Seeds matter: re-run the best few candidates with several seeds before declaring a winner.
- The search space is the hidden design decision; a NAS result is only as good as the space a human defined.
- Compute and carbon cost; budget the search like any other experiment and track it ([[experiment-tracking]]).

## Related
- [[hyperparameter-tuning]] - the same search machinery (TPE, pruning) on non-architecture settings.
- [[automl]] - automated model and pipeline search, mostly for tabular data.
- [[cnn-and-rnn-architectures]] - the hand-designed and NAS-derived backbones to start from.
- [[black-box-and-evolutionary-optimization]] - evolutionary and Bayesian search strategies.
- [[knowledge-distillation-and-compression]] - shrink a model instead of searching a new one.
- [[edge-on-device]] - the deployment targets hardware-aware NAS optimises for.

## References
- Elsken, Metzen & Hutter, Neural Architecture Search: A Survey (2019): https://arxiv.org/abs/1808.05377
- Liu, Simonyan & Yang, DARTS: Differentiable Architecture Search (2019): https://arxiv.org/abs/1806.09055
- Tan & Le, EfficientNet: Rethinking Model Scaling (2019): https://arxiv.org/abs/1905.11946
- Cai et al., Once-for-All: Train One Network and Specialize it for Efficient Deployment (2020): https://arxiv.org/abs/1908.09791
- Li & Talwalkar, Random Search and Reproducibility for NAS (2019): https://arxiv.org/abs/1902.07638
- Optuna documentation: https://optuna.readthedocs.io/
