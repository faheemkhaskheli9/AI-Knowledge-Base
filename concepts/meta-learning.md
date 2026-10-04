---
title: Meta-learning (learning to learn)
category: concepts
tags: [meta-learning, learning-to-learn, maml, reptile, prototypical-networks, few-shot, episodic-training, torch-func, learn2learn, pytorch]
use_cases:
  - "adapt a model to a new customer, user or device from only a handful of labelled examples"
  - "train an initialisation that fine-tunes well in a few gradient steps"
  - "build a few-shot classifier that handles classes unseen during training"
  - "personalise a model per user when each user has very little data"
  - "decide between MAML, prototypical networks and plain fine-tuning for a few-shot problem"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1703.03400
  - https://arxiv.org/abs/1803.02999
  - https://arxiv.org/abs/1703.05175
  - https://arxiv.org/abs/1904.04232
  - https://pytorch.org/docs/stable/func.html
---

# Meta-learning (learning to learn)

## Summary
Meta-learning trains a model across many small related *tasks* so that it learns a new task from very few examples. Instead of one big dataset, training is a stream of *episodes*, each with a small support set (to adapt on) and a query set (to evaluate the adaptation). The three classic families are *optimisation-based* (MAML, Reptile: learn an initialisation that fine-tunes fast), *metric-based* (prototypical networks: learn an embedding where a new class is the mean of its few examples) and *model-based* (a network that reads the support set as input; large language models doing in-context learning are the extreme case). Since about 2019, a strong pretrained backbone plus simple fine-tuning or a nearest-prototype head often matches these methods, so try that baseline first.

## Key concepts
- **Task distribution.** Meta-learning assumes tasks come from a shared family (sine waves with different phase, classifiers for different sets of characters, per-user preference models). It generalises to new tasks *from the same family*, not to arbitrary ones.
- **Episode = N-way K-shot.** Sample N classes, K labelled examples each (support) plus held-out queries; train so that query loss after adaptation is low. Test episodes use classes never seen in training.
- **Inner vs outer loop.** Inner loop: adapt to one task (a few SGD steps, or computing prototypes). Outer loop: update the shared parameters so the *post-adaptation* query loss falls across tasks.
- **MAML.** Backpropagates through the inner SGD steps (second-order gradients). Learns an initialisation from which a few steps reach a good task solution. Model-agnostic but memory-hungry and finicky. *First-order MAML* drops the second-order term.
- **Reptile.** Even simpler first-order method: adapt to a task for k steps, then move the initialisation a fraction toward the adapted weights. No query set, no second-order gradients, nearly MAML-level results on standard benchmarks.
- **Prototypical networks.** Embed support examples, average per class into a *prototype*, classify queries by distance to prototypes. No inner-loop optimisation; new classes are added by computing their mean embedding ([[metric-learning-and-few-shot]]).
- **In-context learning.** A large LM given examples in the prompt is a model-based meta-learner trained implicitly by pretraining; for text tasks it usually replaces explicit meta-learning ([[prompt-engineering]]).

## When to use / scenarios
- Per-user or per-device personalisation where each user contributes a handful of labelled samples (keyboard prediction, wearable activity, recommendation cold start) and there are many users to meta-train on.
- Few-shot classification of rare categories: new defect types, rare species, new characters or logos, where classes keep arriving.
- Robotics and control: adapt a dynamics model or policy to a new payload or terrain from a few trials ([[reinforcement-learning]]).
- Scientific and drug-discovery tasks with many small assays, each with few measurements.
- NOT first choice when a good pretrained model exists: fine-tune its head or use nearest-prototype on frozen embeddings, which "A Closer Look at Few-shot Classification" found competitive with meta-learning ([[transfer-learning-and-domain-adaptation]]).
- NOT for text tasks with an LLM available: few-shot prompting or light fine-tuning ([[fine-tuning-and-peft]]).
- NOT when you lack many *related training tasks*: meta-learning needs a task distribution, not just a small dataset.

## Setup & code
```bash
pip install torch numpy
```
Reptile on the classic sine-wave family: each task is `y = a·sin(x + b)` with random amplitude and phase; at test time the model sees 10 points of an unseen wave and takes 16 SGD steps. Compare a Reptile-trained initialisation to a model pretrained on all tasks pooled together:
```python
import math, torch, torch.nn as nn

torch.manual_seed(0)

def sample_task():
    a, b = torch.empty(1).uniform_(0.1, 5.0), torch.empty(1).uniform_(0, math.pi)
    return lambda x: a * torch.sin(x + b)

def batch(f, n):
    x = torch.empty(n, 1).uniform_(-5, 5)
    return x, f(x)

def net():
    return nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

def adapt(model, x, y, steps, lr=0.02):          # inner loop: plain SGD on the support set
    opt = torch.optim.SGD(model.parameters(), lr)
    for _ in range(steps):
        opt.zero_grad(); nn.functional.mse_loss(model(x), y).backward(); opt.step()
    return model

# Reptile outer loop: nudge the initialisation toward each task's adapted weights.
meta = net()
for it in range(3000):
    f = sample_task()
    fast = net(); fast.load_state_dict(meta.state_dict())
    adapt(fast, *batch(f, 10), steps=5)
    eps = 0.1 * (1 - it / 3000)                  # outer step size, annealed
    with torch.no_grad():
        for p, q in zip(meta.parameters(), fast.parameters()):
            p += eps * (q - p)

# Baseline: ordinary training on all tasks pooled (learns the average wave, roughly zero).
pooled = net(); opt = torch.optim.Adam(pooled.parameters(), 1e-3)
for _ in range(3000):
    x, y = batch(sample_task(), 10)
    opt.zero_grad(); nn.functional.mse_loss(pooled(x), y).backward(); opt.step()

def few_shot_mse(init, n_tasks=100):
    errs = []
    for _ in range(n_tasks):
        f = sample_task()
        m = net(); m.load_state_dict(init.state_dict())
        adapt(m, *batch(f, 10), steps=16)        # 10-shot, 16 steps
        xq, yq = batch(f, 100)
        with torch.no_grad():
            errs.append(nn.functional.mse_loss(m(xq), yq).item())
    return sum(errs) / len(errs)

r, p = few_shot_mse(meta), few_shot_mse(pooled)
print(f"10-shot MSE after 16 steps  reptile {r:.2f}   pooled pretraining {p:.2f}")
assert r < p
```
The Reptile initialisation fits an unseen wave from 10 points several times better than the pooled model, which has learned the average of all waves (near zero) and has nothing task-specific to start from.

For full MAML in PyTorch, write the inner loop functionally with `torch.func.functional_call` and `torch.func.grad` so the outer gradient flows through the inner updates; `learn2learn` and `higher` wrap this pattern, along with standard few-shot datasets (Omniglot, mini-ImageNet) and episode samplers.

## Choosing / trade-offs
- **Baseline first.** Pretrained backbone + fine-tune the last layer, or nearest-prototype on frozen embeddings. If it is within a few points of the target, stop there.
- **Metric-based (ProtoNets)** when the task is classification and new classes arrive often: no gradient steps at test time, adding a class is one forward pass.
- **Optimisation-based (MAML/Reptile)** for regression, control or any task where adaptation must change behaviour, not just add a class. Reptile or first-order MAML unless you have shown second-order gradients help.
- **Model-based / in-context** when a foundation model already does the task family; prompting is the cheapest "meta-learner".
- **Cost.** MAML memory grows with inner steps (it stores the unrolled graph); Reptile and ProtoNets cost about the same as ordinary training.

## Gotchas
- Test tasks must come from the same family as training tasks; a meta-learner trained on sine waves fails on square waves. Check that per-user or per-site tasks really share structure.
- Few-shot results swing a lot between episodes; report mean and 95% confidence interval over hundreds of test episodes, not one.
- BatchNorm inside the inner loop leaks information between query examples if run in training mode ("transductive" batch norm); be explicit about which mode you evaluate in.
- MAML is sensitive to inner learning rate and step count; use the same inner-loop settings at meta-train and meta-test time.
- Benchmark gains on Omniglot/mini-ImageNet often disappear with a better backbone or data augmentation; compare against a tuned fine-tuning baseline on your own data.
- Splitting tasks randomly is not enough when tasks overlap (same user in two tasks, near-duplicate classes); split by the real grouping key.

## Related
- [[metric-learning-and-few-shot]] - embeddings, prototypes and Siamese networks for few-shot classification.
- [[transfer-learning-and-domain-adaptation]] - the fine-tuning baseline that often wins.
- [[continual-learning]] - adding tasks over time without forgetting.
- [[reinforcement-learning]] - meta-RL and fast adaptation of policies.
- [[prompt-engineering]] - in-context learning as model-based meta-learning.
- [[neural-architecture-search]] - another "learn the learning setup" outer loop.

## References
- Finn et al., Model-Agnostic Meta-Learning (MAML): https://arxiv.org/abs/1703.03400
- Nichol et al., On First-Order Meta-Learning Algorithms (Reptile): https://arxiv.org/abs/1803.02999
- Snell et al., Prototypical Networks for Few-shot Learning: https://arxiv.org/abs/1703.05175
- Chen et al., A Closer Look at Few-shot Classification: https://arxiv.org/abs/1904.04232
- PyTorch `torch.func` (functional_call, grad): https://pytorch.org/docs/stable/func.html
