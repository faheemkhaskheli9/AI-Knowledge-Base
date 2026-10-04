---
title: Continual learning (catastrophic forgetting)
category: concepts
tags: [continual-learning, lifelong-learning, catastrophic-forgetting, replay, rehearsal, ewc, regularization, class-incremental, avalanche, pytorch]
use_cases:
  - "keep updating a model on new data without it forgetting what it learned before"
  - "add new classes to a deployed classifier without retraining from scratch on everything"
  - "fine-tune on a new task and keep performance on the old one"
  - "train under a data-retention limit where old data cannot be stored for long"
  - "explain why accuracy on old classes collapsed after the last retrain"
status: draft
last_verified: 2026-10-04
sources:
  - https://www.pnas.org/doi/10.1073/pnas.1611835114
  - https://arxiv.org/abs/1904.07734
  - https://arxiv.org/abs/1706.08840
  - https://avalanche.continualai.org/
---

# Continual learning (catastrophic forgetting)

## Summary
A neural network trained on task B after task A usually forgets A: gradient descent on B's loss freely overwrites the weights A depended on. This is *catastrophic forgetting*. Continual (lifelong) learning is the set of methods that let a model learn from a stream of tasks or data shifts while keeping earlier skills. The three families are *replay* (mix stored or generated old examples into new training), *regularisation* (penalise changes to weights that mattered for old tasks, e.g. EWC) and *architecture* (give new tasks new parameters, e.g. adapters per task). In practice, a small replay buffer is the strong baseline that most fancier methods struggle to beat.

## Key concepts
- **Stability vs plasticity.** A model that never changes cannot learn new tasks; one that changes freely forgets. Every method picks a point on that trade-off.
- **Three scenarios (van de Ven & Tolias).** *Task-incremental*: the task ID is given at test time (easiest, use a head per task). *Domain-incremental*: same label set, input distribution shifts (new camera, new season). *Class-incremental*: new classes arrive and the model must pick among all classes seen so far, with no task ID (hardest; where naive methods collapse).
- **Replay / rehearsal.** Keep a small buffer of old examples (reservoir sampling keeps a uniform sample of the stream) and mix them into every new batch. *Generative replay* trains a generator to produce old-looking data when raw data cannot be kept. *Dark Experience Replay* stores old logits too and distills them.
- **Regularisation.** *EWC* (Elastic Weight Consolidation) estimates each weight's importance for the old task with the diagonal Fisher information (mean squared gradient of the log-likelihood) and adds `λ/2 · Σ F_i (θ_i − θ*_i)²`. *SI* and *MAS* compute importance differently. *LwF* (Learning without Forgetting) distills the old model's outputs on new data. Cheap in memory, weaker than replay in class-incremental settings.
- **Architecture / parameter isolation.** Freeze the backbone and add a new head, adapter or LoRA per task ([[fine-tuning-and-peft]]); or grow the network (progressive nets). No forgetting by construction, but needs the task ID at inference or a router.
- **Gradient projection.** GEM/A-GEM constrain the new gradient so loss on buffered old examples does not increase.
- **Metrics.** Track a matrix `acc[i][j]` = accuracy on task j after training task i. Report *average accuracy* after the last task and *forgetting* (best past accuracy minus final accuracy per task). Compare to the *joint training* upper bound (all data at once).

## When to use / scenarios
- A deployed classifier must learn new product categories, defect types or intents every month, and retraining on the full history is too slow or the old data has expired (retention policy, privacy).
- Domain shift over time: a vision model moved to new stores, a speech model to new accents, a fraud model to new patterns ([[online-learning-and-concept-drift]]).
- Robotics and on-device learning where data arrives as a stream and storage is small.
- Fine-tuning a pretrained model on a narrow task while keeping general ability: replay some general data, or use PEFT so base weights stay frozen ([[fine-tuning-and-peft]]).
- NOT needed when you can afford to retrain on all data so far: periodic full retraining (or warm-start + all data) is the simplest and usually best option. Continual learning earns its complexity only when old data or compute is constrained.
- NOT for slow drift in tabular models: scheduled retraining on a sliding window ([[online-learning-and-concept-drift]]).

## Setup & code
```bash
pip install torch scikit-learn numpy
```
Class-incremental toy on the scikit-learn digits: task A = digits 0-4, then task B = digits 5-9, one shared 10-way head. Compare naive fine-tuning, EWC and a 200-example replay buffer:
```python
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32); y = torch.tensor(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
A_tr, B_tr = ytr < 5, ytr >= 5
A_te, B_te = yte < 5, yte >= 5

def mlp():
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))

def train(model, Xs, ys, epochs=60, penalty=None, replay=None):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(Xs)).split(32):
            xb, yb = Xs[idx], ys[idx]
            if replay is not None:                      # mix old examples into every batch
                r = torch.randint(len(replay[0]), (32,))
                xb, yb = torch.cat([xb, replay[0][r]]), torch.cat([yb, replay[1][r]])
            loss = F.cross_entropy(model(xb), yb)
            if penalty is not None:
                loss = loss + penalty(model)
            opt.zero_grad(); loss.backward(); opt.step()

def acc(model, mask):
    with torch.no_grad():
        return (model(Xte[mask]).argmax(1) == yte[mask]).float().mean().item()

def fisher(model, Xs, ys):                              # diagonal empirical Fisher
    F_ = [torch.zeros_like(p) for p in model.parameters()]
    for x, t in zip(Xs, ys):
        model.zero_grad()
        F.cross_entropy(model(x[None]), t[None]).backward()
        for f, p in zip(F_, model.parameters()):
            f += p.grad ** 2 / len(Xs)
    return F_

results = {}
for method in ["naive", "ewc", "replay"]:
    torch.manual_seed(0)
    m = mlp()
    train(m, Xtr[A_tr], ytr[A_tr])                      # task A: digits 0-4
    penalty = replay = None
    if method == "ewc":
        Fi = fisher(m, Xtr[A_tr], ytr[A_tr])
        old = [p.detach().clone() for p in m.parameters()]
        lam = 5000.0
        penalty = lambda mod: lam / 2 * sum((f * (p - o) ** 2).sum()
                                            for f, p, o in zip(Fi, mod.parameters(), old))
    if method == "replay":
        keep = torch.randperm(int(A_tr.sum()))[:200]   # small buffer of old data
        replay = (Xtr[A_tr][keep], ytr[A_tr][keep])
    train(m, Xtr[B_tr], ytr[B_tr], penalty=penalty, replay=replay)  # task B: digits 5-9
    results[method] = (acc(m, A_te), acc(m, B_te))

for k, (a, b) in results.items():
    print(f"{k:7s} old task acc {a:.2f}   new task acc {b:.2f}")
assert results["naive"][0] < 0.2 and results["replay"][0] > 0.85
```
Naive fine-tuning drops old-task accuracy to near zero (the shared head only ever sees classes 5-9). EWC slows the damage but still loses badly in this class-incremental setup. A 200-example buffer keeps both tasks above 90%. That gap is the usual result in class-incremental benchmarks.

For real projects, **Avalanche** (ContinualAI, PyTorch) packages benchmarks (split MNIST/CIFAR, CORe50), strategies (Naive, Replay, EWC, LwF, GEM, DER, ...) and the accuracy-matrix metrics, so you can compare methods on your stream without re-implementing them.

## Choosing / trade-offs
- **Can you store some old data?** Yes: start with replay (reservoir buffer, class-balanced). It is simple, strong and model-agnostic. Add logit distillation (DER, LwF) if the buffer is tiny.
- **No raw data allowed (privacy, retention)?** Regularisation (EWC, LwF), generative replay, or storing embeddings/features instead of inputs. Expect lower accuracy than raw replay.
- **Task ID known at inference?** Parameter isolation (a head or adapter per task) removes forgetting entirely at the cost of a growing model.
- **Pretrained backbone available?** Freezing it and training only a classifier (e.g. nearest-class-mean on frozen embeddings) is a surprisingly strong class-incremental baseline: nothing to forget in the backbone ([[transfer-learning-and-domain-adaptation]]).
- **Compute vs simplicity.** If a full retrain on all data fits the schedule, do that and skip this whole topic.

## Gotchas
- Evaluating only on the newest task hides forgetting; always log the full task-by-task accuracy matrix.
- Task-incremental results (task ID given) look far better than class-incremental ones; check which scenario a paper or library benchmark reports before comparing.
- In class-incremental learning, the final layer's bias drifts toward the newest classes; replay with class-balanced sampling or a bias-correction step on the head fixes much of it.
- EWC's λ spans orders of magnitude between problems; tune it on held-out old-task data. Fisher computed with predicted labels instead of true labels behaves differently.
- Hyperparameters tuned with access to the whole stream leak future information; tune on a separate validation stream.
- BatchNorm statistics drift to the new task even when weights are protected; freeze BN stats or use GroupNorm/LayerNorm.
- Replay buffers hold user data: they fall under the same retention and deletion rules as the original dataset.

## Related
- [[online-learning-and-concept-drift]] - streaming updates for drifting tabular data.
- [[transfer-learning-and-domain-adaptation]] - reusing a pretrained backbone; one-shot domain shift.
- [[fine-tuning-and-peft]] - adapters/LoRA per task as parameter isolation.
- [[knowledge-distillation-and-compression]] - distillation is the core of LwF and DER.
- [[deep-learning-training]] - optimizers and normalisation layers involved.
- [[metric-learning-and-few-shot]] - prototype classifiers that add classes without retraining.

## References
- Kirkpatrick et al., Overcoming catastrophic forgetting in neural networks (EWC, PNAS 2017): https://www.pnas.org/doi/10.1073/pnas.1611835114
- van de Ven & Tolias, Three scenarios for continual learning: https://arxiv.org/abs/1904.07734
- Lopez-Paz & Ranzato, Gradient Episodic Memory (GEM): https://arxiv.org/abs/1706.08840
- Avalanche continual learning library: https://avalanche.continualai.org/
