---
title: Federated learning and differential privacy
category: ml
tags: [federated-learning, fedavg, differential-privacy, dp-sgd, privacy, flower, opacus, non-iid, secure-aggregation]
use_cases:
  - "train one model across hospitals or banks without pooling their raw data"
  - "learn from data on users' phones without uploading it"
  - "limit how much a trained model can reveal about any single person"
  - "release aggregate statistics with a formal privacy guarantee"
  - "decide whether federated training is worth the engineering cost"
status: draft
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/1602.05629
  - https://arxiv.org/abs/1607.00133
  - https://flower.ai/docs/framework/
  - https://opacus.ai/
---

# Federated learning and differential privacy

## Summary
Federated learning (FL) trains a shared model while the data stays with its
owners: each client trains on its own data and sends only model updates to a
server, which averages them. Differential privacy (DP) is a separate,
mathematical guarantee that a model or statistic changes very little whether
any one person's record is included, which limits what can be inferred about
that person. FL alone is not private (updates can leak data); production
systems combine FL with DP and secure aggregation.

## Key concepts
- FedAvg: each round, the server sends the global weights, clients run a few
  local epochs, the server averages the returned weights weighted by client
  data size.
- Cross-silo (a few organizations, reliable servers) vs cross-device
  (millions of phones, intermittent, sample a fraction per round).
- Non-IID data: each client sees a skewed slice (one hospital's population, one
  user's typing). It is the main reason FL accuracy trails central training.
  Mitigations: FedProx, more rounds with fewer local steps, personalization
  layers fine-tuned per client.
- Secure aggregation: cryptographic protocol so the server sees only the sum
  of updates, not any one client's update.
- (epsilon, delta)-DP: epsilon bounds how much one record can change output
  probabilities. Smaller is more private; single digits are common in
  practice, but there is no universally safe value.
- DP-SGD: clip each example's gradient to norm C, add Gaussian noise with std
  `sigma * C` to the summed gradient. A privacy accountant turns sigma, the
  sampling rate and the number of steps into epsilon.
- User-level vs example-level DP: in FL you usually want to protect a whole
  client's contribution (clip and noise per client update), not one row.
- Attacks DP defends against: membership inference, training-data extraction,
  gradient inversion.

## When to use / scenarios
- Healthcare: multi-hospital imaging or EHR models where data-sharing
  agreements forbid pooling ([[healthcare]]).
- Finance: fraud/AML models across banks or branches ([[finance]]).
- Mobile/edge: keyboard prediction, on-device personalization ([[edge-on-device]]).
- Statistics release: counts and means over sensitive data with DP noise
  (OpenDP, Google's DP library).
- NOT for: data you are allowed to centralize. Central training is simpler,
  more accurate and easier to debug. Check first whether pseudonymization plus
  a data-processing agreement solves the legal problem
  ([[ai-security-privacy-compliance]]).

## Setup & code
Plain PyTorch, no framework, so the mechanics are visible. Five clients each
hold mostly two digit classes (strongly non-IID).
```bash
pip install torch scikit-learn
```
```python
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32); y = torch.tensor(y)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# Non-IID split: each of 5 clients holds mostly 2 digit classes
clients = [(X_tr[(y_tr // 2) == k], y_tr[(y_tr // 2) == k]) for k in range(5)]

def make_model():
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))

def accuracy(m):
    with torch.no_grad():
        return (m(X_te).argmax(1) == y_te).float().mean().item()

def local_train(global_model, Xc, yc, epochs=1, dp=None):
    m = make_model(); m.load_state_dict(global_model.state_dict())
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for _ in range(epochs):
        for i in torch.randperm(len(Xc)).split(32):
            opt.zero_grad()
            if dp is None:
                nn.functional.cross_entropy(m(Xc[i]), yc[i]).backward()
            else:  # DP-SGD: clip each example's gradient, add Gaussian noise to the sum
                clip, sigma = dp
                summed = [torch.zeros_like(p) for p in m.parameters()]
                for j in i:
                    g = torch.autograd.grad(nn.functional.cross_entropy(m(Xc[j:j+1]), yc[j:j+1]), m.parameters())
                    norm = torch.sqrt(sum((t ** 2).sum() for t in g))
                    for s, t in zip(summed, g):
                        s += t * min(1.0, clip / (norm.item() + 1e-12))
                for p, s in zip(m.parameters(), summed):
                    p.grad = (s + sigma * clip * torch.randn_like(s)) / len(i)
            opt.step()
    return m.state_dict(), len(Xc)

def fedavg(rounds, dp=None):
    torch.manual_seed(0)
    g = make_model()
    for _ in range(rounds):
        updates = [local_train(g, Xc, yc, dp=dp) for Xc, yc in clients]
        total = sum(n for _, n in updates)
        g.load_state_dict({k: sum(sd[k] * n / total for sd, n in updates) for k in updates[0][0]})
    return accuracy(g)

print(f"FedAvg 20 rounds: {fedavg(20):.3f}")
print(f"FedAvg 20 rounds + DP-SGD (clip=1, sigma=1): {fedavg(20, dp=(1.0, 1.0)):.3f}")
```
Output (torch 2.13.0 CPU), test accuracy on all 10 digits:

| Setup | Accuracy |
|---|---|
| Central training, 20 epochs (reference) | 0.957 |
| One client alone, 20 epochs | 0.180 |
| FedAvg, 20 rounds x 1 local epoch | 0.793 |
| FedAvg + DP-SGD (clip 1, sigma 1) | 0.576 |

FedAvg recovers most of what one client cannot learn alone, but non-IID data
leaves a clear gap to central training, and DP noise costs a lot more on a
dataset this small. The example does not compute epsilon; use a privacy
accountant (Opacus's RDP accountant) for that. For real projects, use Flower
(`pip install flwr`) for the client/server plumbing and Opacus
(`pip install opacus`, `PrivacyEngine`) for vectorized per-sample gradients
and accounting instead of the per-example loop above.

## Choosing / trade-offs
- Few organizations, legal barrier to pooling: cross-silo FL (Flower, NVIDIA
  FLARE). Consider simpler options first: train at each site and share only
  evaluation results, or use a trusted research environment.
- Millions of devices: cross-device FL with client sampling and secure
  aggregation; very high engineering cost.
- Need a privacy claim you can defend: DP with a stated epsilon. FL without DP
  is a data-minimization measure, not a privacy guarantee.
- DP utility cost shrinks with more data: millions of records can afford
  small epsilon; thousands usually cannot.
- Synthetic data as an alternative to sharing: only private if generated with
  DP ([[data-labeling-and-synthetic-data]]).

## Gotchas
- Model updates leak training data (gradient inversion recovers images and
  text from small batches). Do not call plain FL "privacy-preserving".
- Batch norm statistics differ per client on non-IID data; GroupNorm or
  keeping BN local (FedBN) works better.
- DP-SGD needs per-example gradients; BatchNorm mixes examples and is not
  compatible. Opacus will refuse it or ask for GroupNorm.
- Epsilon grows with every training step and every release. Hyperparameter
  tuning on the private data also spends privacy budget.
- Client dropout and stragglers: the server must handle partial rounds.
- Evaluate per client, not only globally: the average hides clients the model fails.
- A malicious client can poison the global model; robust aggregation (median,
  trimmed mean) helps against a few attackers.

## Related
- [[ai-security-privacy-compliance]] - legal side: GDPR/HIPAA, DPAs, data minimization.
- [[edge-on-device]] - running and training models on devices.
- [[deep-learning-training]] - optimizers and normalization choices referenced above.
- [[data-labeling-and-synthetic-data]] - synthetic data as an alternative to sharing.
- [[healthcare]], [[finance]] - main cross-silo use cases.

## References
- McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg): https://arxiv.org/abs/1602.05629
- Abadi et al., Deep Learning with Differential Privacy (DP-SGD): https://arxiv.org/abs/1607.00133
- Flower framework docs: https://flower.ai/docs/framework/
- Opacus (DP training for PyTorch): https://opacus.ai/
