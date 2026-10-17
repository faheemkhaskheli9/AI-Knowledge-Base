---
title: Graph ML for fraud and AML detection
category: ml
tags: [fraud, aml, anti-money-laundering, graph, gnn, graph-features, entity-resolution, pytorch-geometric, networkx, transaction-graph]
use_cases:
  - "find fraud rings that share devices, cards, addresses or phone numbers"
  - "add graph features to an existing transaction fraud model"
  - "detect money-laundering patterns (fan-in, fan-out, cycles) in a payment network"
  - "decide whether a GNN is worth it over gradient boosting with graph features"
status: draft
last_verified: 2026-10-15
sources:
  - https://arxiv.org/abs/1908.02591
  - https://arxiv.org/abs/2306.16424
  - https://pytorch-geometric.readthedocs.io/
  - https://networkx.org/documentation/stable/
  - https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html
---

# Graph ML for fraud and AML detection

## Summary
Fraudsters and money launderers reuse resources (devices, cards, emails,
addresses, mule accounts) and move money through chains of accounts, so the
relationships between entities carry signal that per-transaction features
miss. Graph ML turns those relationships into either engineered graph
features for a gradient-boosted model (the usual first step) or a graph
neural network (GNN) trained on the graph itself. The hard parts are building
the graph correctly over time, avoiding label leakage through neighbours, and
explaining alerts to investigators.

## Key concepts
- **Graph construction.**
  - *Transaction graph:* accounts as nodes, payments as directed, timestamped
    edges (amount, channel). Used for AML flow patterns.
  - *Shared-attribute (identity) graph:* accounts or applications linked
    through shared devices, IPs, cards, emails, phones, addresses. Used for
    fraud rings, synthetic identities, promo abuse.
  - *Heterogeneous graph:* several node types (customer, card, merchant,
    device) and edge types; GNN libraries support this directly.
- **Entity resolution first.** Normalise and match identifiers (address
  formatting, email aliases) or the graph fragments and rings look unrelated.
- **Graph features for tabular models.** Degree and in/out degree, connected
  component size, number of distinct accounts per device, PageRank, shared-
  neighbour counts, k-hop aggregates (sum/mean amount in and out), and
  *neighbour label rates* (fraction of known-fraud neighbours, computed from
  training-time labels only).
- **AML typologies as subgraph patterns.** Fan-out, fan-in, scatter-gather,
  gather-scatter, cycles, and long chains of quick pass-through transfers;
  detect with rules, motif counts or GNNs.
- **GNNs.** GraphSAGE-style inductive models score new nodes from sampled
  neighbourhoods; temporal GNNs use edge timestamps. See
  [[graph-neural-networks]] for the layers.
- **Label propagation / guilt by association.** Spread known-fraud scores
  over edges ([[label-propagation-from-scratch]]); cheap and explainable.
- **Investigation view.** Alerts are triaged by analysts who need the
  subgraph ("these 7 accounts share 2 devices and moved funds in a ring"),
  not just a score.

## When to use / scenarios
- **Card-not-present and account-takeover fraud:** add device/IP/card
  shared-attribute features to the existing GBM ([[finance]]).
- **Fraud rings and synthetic identities** in lending, insurance claims,
  marketplace sellers, promo abuse: connected components and community
  detection over shared identifiers, then review the whole cluster.
- **AML transaction monitoring:** flow patterns between accounts and banks;
  rules plus graph scores feeding a case-management queue.
- **Mule account detection:** accounts that receive from many and forward
  quickly (high in-degree, short dwell time).
- **Not when:** entities rarely share anything (graph is mostly isolated
  nodes), or the signal is in the transaction itself; a plain
  [[gradient-boosting-tabular]] model with velocity features is enough.
  Unlabelled cases start with [[anomaly-detection]] instead.

## Setup & code
```bash
pip install numpy scipy scikit-learn       # this example
pip install networkx torch torch_geometric # production graph work and GNNs
```
Synthetic identity graph: legit accounts share few devices, fraud rings
share many. Compare a GBM with and without graph features, computing
neighbour-label features from training labels only.
```python
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

rng = np.random.default_rng(0)
n_acc, n_dev = 4000, 3500
fraud = np.zeros(n_acc, int)
acc, dev = [], []
for a in range(n_acc):                                    # legit: 1-2 own-ish devices
    for d in rng.choice(n_dev, rng.integers(1, 3), replace=False):
        acc.append(a); dev.append(d)
ring_members = rng.choice(n_acc, 200, replace=False)      # 20 rings of 10 accounts
for ring in ring_members.reshape(20, 10):
    fraud[ring] = 1
    ring_devs = rng.choice(n_dev, 3, replace=False)       # ring shares a few devices
    for a in ring:
        for d in rng.choice(ring_devs, 2, replace=False):
            acc.append(a); dev.append(d)
acc, dev = np.array(acc), np.array(dev)

# Bipartite account-device graph -> account-account graph via shared devices
B = coo_matrix((np.ones(len(acc)), (acc, dev)), shape=(n_acc, n_dev)).tocsr()
B.data[:] = 1
AA = (B @ B.T).tocsr(); AA.setdiag(0); AA.eliminate_zeros(); AA.data[:] = 1

# Tabular features: weak per-account signal (e.g. account age, amount)
X_tab = rng.normal(size=(n_acc, 3)) + fraud[:, None] * 0.4

train = rng.random(n_acc) < 0.6                           # labels known for 60% only
y_known = np.where(train, fraud, 0)                        # test labels hidden

_, comp = connected_components(AA, directed=False)
comp_size = np.bincount(comp)[comp]
deg = np.asarray(AA.sum(1)).ravel()
accs_per_dev = np.asarray(B.sum(0)).ravel()
max_dev_share = np.array([accs_per_dev[B[i].indices].max() for i in range(n_acc)])
known_nb = AA @ train.astype(float)
fraud_nb = AA @ y_known.astype(float)                     # leakage-safe: train labels only
nb_rate = np.divide(fraud_nb, known_nb, out=np.zeros(n_acc), where=known_nb > 0)
X_graph = np.column_stack([X_tab, deg, comp_size, max_dev_share, nb_rate])

for name, X in [("tabular only", X_tab), ("tabular + graph", X_graph)]:
    m = HistGradientBoostingClassifier(random_state=0).fit(X[train], fraud[train])
    ap = average_precision_score(fraud[~train], m.predict_proba(X[~train])[:, 1])
    print(f"{name:16s} average precision on held-out accounts: {ap:.3f}")
```
The synthetic rings are easy to separate, so the gap is exaggerated; on real
data, measure the lift on a time-based split before adding graph machinery.

A GNN version (PyTorch Geometric, sketch) on the same account graph:
<!-- skip-check: needs torch and torch_geometric -->
```python
import torch
from torch_geometric.nn import SAGEConv

class Sage(torch.nn.Module):
    def __init__(self, d_in, d_h=64):
        super().__init__()
        self.c1, self.c2 = SAGEConv(d_in, d_h), SAGEConv(d_h, 2)
    def forward(self, x, edge_index):
        return self.c2(self.c1(x, edge_index).relu(), edge_index)

# edge_index = torch.tensor(np.vstack(AA.nonzero()), dtype=torch.long)
# train with cross-entropy on train-mask nodes; evaluate on the held-out mask
```

## Choosing / trade-offs
| Option | Pros | Cons |
|---|---|---|
| Rules on graph (component size, shared devices > k) | Explainable, fast to ship | Brittle, many false positives |
| Graph features + GBM | Reuses existing pipeline, explainable with SHAP, strong baseline | Features fixed by hand; k-hop aggregates costly online |
| Label propagation | No training, investigator-friendly | Needs some seed labels; spreads errors |
| GNN (GraphSAGE, GAT, temporal GNN) | Learns multi-hop patterns, inductive | Harder to serve in real time, explain and validate; often small gain over good features |

- **Batch vs real time:** graph features for card authorisation must be
  computed in milliseconds; precompute per-entity aggregates in a feature
  store and update incrementally, run heavier graph jobs (components,
  PageRank, GNN embeddings) in batch.
- **Graph store:** a graph database helps investigators explore; training
  usually reads edge lists from the warehouse.
- **Explainability:** regulators and investigators need reasons; prefer
  features and subgraph evidence over opaque embeddings for decisions.

## Gotchas
- **Neighbour label leakage:** a "fraud neighbour rate" computed with all
  labels lets test labels leak into features. Use only labels available at
  the prediction time and in the training fold.
- **Temporal leakage:** build the graph as of the event time; edges created
  after the transaction must not be in its features. Split by time
  ([[data-leakage-and-validation-splits]]).
- **Hub nodes:** shared public Wi-Fi IPs, payment processors, default
  devices or placeholder addresses ("N/A") connect thousands of unrelated
  accounts; cap or drop super-nodes.
- **Label quality:** many laundering cases are never detected, so "negative"
  labels are noisy; synthetic AML datasets exist partly for this reason
  ([[label-noise-and-data-cleaning]]).
- **Extreme imbalance:** evaluate with precision at alert budget and average
  precision, not accuracy or ROC AUC alone ([[imbalanced-data]]).
- **Adversarial drift:** rings change devices and patterns once blocked;
  monitor and retrain ([[online-learning-and-concept-drift]]).
- **Over-smoothing and over-linking:** deep GNNs or loose entity resolution
  blur fraud and legit nodes together.
- **Fairness and privacy:** shared-address links can punish households or
  shared housing; device and location data are personal data.

## Related
- [[graph-neural-networks]] - GNN layers, message passing, PyG/DGL.
- [[gcn-from-scratch-numpy]] - a GCN built by hand.
- [[label-propagation-from-scratch]] - guilt-by-association scoring.
- [[gradient-boosting-tabular]] - the model that consumes graph features.
- [[anomaly-detection]] - unlabelled fraud signals.
- [[imbalanced-data]] - rare-positive evaluation and thresholds.
- [[finance]] - wider fraud, credit and AML context.
- [[clustering]] - community detection and grouping of suspicious entities.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/data/graph-databases.md

## References
- Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics" (Elliptic dataset): https://arxiv.org/abs/1908.02591
- Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models", NeurIPS 2023 Datasets and Benchmarks: https://arxiv.org/abs/2306.16424
- PyTorch Geometric documentation: https://pytorch-geometric.readthedocs.io/
- NetworkX documentation: https://networkx.org/documentation/stable/
- SciPy sparse graph routines: https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html
- Hamilton et al., GraphSAGE: https://arxiv.org/abs/1706.02216
