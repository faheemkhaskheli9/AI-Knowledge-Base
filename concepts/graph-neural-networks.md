---
title: Graph neural networks (GNNs)
category: concepts
tags: [gnn, graph-neural-network, gcn, graphsage, gat, message-passing, node-classification, link-prediction, pytorch-geometric, dgl]
use_cases:
  - "detect fraud rings from who-pays-whom or shared-device relationships"
  - "predict properties of molecules or materials from their atom-bond graph"
  - "recommend items or friends by predicting missing links in a user-item graph"
  - "classify nodes in a network (accounts, papers, devices) using their neighbours"
status: draft
last_verified: 2026-10-03
sources:
  - https://pytorch-geometric.readthedocs.io/
  - https://www.dgl.ai/
  - https://arxiv.org/abs/1609.02907
  - https://arxiv.org/abs/1706.02216
  - https://distill.pub/2021/gnn-intro/
---

# Graph neural networks (GNNs)

## Summary
Graph neural networks learn from data shaped as nodes and edges: accounts and transactions, atoms and bonds, users and items, roads and junctions. Each layer lets every node gather ("message pass") information from its neighbours, so after k layers a node's representation reflects its k-hop neighbourhood. GNNs are the deep-learning tool for node classification, link prediction and whole-graph prediction, when the relationships carry signal that per-row features miss.

## Key concepts
- **Graph data.** Node features `X` (n × d), edges as an adjacency matrix or an `edge_index` list (2 × E), optional edge features and weights. Directed or undirected; homogeneous (one node type) or heterogeneous (users, items, merchants).
- **Message passing.** Each layer: every node aggregates its neighbours' vectors (mean, sum or max), combines them with its own, and applies a learned transform. Stacking k layers covers the k-hop neighbourhood.
- **Core layers.** **GCN** (degree-normalised mean of neighbours), **GraphSAGE** (sampled neighbours, works on unseen nodes, scales to large graphs), **GAT** (attention weights per neighbour), **GIN** (sum aggregation, most expressive for graph classification).
- **Tasks.** Node classification (is this account fraudulent?), link prediction (will this user buy this item?), graph classification/regression (is this molecule toxic?).
- **Transductive vs inductive.** Transductive models see the full graph during training and predict on its own nodes; inductive models (GraphSAGE) generalise to new nodes or new graphs.
- **Over-smoothing.** With many layers all node vectors converge to the same value; 2-3 layers is typical.
- **Neighbour sampling / mini-batching.** Big graphs do not fit in memory; train on sampled subgraphs around each batch of target nodes.

## When to use / scenarios
- **Fraud and AML:** transaction or shared-attribute graphs (same device, card, address) expose rings that per-transaction models miss.
- **Drug discovery and materials:** molecular property prediction from atom-bond graphs.
- **Recommendation:** user-item interaction graphs (PinSage-style link prediction); see [[recommender-systems]].
- **Networks and infrastructure:** traffic forecasting on road graphs, telecom or power-grid fault localisation, citation and knowledge-graph completion.
- **Not when:** relationships add little beyond row features; often you can get most of the gain with hand-made graph features (degree, PageRank, neighbour aggregates, connected-component size) fed to [[gradient-boosting-tabular]]. Try that baseline first. For knowledge-graph-augmented LLM answers see [[advanced-rag]].

## Setup & code
Production code usually uses PyTorch Geometric (`pip install torch_geometric`) or DGL. The minimal example below writes a two-layer GCN in plain PyTorch so the mechanics are visible.
```bash
pip install torch
```
```python
import torch
from torch import nn

torch.manual_seed(0)
# Two communities of 100 nodes: dense links inside, sparse links across.
n, half = 200, 100
labels = torch.cat([torch.zeros(half), torch.ones(half)]).long()
same = labels[:, None] == labels[None, :]
A = (torch.rand(n, n) < torch.where(same, 0.08, 0.005)).float()
A = torch.triu(A, 1); A = A + A.T                      # undirected, no self-loops
X = torch.randn(n, 8) + labels[:, None] * 0.3          # features: weak signal

# GCN propagation: D^-1/2 (A + I) D^-1/2
A_hat = A + torch.eye(n)
d = A_hat.sum(1).pow(-0.5)
A_norm = d[:, None] * A_hat * d[None, :]

class GCN(nn.Module):
    def __init__(self, d_in, d_hid, n_cls):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d_in, d_hid), nn.Linear(d_hid, n_cls)
    def forward(self, x, a):
        x = torch.relu(a @ self.l1(x))                 # aggregate neighbours
        return a @ self.l2(x)

train = torch.zeros(n, dtype=torch.bool)
train[torch.randperm(n)[:20]] = True                   # only 20 labelled nodes
for name, a in [("MLP (no edges)", torch.eye(n)), ("GCN", A_norm)]:
    torch.manual_seed(0)
    model = GCN(8, 16, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        loss = nn.functional.cross_entropy(model(X, a)[train], labels[train])
        opt.zero_grad(); loss.backward(); opt.step()
    acc = (model(X, a).argmax(1)[~train] == labels[~train]).float().mean()
    print(f"{name}: test accuracy {acc:.2f}")
```
With only 20 labels and weak node features, the MLP (same network with the graph replaced by the identity) stays well below the GCN, which reaches close to perfect accuracy by pooling evidence from neighbours. In PyTorch Geometric the same model is two `GCNConv` layers called with `edge_index`.

## Choosing / trade-offs
- **GCN vs GraphSAGE vs GAT.** GCN is a simple, strong baseline on small transductive graphs; GraphSAGE for large graphs and new nodes arriving (fraud, recommendations); GAT when neighbours differ in importance; GIN for graph-level classification.
- **PyTorch Geometric vs DGL.** Both are mature; PyG has the larger model zoo and is the common default in research code, DGL has strong distributed and heterogeneous-graph training. Pick the one your team or reference implementation uses.
- **GNN vs graph features + boosting.** Engineered graph features are cheaper to serve and explain; a GNN learns richer patterns but needs graph infrastructure at inference (fetching neighbourhoods in real time).
- **Graph construction is a modelling decision.** Which entities become nodes, which relations become edges, and edge time windows usually matter more than the layer type.

## Gotchas
- **Temporal leakage:** building the graph from all data, including edges created after the prediction time, leaks the future. Build the graph as of each prediction time.
- **Link-prediction leakage:** the edges you test on must be removed from the message-passing graph during training.
- Random node splits on graphs overstate accuracy when test nodes are densely linked to training nodes; consider time-based or community-based splits.
- High-degree hub nodes dominate sum aggregation; use mean or degree normalisation, and sample neighbours for huge hubs.
- Many layers cause over-smoothing; deeper is rarely better.
- Real-time serving needs the neighbourhood at request time; check the feature store or graph database can deliver it within the latency budget.

## Related
- [[neural-network-fundamentals]] - the layers and training loop GNNs build on.
- [[transformers-and-attention]] - attention is message passing over a fully connected graph; GAT applies it to graph edges.
- [[recommender-systems]] - link prediction on user-item graphs.
- [[advanced-rag]] - knowledge graphs combined with LLM retrieval.
- [[anomaly-detection]] - fraud detection without labels.
- [[gradient-boosting-tabular]] - the baseline with engineered graph features.

## References
- PyTorch Geometric documentation: https://pytorch-geometric.readthedocs.io/
- Deep Graph Library (DGL): https://www.dgl.ai/
- Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks: https://arxiv.org/abs/1609.02907
- Hamilton et al., GraphSAGE: https://arxiv.org/abs/1706.02216
- A Gentle Introduction to Graph Neural Networks (Distill): https://distill.pub/2021/gnn-intro/
