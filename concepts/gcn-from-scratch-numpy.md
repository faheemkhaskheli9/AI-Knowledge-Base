---
title: Graph convolutional network from scratch in NumPy (normalised adjacency, message passing, over-smoothing)
category: concepts
tags: [gcn, graph-convolutional-network, graph-neural-networks, message-passing, node-classification, semi-supervised-learning, over-smoothing, label-propagation, backpropagation, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement a two-layer GCN forward and backward pass by hand"
  - "classify nodes in a graph when only a few nodes are labelled"
  - "understand the D^-1/2 (A + I) D^-1/2 normalisation and why self-loops are added"
  - "explain over-smoothing and why GCNs are shallow in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1609.02907
  - https://arxiv.org/abs/1801.07606
  - https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html
---

# Graph convolutional network from scratch in NumPy (normalised adjacency, message passing, over-smoothing)

## Summary
A graph convolutional network (GCN) learns node representations by repeatedly averaging each node's features with its neighbours' and passing the result through a shared linear layer and a nonlinearity. One layer is `H' = ReLU(Â H W)`, where `Â` is the adjacency matrix with self-loops, normalised by node degree. With two layers each node sees its 2-hop neighbourhood, so a handful of labelled nodes can teach the whole graph. The NumPy version below passes a gradient check. On a 400-node graph with 4 communities and only 20 labelled nodes, it reaches 97.5% test accuracy, against 62.9% for the same network without the graph. It also shows over-smoothing: after 32 propagation steps every node's representation is the same.

## Key concepts
- **Normalised adjacency** `Â = D̃^{−1/2} (A + I) D̃^{−1/2}`, with `D̃` the degree matrix of `A + I`. Self-loops keep a node's own features in the average. The symmetric normalisation keeps feature scale stable whatever the degree, so hubs do not blow up.
- **Layer.** `H^{(l+1)} = σ(Â H^{(l)} W^{(l)})`: aggregate neighbours (`Â H`), then transform (`· W`). The same `W` is shared by every node, as a convolution kernel is shared by every pixel.
- **Two-layer model.** `softmax(Â ReLU(Â X W₁) W₂)`. Each output depends on nodes up to 2 hops away.
- **Semi-supervised loss.** Cross-entropy on the labelled nodes only. Gradients still reach unlabelled nodes' features through `Â`, which is how their structure is used.
- **Backward pass.** For `Z = Â H W`: `∂L/∂W = (Â H)ᵀ ∂L/∂Z` and `∂L/∂H = Âᵀ ∂L/∂Z Wᵀ`. It is an ordinary dense layer with an extra fixed matrix multiply.
- **Homophily.** GCNs assume linked nodes tend to share labels. Here 78% of edges join same-class nodes.
- **Over-smoothing.** Repeated averaging converges to the graph's stationary vector, so deep stacks make all nodes look alike. That is why most GCNs have 2-3 layers.

## When to use / scenarios
- Learning: the basic graph neural network layer, and a clear case of inductive bias from structure: same parameters, 63% vs 98% accuracy.
- Interviews: "what is message passing", "why normalise the adjacency", "why are GNNs shallow", "GCN vs GraphSAGE vs GAT".
- Node classification with few labels: citation or paper topics, fraud rings in transaction graphs, user interests in social graphs, protein function in interaction networks.
- Link prediction and graph classification build on the same layer with a different head ([[graph-neural-networks]]).
- Not for: graphs where neighbours tend to differ (heterophily; use models that separate self and neighbour features, such as GraphSAGE); very large graphs with a dense `Â` (use sparse matrices and neighbour sampling); data without meaningful edges, where a plain MLP or gradient boosting is simpler ([[gradient-boosting-tabular]]).

## Setup & code
`pip install numpy`. Runs in about 3 seconds on CPU (dense 400×400 adjacency).

```python
import numpy as np

rng = np.random.default_rng(0)

# synthetic citation-style graph: 4 communities, edges mostly within a community, weak noisy features
n, k, f = 400, 4, 50
y = np.repeat(np.arange(k), n // k)
p_in, p_out = 0.05, 0.005
same = y[:, None] == y[None]
upper = np.triu(rng.random((n, n)) < np.where(same, p_in, p_out), 1)
Adj = (upper | upper.T).astype(float)
X = rng.normal(size=(n, f)) + 0.4 * np.eye(k)[y] @ rng.normal(size=(k, f))   # class signal is weak per node

# semi-supervised split: 5 labelled nodes per class
train = np.concatenate([np.flatnonzero(y == c)[:5] for c in range(k)])
test = np.setdiff1d(np.arange(n), train)[::2]
val = np.setdiff1d(np.setdiff1d(np.arange(n), train), test)


def normalise(Adj):
    """A_hat = D^-1/2 (A + I) D^-1/2: add self-loops, then symmetric degree normalisation."""
    A = Adj + np.eye(len(Adj))
    d = A.sum(1) ** -0.5
    return d[:, None] * A * d[None]


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)


def model(params, A_hat, X, drop=0.0):
    """Two-layer GCN: softmax(A_hat relu(A_hat X W1) W2). A_hat = I gives a plain MLP."""
    W1, W2 = params
    mask = (rng.random(X.shape) > drop) / (1 - drop) if drop else 1
    Xd = X * mask
    AX = A_hat @ Xd
    H = np.maximum(AX @ W1, 0)
    AH = A_hat @ H
    return softmax(AH @ W2), (Xd, AX, H, AH)


def loss_grad(params, A_hat, X, idx, wd, drop):
    W1, W2 = params
    P, (Xd, AX, H, AH) = model(params, A_hat, X, drop)
    Y = np.eye(k)[y[idx]]
    loss = -np.log(P[idx][Y == 1]).mean() + wd * (W1 ** 2).sum()
    dZ = np.zeros_like(P); dZ[idx] = (P[idx] - Y) / len(idx)        # loss only on labelled nodes ...
    dW2 = AH.T @ dZ
    dH = (A_hat.T @ (dZ @ W2.T)) * (H > 0)                          # ... but gradient reaches their neighbours
    dW1 = AX.T @ dH + 2 * wd * W1
    return loss, [dW1, dW2]


def fit(A_hat, epochs=200, hidden=16, lr=0.01, wd=5e-4, drop=0.5, seed=0):
    r = np.random.default_rng(seed)
    params = [r.normal(0, np.sqrt(2 / (f + hidden)), (f, hidden)), r.normal(0, np.sqrt(2 / (hidden + k)), (hidden, k))]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    best, best_params = -1, None
    for t in range(1, epochs + 1):
        _, g = loss_grad(params, A_hat, X, train, wd, drop)
        for i in range(2):                                           # Adam
            m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
            params[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        acc = (model(params, A_hat, X)[0][val].argmax(1) == y[val]).mean()
        if acc > best:
            best, best_params = acc, [p.copy() for p in params]      # early stopping on validation
    return best_params


A_hat = normalise(Adj)

# gradient check through the graph
p0 = [rng.normal(size=(f, 16)) * 0.1, rng.normal(size=(16, k)) * 0.1]
_, g0 = loss_grad(p0, A_hat, X, train, 0.0, 0.0)
eps = 1e-6
p0[0][3, 5] += eps; lp = loss_grad(p0, A_hat, X, train, 0.0, 0.0)[0]
p0[0][3, 5] -= 2 * eps; lm = loss_grad(p0, A_hat, X, train, 0.0, 0.0)[0]; p0[0][3, 5] += eps
print(f"grad check dW1[3,5]: analytic {g0[0][3, 5]:.8f} | numeric {(lp - lm) / (2 * eps):.8f}")

print(f"graph: {n} nodes, {int(Adj.sum() / 2)} edges, mean degree {Adj.sum(1).mean():.1f}, "
      f"{(Adj * same).sum() / Adj.sum():.0%} of edges within a class; {len(train)} labelled nodes")
for name, A in [("MLP (no graph)", np.eye(n)), ("GCN", A_hat)]:
    accs = [(model(fit(A, seed=s), A, X)[0][test].argmax(1) == y[test]).mean() for s in range(5)]
    print(f"{name:15s} test accuracy {np.mean(accs):.3f} +/- {np.std(accs):.3f} (5 seeds)")

# label propagation alone (no features): A_hat^K applied to the one-hot training labels
Y0 = np.zeros((n, k)); Y0[train, y[train]] = 1
for K in [2, 10]:
    print(f"label propagation, {K:2d} hops: test accuracy {(np.linalg.matrix_power(A_hat, K) @ Y0)[test].argmax(1).__eq__(y[test]).mean():.3f}")

# over-smoothing: stacking many A_hat hops makes node representations indistinguishable
for K in [1, 2, 8, 32, 128]:
    Z = np.linalg.matrix_power(A_hat, K) @ X
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    sim = Z @ Z.T
    print(f"{K:3d} propagation steps: mean cosine similarity same class {sim[same].mean():.3f} | different class {sim[~same].mean():.3f}")
```

Output (numpy 2.5):
```
grad check dW1[3,5]: analytic 0.00848588 | numeric 0.00848588
graph: 400 nodes, 1312 edges, mean degree 6.6, 78% of edges within a class; 20 labelled nodes
MLP (no graph)  test accuracy 0.629 +/- 0.020 (5 seeds)
GCN             test accuracy 0.975 +/- 0.006 (5 seeds)
label propagation,  2 hops: test accuracy 0.695
label propagation, 10 hops: test accuracy 0.842
  1 propagation steps: mean cosine similarity same class 0.419 | different class 0.112
  2 propagation steps: mean cosine similarity same class 0.659 | different class 0.272
  8 propagation steps: mean cosine similarity same class 0.975 | different class 0.885
 32 propagation steps: mean cosine similarity same class 1.000 | different class 1.000
128 propagation steps: mean cosine similarity same class 1.000 | different class 1.000
```

The analytic gradient through both `Â` multiplications matches the numerical one. Each node's features carry only a weak class signal, so an MLP trained on 20 labelled nodes reaches 62.9%. The GCN, with the same parameters and training, reaches 97.5%: averaging over neighbours, most of which share the class, cancels the per-node noise. Label propagation uses the graph but not the features and reaches 69.5% at 2 hops and 84.2% at 10, so the GCN's result needs both. Over-smoothing is visible directly. One or two averaging steps make same-class nodes more alike (cosine 0.42, then 0.66) faster than different-class nodes. By 8 steps different classes are at 0.885, and by 32 every node has the same direction (1.000 for both), so no classifier can separate them.

## Choosing / trade-offs
- **GCN vs GraphSAGE vs GAT.** GCN is the simplest and works well on homophilous graphs. GraphSAGE concatenates self and neighbour features and samples neighbours, so it scales and copes better with heterophily. GAT learns attention weights per edge instead of fixed degree weights.
- **Depth.** 2 layers is the usual default. Need more reach? Use residual connections, jumping-knowledge (combine all layers) or APPNP-style propagation, rather than stacking plain GCN layers.
- **Dense vs sparse.** A dense `Â` is fine up to a few thousand nodes. Beyond that use sparse matrices (`scipy.sparse`, PyTorch Geometric, DGL) and mini-batching by neighbour sampling.
- **GNN vs graph features + boosting.** Hand-made graph features (degree, PageRank, neighbour label counts) fed to gradient boosting are a strong baseline on tabular-plus-graph problems and are cheaper to serve.

## Gotchas
- Forgetting self-loops makes a node's own features disappear after one layer: `Â` without `+ I` only averages neighbours.
- Using `D⁻¹A` (row normalisation) instead of the symmetric form changes the weighting of hubs; both work but do not mix them up when matching a paper's numbers.
- Transductive setting: the test nodes' features and edges are present during training, only their labels are hidden. Results do not transfer to new graphs or new nodes without retraining (GraphSAGE is inductive).
- Leakage: random splits on graphs with duplicated or near-duplicate nodes, or time-ordered graphs, overstate accuracy ([[data-leakage-and-validation-splits]]).
- Isolated nodes have degree 0 before self-loops; with self-loops they reduce to an MLP on their own features, which is the correct behaviour.
- Early stopping on a validation set matters with only 20 labels: the training loss reaches near zero quickly while test accuracy keeps moving.

## Related
- [[graph-neural-networks]] - GCN, GraphSAGE, GAT in PyTorch Geometric, plus link prediction and graph classification.
- [[cnn-from-scratch-numpy]] - the grid version of the same shared-weight neighbour aggregation.
- [[neural-network-from-scratch-numpy]] - the dense layer and softmax cross-entropy reused here.
- [[semi-supervised-and-active-learning]] - learning from few labels, including label propagation.
- [[batchnorm-and-dropout-from-scratch-numpy]] - dropout as applied to the input features here.
- [[transformer-block-from-scratch-numpy]] - self-attention as message passing on a fully connected graph.

## References
- Kipf and Welling (2016), "Semi-Supervised Classification with Graph Convolutional Networks": https://arxiv.org/abs/1609.02907
- Li, Han and Wu (2018), "Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning" (over-smoothing): https://arxiv.org/abs/1801.07606
- PyTorch Geometric `GCNConv`: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html
