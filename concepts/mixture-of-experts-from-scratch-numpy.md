---
title: Mixture of experts from scratch in NumPy (top-k router, sparse dispatch, load-balancing loss)
category: concepts
tags: [mixture-of-experts, moe, sparse-moe, top-k-routing, gating, router, load-balancing, auxiliary-loss, conditional-computation, switch-transformer, backpropagation, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement a sparse mixture-of-experts layer with top-k routing, forward and backward, in NumPy"
  - "see why an MoE has many more parameters than it uses per token, and what that buys"
  - "understand expert collapse and how the Switch-Transformer load-balancing loss prevents it"
  - "explain router gradients, gate renormalisation and the auxiliary loss in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1701.06538
  - https://arxiv.org/abs/2101.03961
  - https://arxiv.org/abs/2401.04088
---

# Mixture of experts from scratch in NumPy (top-k router, sparse dispatch, load-balancing loss)

## Summary
A sparse mixture-of-experts (MoE) layer replaces one feed-forward block with E smaller "experts" and a router. For each token, the router picks the top k experts and mixes their outputs with its softmax weights. Total parameters grow with E, while compute per token grows only with k. That is how models such as Switch Transformer and Mixtral reach large parameter counts at a fixed inference cost. Below, an 8-expert, top-2 MoE learns a regression task with four regimes. It reaches a test MSE of 0.055 while using 352 parameters per token, against 0.158 for a dense MLP with 1280 parameters. Without a balancing term, two of the eight experts receive no tokens at all. A Switch-style auxiliary loss with coefficient 0.1 spreads the load almost evenly (aux 1.02, where 1.00 is perfectly uniform), but test MSE rises to 0.077. Balance and quality pull against each other.

## Key concepts
- **Router (gate).** A linear layer and a softmax over experts: `p = softmax(x W_g)`. Keep the top-k probabilities and renormalise them to sum to 1 (as Mixtral does). The layer output is `y = Σ_{i∈topk} g_i · expert_i(x)`.
- **Sparse dispatch.** Each expert runs only on the tokens routed to it. Compute per token is k experts plus the router, whatever E is.
- **Router gradients.** Gradients reach `W_g` only through the selected gates, via `dL/dg_i = dL/dy · expert_i(x)`. The top-k choice itself is not differentiable. The router learns to raise the weight of experts that helped and, through the softmax, to lower the rest.
- **Expert collapse.** Early on, a few experts get slightly more tokens, train faster, and so get chosen even more. That rich-get-richer loop leaves some experts dead.
- **Load-balancing loss (Switch Transformer).** `L_aux = E · Σ_e f_e · P_e`, where `f_e` is the fraction of routing assignments that go to expert e (not differentiable, treated as a constant) and `P_e` is the mean router probability for e. It equals 1 when both are uniform and grows as routing concentrates. Add it to the task loss with a small coefficient.
- **Capacity.** At scale, each expert gets a fixed buffer of `capacity_factor · tokens · k / E` slots, and overflow tokens are dropped (they pass through the residual connection). This example has no capacity limit.

## When to use / scenarios
- Learning: conditional computation, non-differentiable routing decisions, and why auxiliary losses exist.
- Interviews: "how does Mixtral have 47B parameters but run like a 13B model", "what is expert collapse", "why is MoE harder to train and serve".
- Practice: scaling a transformer's parameters at a fixed FLOP budget (the FFN block becomes the MoE), multi-domain or multilingual models where experts specialise, and serving open MoE models (Mixtral, DeepSeek, Qwen MoE). See [[mixture-of-experts]] for using real MoE LLMs.
- Not for: small models or single-GPU training, where a dense model of the same compute is simpler and usually as good; memory-bound deployment (all experts must stay in memory even though few run); or tasks with too little data to train E experts.

## Setup & code
`pip install numpy`. Runs in about 6 seconds on CPU. Each expert is `ReLU(x W1) W2`, and a constant input column acts as the bias.

```python
import numpy as np


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class MoE:
    """Sparse mixture of experts: a softmax router picks the top-k of E linear-ReLU-linear experts per token."""

    def __init__(self, d, h, n_experts, k, rng):
        self.k, self.E = k, n_experts
        self.Wg = rng.normal(0, 0.01, (d, n_experts))
        self.W1 = rng.normal(0, np.sqrt(2 / d), (n_experts, d, h))
        self.W2 = rng.normal(0, np.sqrt(1 / h), (n_experts, h, 1))

    def forward(self, x):
        T = len(x)
        p = softmax(x @ self.Wg)                                  # router probabilities (T, E)
        top = np.argsort(-p, axis=1)[:, :self.k]                  # chosen experts (T, k)
        gate = np.take_along_axis(p, top, 1)
        gate = gate / gate.sum(1, keepdims=True)                  # renormalise over the k chosen
        y = np.zeros((T, 1))
        cache = []
        for e in range(self.E):                                   # each expert sees only its tokens
            tok, slot = np.nonzero(top == e)
            if len(tok) == 0:
                cache.append(None); continue
            h = np.maximum(0, x[tok] @ self.W1[e])
            out = h @ self.W2[e]
            y[tok] += gate[tok, slot, None] * out
            cache.append((tok, slot, h, out))
        self.c = (x, p, top, gate, cache)
        return y

    def aux_loss(self):
        """Switch-Transformer load-balancing loss E * sum_e f_e * P_e (minimum 1.0 when uniform)."""
        x, p, top, *_ = self.c
        f = np.bincount(top.ravel(), minlength=self.E) / top.size  # fraction of assignments
        P = p.mean(0)                                              # mean router probability
        return self.E * np.sum(f * P), f, P

    def backward(self, dy, aux_coef):
        x, p, top, gate, cache = self.c
        T = len(x)
        g = {"W1": np.zeros_like(self.W1), "W2": np.zeros_like(self.W2)}
        dgate = np.zeros_like(gate)
        for e, c in enumerate(cache):
            if c is None: continue
            tok, slot, h, out = c
            dout = gate[tok, slot, None] * dy[tok]
            dgate[tok, slot] = np.sum(dy[tok] * out, 1)
            g["W2"][e] = h.T @ dout
            dh = (dout @ self.W2[e].T) * (h > 0)
            g["W1"][e] = x[tok].T @ dh
        # renormalised gate g_i = p_i / s: dp_i = (dg_i - sum_j dg_j g_j) / s
        s = np.take_along_axis(p, top, 1).sum(1, keepdims=True)
        dp_sel = (dgate - np.sum(dgate * gate, 1, keepdims=True)) / s
        dp = np.zeros_like(p)
        np.put_along_axis(dp, top, dp_sel, 1)
        f = np.bincount(top.ravel(), minlength=self.E) / top.size
        dp += aux_coef * self.E * f / T                            # f is treated as a constant
        dz = p * (dp - np.sum(dp * p, 1, keepdims=True))           # softmax backward
        g["Wg"] = x.T @ dz
        return g


def make_data(n, rng):
    """4 regimes: the sign pattern of (x0, x1) selects a different target function of x2."""
    x = rng.normal(size=(n, 3))
    q = (x[:, 0] > 0) * 2 + (x[:, 1] > 0)
    fns = [np.sin(3 * x[:, 2]), x[:, 2] ** 2 - 1, -2 * x[:, 2], np.abs(x[:, 2]) - 0.8]
    y = np.choose(q, fns)
    return np.hstack([x, np.ones((n, 1))]), y[:, None], q     # constant column = bias for every layer input


def train(n_experts, k, aux_coef, hidden, steps=3000, seed=0):
    rng = np.random.default_rng(seed)
    X, Y, _ = make_data(4000, rng)
    model = MoE(4, hidden, n_experts, k, rng)
    params = ["Wg", "W1", "W2"]
    mom = {n: np.zeros_like(getattr(model, n)) for n in params}
    vel = {n: np.zeros_like(getattr(model, n)) for n in params}
    for t in range(1, steps + 1):
        idx = rng.integers(0, len(X), 256)
        y = model.forward(X[idx])
        grads = model.backward(2 * (y - Y[idx]) / len(idx), aux_coef)
        for n in params:                                            # Adam
            mom[n] = 0.9 * mom[n] + 0.1 * grads[n]
            vel[n] = 0.999 * vel[n] + 0.001 * grads[n] ** 2
            upd = 0.01 * (mom[n] / (1 - 0.9**t)) / (np.sqrt(vel[n] / (1 - 0.999**t)) + 1e-8)
            setattr(model, n, getattr(model, n) - upd)
    Xt, Yt, _ = make_data(4000, np.random.default_rng(99))
    mse = np.mean((model.forward(Xt) - Yt) ** 2)
    aux, f, _ = model.aux_loss()
    return model, mse, aux, f


# gradient check of the router on a tiny model
rng = np.random.default_rng(1)
m = MoE(3, 4, 4, 2, rng)
m.Wg = rng.normal(0, 1, m.Wg.shape)
x = rng.normal(size=(6, 3))
loss = lambda: np.sum(m.forward(x) ** 2) / 2 + 0.1 * m.aux_loss()[0]
an = m.backward(m.forward(x), 0.1)["Wg"]
num = np.zeros_like(m.Wg)
for i in np.ndindex(m.Wg.shape):
    m.Wg[i] += 1e-6; lp = loss(); m.Wg[i] -= 2e-6; lm = loss(); m.Wg[i] += 1e-6
    num[i] = (lp - lm) / 2e-6
print(f"router gradient check, max abs diff: {np.max(np.abs(an - num)):.1e}")

print("model                          params  active/token  test MSE  aux    expert load")
for name, E, k, aux_c, h in [("dense MLP, h=64", 1, 1, 0.0, 64),
                             ("dense MLP, h=256", 1, 1, 0.0, 256),
                             ("MoE 8 experts top-2, no aux", 8, 2, 0.0, 32),
                             ("MoE 8 experts top-2, aux 0.01", 8, 2, 0.01, 32),
                             ("MoE 8 experts top-2, aux 0.1", 8, 2, 0.1, 32)]:
    model, mse, aux, f = train(E, k, aux_c, h)
    n_par = model.Wg.size * (E > 1) + model.W1.size + model.W2.size
    active = model.Wg.size * (E > 1) + k * (model.W1[0].size + model.W2[0].size)
    print(f"{name:<30} {n_par:>6}  {active:>12}  {mse:>8.4f}  {aux:.2f}   {np.round(f, 2).tolist()}")

model, *_ = train(8, 2, 0.01, 32)
Xt, _, qt = make_data(4000, np.random.default_rng(99))
model.forward(Xt)
top1 = model.c[2][:, 0]
print("\nmost-used top-1 expert per regime (share of tokens), aux 0.01:")
for r in range(4):
    c = np.bincount(top1[qt == r], minlength=8)
    print(f"  regime {r}: expert {c.argmax()} ({c.max() / c.sum():.0%})")
```

Output (numpy 2.5):
```
router gradient check, max abs diff: 2.0e-10
model                          params  active/token  test MSE  aux    expert load
dense MLP, h=64                   320           320    0.2057  1.00   [1.0]
dense MLP, h=256                 1280          1280    0.1577  1.00   [1.0]
MoE 8 experts top-2, no aux      1312           352    0.0549  1.35   [0.0, 0.09, 0.16, 0.16, 0.14, 0.22, 0.21, 0.0]
MoE 8 experts top-2, aux 0.01    1312           352    0.0553  1.25   [0.05, 0.09, 0.2, 0.2, 0.02, 0.2, 0.09, 0.14]
MoE 8 experts top-2, aux 0.1     1312           352    0.0770  1.02   [0.1, 0.12, 0.16, 0.15, 0.12, 0.13, 0.1, 0.11]

most-used top-1 expert per regime (share of tokens), aux 0.01:
  regime 0: expert 3 (100%)
  regime 1: expert 7 (67%)
  regime 2: expert 2 (72%)
  regime 3: expert 5 (100%)
```

The finite-difference check shows the router gradient is right, including the top-k renormalisation and the auxiliary term. Both dense MLPs struggle with the four regimes: one network has to switch functions of `x2` on the signs of `x0` and `x1`. Quadrupling the dense width only moves test MSE from 0.206 to 0.158. The MoE gets 0.055 with about the same total parameter count as the wide dense model (1312) and 352 parameters active per token, because the router splits the quadrants and each expert fits one simpler function. Regimes 0 and 3 go 100% to a single top-1 expert.

Without the auxiliary loss, experts 0 and 7 receive no tokens (load 0.0), and the aux value is 1.35. Those parameters are wasted, and at scale their GPUs would sit idle. A coefficient of 0.01 (the Switch Transformer default) narrows the gap (smallest load 0.02, aux 1.25). A coefficient of 0.1 gives an almost uniform 0.10 to 0.16 load per expert, but test MSE rises from 0.055 to 0.077, because the router is now pushed to send tokens to experts that are not the best for them.

## Choosing / trade-offs
- **E and k.** More experts means more capacity at the same per-token compute, but more memory, more communication and more data needed. k = 1 (Switch) is cheapest. k = 2 (GShard, Mixtral) is more stable and lets two experts share a token.
- **Auxiliary loss strength.** Too low and experts collapse. Too high and routing ignores quality. Typical values are 0.001 to 0.01. Some recent models (DeepSeek-V3) instead use a per-expert bias on the router logits, adjusted to balance load without a loss term.
- **Router z-loss.** Penalising large router logits (ST-MoE) keeps the router softmax numerically stable in low precision.
- **Fine-grained and shared experts.** Many small experts plus one or two always-on shared experts (DeepSeekMoE) improve specialisation over a few large ones.
- **Serving.** MoE trades FLOPs for memory and all-to-all communication. Expert parallelism places experts on different GPUs. For single-GPU inference the memory footprint, not the FLOPs, sets the hardware.

## Gotchas
- Loop-free dispatch matters at scale. This example loops over experts and gathers tokens with `nonzero`. Real kernels sort tokens by expert and use grouped GEMMs (MegaBlocks, for example).
- `f_e` uses argmax counts, so it carries no gradient. Only `P_e` does. Getting this backwards silently disables the balancing.
- Renormalising the top-k gates changes the gradient (see `dp_sel` in `backward`). Without renormalisation, the output scale depends on how confident the router is.
- Balance on the training batch is not balance at inference. Batch size, sequence packing and domain shift all change the routing distribution, so monitor per-expert load in production.
- Dead experts rarely recover on their own. Watch per-expert token counts from the first steps.
- Fine-tuning an MoE is brittle: experts overfit small datasets, and changing the router can break pretrained specialisation. Freezing the router or using LoRA on the experts is common.

## Related
- [[mixture-of-experts]] - MoE LLMs in practice: architectures, memory needs and serving.
- [[transformer-block-from-scratch-numpy]] - the dense feed-forward block an MoE layer replaces.
- [[neural-network-from-scratch-numpy]] - the backprop through linear and ReLU layers each expert uses.
- [[optimizers-from-scratch-numpy]] - the Adam update used in the training loop.
- [[gmm-em-from-scratch]] - the classic, unsupervised mixture model that "mixture of experts" borrowed its gating idea from.

## References
- Shazeer et al. (2017), "Outrageously large neural networks: the sparsely-gated mixture-of-experts layer": https://arxiv.org/abs/1701.06538
- Fedus, Zoph and Shazeer (2021), "Switch Transformers: scaling to trillion parameter models with simple and efficient sparsity": https://arxiv.org/abs/2101.03961
- Jiang et al. (2024), "Mixtral of experts": https://arxiv.org/abs/2401.04088
