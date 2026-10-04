---
title: GRU from scratch in NumPy (reset and update gates, backprop through time, checked against PyTorch)
category: concepts
tags: [gru, gated-recurrent-unit, reset-gate, update-gate, bptt, backpropagation-through-time, vanishing-gradients, rnn, numpy, pytorch, from-scratch, deep-learning-basics]
use_cases:
  - "implement a GRU cell forward and backward pass from scratch"
  - "check a hand-written GRU against torch.nn.GRU"
  - "understand how the update gate lets gradients skip many time steps"
  - "explain GRU vs LSTM in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
  - https://arxiv.org/abs/1406.1078
  - https://arxiv.org/abs/1412.3555
---

# GRU from scratch in NumPy (reset and update gates, backprop through time, checked against PyTorch)

## Summary
A GRU (gated recurrent unit) is a gated RNN with no separate cell state. Each step computes a reset gate `r`, an update gate `z` and a candidate `n`, then blends: `hₜ = (1 − z) ⊙ n + z ⊙ hₜ₋₁`. When `z ≈ 1` the old state is copied forward almost unchanged, which gives gradients a direct path back through time, the same job the LSTM's cell state does with fewer parameters (3 weight blocks instead of 4). The NumPy forward and backward passes below match `torch.nn.GRU` and its autograd gradients to `1e-15`.

## Key concepts
- **Equations (PyTorch's form).** `r = σ(W_ir x + b_ir + W_hr h + b_hr)`, `z = σ(W_iz x + b_iz + W_hz h + b_hz)`, `n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))`, `h' = (1 − z) ⊙ n + z ⊙ h`.
- **Stacked weights.** `W_ih` is `(3H, D)` and `W_hh` is `(3H, H)`, gate order `r, z, n`. Compute `Wx·x + bx` and `Wh·h + bh` once per step and slice.
- **Reset gate.** It scales only the hidden part of the candidate, `r ⊙ (W_hn h + b_hn)`, so `r ≈ 0` makes the candidate ignore the past. The two biases are not interchangeable here: `b_hn` sits inside the `r ⊙ (...)`, which is why PyTorch keeps both.
- **Update gate.** `z` interpolates between keeping the old state and writing the candidate. `∂hₜ/∂hₜ₋₁` contains the direct term `diag(z)`, the gradient highway.
- **Backward step.** With `dh = dLₜ/dhₜ + dh_next`: `dn = dh ⊙ (1 − z) ⊙ (1 − n²)`, `dz = dh ⊙ (hₜ₋₁ − n) ⊙ z(1 − z)`, `dr = dn ⊙ (W_hn h + b_hn) ⊙ r(1 − r)`. The hidden-side gradient for the `n` block is `dn ⊙ r`. Then `dh_next = dh ⊙ z + W_hhᵀ · dah`.
- **Parameter count.** `3H(D + H) + 6H` in PyTorch (two bias vectors), against `4H(D + H) + 8H` for an LSTM: exactly 3/4.

## When to use / scenarios
- Learning: the second gated cell after the LSTM, with a shorter backward pass that is easy to derive by hand ([[lstm-from-scratch-numpy]]).
- Interviews: "write the GRU equations", "GRU vs LSTM", "why does gating help with vanishing gradients".
- Debugging a custom recurrent cell: check forward and gradients against a float64 PyTorch reference, as below.
- Production: `torch.nn.GRU` for small streaming or on-device sequence models and time series when an LSTM is too heavy ([[recurrent-neural-networks-lstm-gru]], [[time-series-forecasting]]).
- Not for: long text contexts where parallel training matters; use a transformer or a state space model ([[transformers-and-attention]], [[state-space-models]]).

## Setup & code
`pip install numpy torch`. Runs on CPU in a few seconds.

```python
import numpy as np
import torch


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gru_forward(x, Wx, Wh, bx, bh, h0):
    """x: (T, D). Wx: (3H, D), Wh: (3H, H), bx, bh: (3H,). Gate order r, z, n (PyTorch's)."""
    H = h0.shape[0]
    h, hs, cache = h0, [], []
    for t in range(len(x)):
        ax, ah = Wx @ x[t] + bx, Wh @ h + bh
        r = sigmoid(ax[:H] + ah[:H])
        z = sigmoid(ax[H:2 * H] + ah[H:2 * H])
        n = np.tanh(ax[2 * H:] + r * ah[2 * H:])        # reset gate scales the hidden part only
        h_prev = h
        h = (1 - z) * n + z * h_prev                     # z = 1 keeps the old state
        cache.append((x[t], h_prev, r, z, n, ah[2 * H:]))
        hs.append(h)
    return np.array(hs), cache


def gru_backward(dhs, Wx, Wh, cache):
    """dhs: (T, H) gradient of the loss wrt every h_t. Returns dWx, dWh, dbx, dbh, dx."""
    H = Wh.shape[1]
    dWx, dWh = np.zeros_like(Wx), np.zeros_like(Wh)
    dbx, dbh = np.zeros(3 * H), np.zeros(3 * H)
    dx = np.zeros((len(cache), Wx.shape[1]))
    dh_next = np.zeros(H)
    for t in reversed(range(len(cache))):
        x, h_prev, r, z, n, ahn = cache[t]
        dh = dhs[t] + dh_next
        dn = dh * (1 - z) * (1 - n ** 2)                 # through tanh
        dz = dh * (h_prev - n) * z * (1 - z)
        dr = dn * ahn * r * (1 - r)
        dax = np.concatenate([dr, dz, dn])
        dah = np.concatenate([dr, dz, dn * r])           # r multiplies W_hn h + b_hn
        dWx += np.outer(dax, x); dbx += dax
        dWh += np.outer(dah, h_prev); dbh += dah
        dx[t] = Wx.T @ dax
        dh_next = dh * z + Wh.T @ dah                    # direct path through z, plus the gates
    return dWx, dWh, dbx, dbh, dx


T, D, H = 12, 5, 8
torch.manual_seed(0)
ref = torch.nn.GRU(D, H).double()
Wx, Wh = (p.detach().numpy().copy() for p in (ref.weight_ih_l0, ref.weight_hh_l0))
bx, bh = (p.detach().numpy().copy() for p in (ref.bias_ih_l0, ref.bias_hh_l0))

rng = np.random.default_rng(0)
x = rng.normal(size=(T, D))
hs, cache = gru_forward(x, Wx, Wh, bx, bh, np.zeros(H))

xt = torch.tensor(x[:, None, :], requires_grad=True)     # (T, batch=1, D)
out, _ = ref(xt)
print(f"forward max |ours - torch| = {np.abs(hs - out[:, 0].detach().numpy()).max():.1e}")

R = rng.normal(size=(T, H))                              # loss = sum(out * R)
(out[:, 0] * torch.tensor(R)).sum().backward()
grads = gru_backward(R, Wx, Wh, cache)
refs = (ref.weight_ih_l0.grad, ref.weight_hh_l0.grad, ref.bias_ih_l0.grad, ref.bias_hh_l0.grad, xt.grad[:, 0])
for name, ours, theirs in zip(("dWx", "dWh", "dbx", "dbh", "dx"), grads, refs):
    print(f"{name} max |ours - torch| = {np.abs(ours - theirs.numpy()).max():.1e}")

# The update gate as a gradient highway: loss on the last of 60 steps only.
dlast = np.zeros((60, H)); dlast[-1] = 1.0
xl = rng.normal(size=(60, D))
for zb in (0.0, 3.0):
    bx2 = bx.copy(); bx2[H:2 * H] += zb                  # push z toward 1 = "keep the old state"
    _, cache = gru_forward(xl, Wx, Wh, bx2, bh, np.zeros(H))
    gx = gru_backward(dlast, Wx, Wh, cache)[4]
    print(f"update bias +{zb}: |dL/dx_0| = {np.linalg.norm(gx[0]):.1e}, "
          f"|dL/dx_59| = {np.linalg.norm(gx[-1]):.1e}")

print("params GRU vs LSTM (D=5, H=8):", sum(p.numel() for p in ref.parameters()),
      sum(p.numel() for p in torch.nn.LSTM(D, H).parameters()))
```

Output (numpy 2.5, torch 2.13):
```
forward max |ours - torch| = 3.1e-16
dWx max |ours - torch| = 8.9e-16
dWh max |ours - torch| = 3.3e-16
dbx max |ours - torch| = 8.9e-16
dbh max |ours - torch| = 4.4e-16
dx max |ours - torch| = 2.2e-16
update bias +0.0: |dL/dx_0| = 4.8e-13, |dL/dx_59| = 4.3e-01
update bias +3.0: |dL/dx_0| = 5.0e-03, |dL/dx_59| = 3.9e-02
params GRU vs LSTM (D=5, H=8): 360 480
```

The outputs and all five gradients match PyTorch's to float64 rounding, so the gate order, the placement of `r` and the backward pass are right. With PyTorch's default initialisation, the gradient from a loss at step 59 has shrunk to `5e-13` by step 0. Adding 3 to the update-gate bias (`z ≈ 0.95`) raises it to `5e-3`, ten orders of magnitude more. The cost shows in the last line: the gradient into the current input drops from 0.43 to 0.04, because a GRU that mostly keeps its old state also mostly ignores new input. The LSTM separates those two decisions (forget and input gates); the GRU ties them with `z` and `1 − z`.

## Choosing / trade-offs
- **GRU vs LSTM.** The GRU has 3/4 of the parameters and one state vector, so it is a bit faster and lighter. Accuracy is usually similar; Chung et al. (2014) found neither consistently better. Try both if the recurrent model matters.
- **GRU vs vanilla RNN.** Always prefer a gated cell for sequences longer than a few dozen steps ([[rnn-from-scratch-numpy]], [[vanishing-and-exploding-gradients]]).
- **GRU vs transformer.** A GRU has constant memory per step and streams naturally, but trains sequentially. Transformers train in parallel and model long-range dependencies better.
- **Update-gate bias.** A positive `z` bias at initialisation helps long-memory tasks and hurts tasks that must react to every input. Leave the default unless gradients clearly vanish.

## Gotchas
- Two GRU conventions exist. PyTorch and cuDNN apply `r` after the hidden matmul (`r ⊙ (W_hn h + b_hn)`). The original paper and Keras with `reset_after=False` apply it before (`W_hn (r ⊙ h)`). Weights do not transfer between them unchanged.
- Some texts write `h' = z ⊙ n + (1 − z) ⊙ h`, swapping what `z` means. Check which convention a reference uses before copying equations or a bias trick.
- PyTorch's gate order is `r, z, n`. Keras orders the blocks `z, r, h`. Slice accordingly when porting weights.
- `torch.nn.GRU` expects `(T, batch, D)` unless `batch_first=True`.
- Cache `hₜ₋₁` and `W_hn h + b_hn` for every step; the reset gate's gradient needs the latter.
- Clip gradients: gating slows vanishing but does not stop explosion.

## Related
- [[lstm-from-scratch-numpy]] - the four-gate cell with a separate cell state, built and checked the same way.
- [[rnn-from-scratch-numpy]] - the vanilla RNN and BPTT this builds on.
- [[recurrent-neural-networks-lstm-gru]] - GRU and LSTM usage in PyTorch.
- [[vanishing-and-exploding-gradients]] - the problem the update gate addresses.
- [[autograd-engine-from-scratch]] - the machinery that derives these gradients automatically.
- [[state-space-models]] - linear recurrences that train in parallel.

## References
- PyTorch `torch.nn.GRU` (equations and gate order): https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html
- Cho et al. (2014), "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation": https://arxiv.org/abs/1406.1078
- Chung, Gulcehre, Cho, Bengio (2014), "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling": https://arxiv.org/abs/1412.3555
