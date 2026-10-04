---
title: LSTM from scratch in NumPy (gates, cell state, backprop through time, checked against PyTorch)
category: concepts
tags: [lstm, long-short-term-memory, gates, cell-state, forget-gate, bptt, backpropagation-through-time, vanishing-gradients, numpy, pytorch, from-scratch, deep-learning-basics]
use_cases:
  - "implement an LSTM cell forward and backward pass from scratch"
  - "check a hand-written LSTM against torch.nn.LSTM"
  - "understand why the cell state lets gradients reach far-back time steps"
status: draft
last_verified: 2026-10-04
sources:
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
  - https://colah.github.io/posts/2015-08-Understanding-LSTMs/
  - https://www.bioinf.jku.at/publications/older/2604.pdf
  - https://proceedings.mlr.press/v37/jozefowicz15.html
---

# LSTM from scratch in NumPy (gates, cell state, backprop through time, checked against PyTorch)

## Summary
An LSTM is an RNN whose hidden state is protected by gates. Each step computes four gate vectors from `[xₜ, hₜ₋₁]`: input `i`, forget `f`, candidate `g` and output `o`. It then updates a separate **cell state** additively, `cₜ = f ⊙ cₜ₋₁ + i ⊙ g`, and outputs `hₜ = o ⊙ tanh(cₜ)`. Because the cell's gradient flows back through a multiplication by `f` instead of through a squashing non-linearity and a weight matrix, it can survive many steps. The NumPy forward and backward passes below match `torch.nn.LSTM` and its autograd gradients to `1e-16`, and show the gradient reaching 60 steps back growing from `1e-15` to `2e-2` with a positive forget-gate bias.

## Key concepts
- **Four gates, one matmul.** Stack the four gate weight matrices into `W` of shape `(4H, D)` and `U` of shape `(4H, H)`; `a = Wxₜ + Uhₜ₋₁ + b`, then slice. PyTorch's order is `i, f, g, o`.
- **Sigmoid gates, tanh candidate.** `i, f, o = σ(·)` are in `(0, 1)` and act as soft switches; `g = tanh(·)` is the proposed new content.
- **Cell state.** `cₜ = f ⊙ cₜ₋₁ + i ⊙ g`. Forget part of the old memory, write part of the new.
- **Output.** `hₜ = o ⊙ tanh(cₜ)`. `h` is what the next layer and the next step see; `c` stays internal.
- **The gradient highway.** `∂cₜ/∂cₜ₋₁ = f` (element-wise). With `f ≈ 1` the gradient passes almost unchanged, unlike a vanilla RNN where each step multiplies by `W_hhᵀ diag(1 − h²)` ([[rnn-from-scratch-numpy]], [[vanishing-and-exploding-gradients]]).
- **Backward step.** With `dh` (from the output plus the next step) and `dc_next`: `dc = dc_next + dh ⊙ o ⊙ (1 − tanh²c)`, then each gate's pre-activation gradient is its upstream gradient times its own derivative. Finally `dh_next = Uᵀda` and `dc_next = dc ⊙ f`.
- **Forget-gate bias.** Initialising `b_f` to 1 or more keeps `f` near 1 at the start of training, so long-range gradients exist from step one (Jozefowicz et al., 2015).

## When to use / scenarios
- Learning: the step from a vanilla RNN to gated recurrence, and a good exercise in hand-written backprop.
- Interviews: "write the LSTM equations", "why does the LSTM fix vanishing gradients", "LSTM vs GRU".
- Debugging a custom recurrent cell: compare its forward and gradients to a PyTorch reference as below.
- Production: `torch.nn.LSTM` (cuDNN-fused) for streaming or low-latency sequence models, small on-device models, and time series ([[recurrent-neural-networks-lstm-gru]], [[time-series-forecasting]]). For most text tasks use a transformer ([[transformers-and-attention]]).
- Not for: very long contexts where parallel training matters; transformers and state space models train in parallel over time ([[state-space-models]]).

## Setup & code
`pip install numpy torch`. Runs on CPU in a few seconds.

```python
import numpy as np
import torch


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def lstm_forward(x, W, U, b, h0, c0):
    """x: (T, D). W: (4H, D), U: (4H, H), b: (4H,). Gate order i, f, g, o (PyTorch's)."""
    H = h0.shape[0]
    h, c, cache = h0, c0, []
    hs = []
    for t in range(len(x)):
        a = W @ x[t] + U @ h + b
        i, f, o = sigmoid(a[:H]), sigmoid(a[H:2 * H]), sigmoid(a[3 * H:])
        g = np.tanh(a[2 * H:3 * H])
        c_prev, h_prev = c, h
        c = f * c_prev + i * g                       # additive cell update
        h = o * np.tanh(c)
        cache.append((x[t], h_prev, c_prev, i, f, g, o, c))
        hs.append(h)
    return np.array(hs), cache


def lstm_backward(dhs, W, U, cache):
    """dhs: (T, H) gradient of the loss wrt every h_t. Returns dW, dU, db, dx."""
    H = U.shape[1]
    dW, dU, db = np.zeros_like(W), np.zeros_like(U), np.zeros(4 * H)
    dx = np.zeros((len(cache), W.shape[1]))
    dh_next, dc_next = np.zeros(H), np.zeros(H)
    for t in reversed(range(len(cache))):
        x, h_prev, c_prev, i, f, g, o, c = cache[t]
        dh = dhs[t] + dh_next
        tc = np.tanh(c)
        dc = dc_next + dh * o * (1 - tc ** 2)
        da = np.concatenate([dc * g * i * (1 - i),      # input gate
                             dc * c_prev * f * (1 - f), # forget gate
                             dc * i * (1 - g ** 2),     # candidate
                             dh * tc * o * (1 - o)])    # output gate
        dW += np.outer(da, x); dU += np.outer(da, h_prev); db += da
        dx[t] = W.T @ da
        dh_next = U.T @ da
        dc_next = dc * f                               # the cell's gradient highway
    return dW, dU, db, dx


T, D, H = 12, 5, 8
torch.manual_seed(0)
ref = torch.nn.LSTM(D, H).double()
W = ref.weight_ih_l0.detach().numpy().copy()
U = ref.weight_hh_l0.detach().numpy().copy()
b = (ref.bias_ih_l0 + ref.bias_hh_l0).detach().numpy()   # PyTorch keeps two biases; only the sum matters

rng = np.random.default_rng(0)
x = rng.normal(size=(T, D))
hs, cache = lstm_forward(x, W, U, b, np.zeros(H), np.zeros(H))

xt = torch.tensor(x[:, None, :], requires_grad=True)     # (T, batch=1, D)
out, _ = ref(xt)
print(f"forward max |ours - torch| = {np.abs(hs - out[:, 0].detach().numpy()).max():.1e}")

# loss = sum of all outputs times fixed random weights
R = rng.normal(size=(T, H))
(out[:, 0] * torch.tensor(R)).sum().backward()
dW, dU, db, dx = lstm_backward(R, W, U, cache)
for name, ours, theirs in [("dW", dW, ref.weight_ih_l0.grad), ("dU", dU, ref.weight_hh_l0.grad),
                           ("db", db, ref.bias_ih_l0.grad), ("dx", dx, xt.grad[:, 0])]:
    print(f"{name} max |ours - torch| = {np.abs(ours - theirs.numpy()).max():.1e}")

# Why the cell helps: gradient reaching x_0 from a loss on the last step only.
dlast = np.zeros((60, H)); dlast[-1] = 1.0
xl = rng.normal(size=(60, D))
for fb in (0.0, 3.0):
    b2 = b.copy(); b2[H:2 * H] += fb                    # forget-gate bias init trick
    _, cache = lstm_forward(xl, W, U, b2, np.zeros(H), np.zeros(H))
    gx = lstm_backward(dlast, W, U, cache)[3]
    print(f"forget bias +{fb}: |dL/dx_0| = {np.linalg.norm(gx[0]):.1e}, "
          f"|dL/dx_59| = {np.linalg.norm(gx[-1]):.1e}")
```

Output (numpy 2.5, torch 2.13):
```
forward max |ours - torch| = 1.1e-16
dW max |ours - torch| = 5.6e-16
dU max |ours - torch| = 1.1e-16
db max |ours - torch| = 4.4e-16
dx max |ours - torch| = 9.7e-17
forget bias +0.0: |dL/dx_0| = 1.3e-15, |dL/dx_59| = 2.9e-01
forget bias +3.0: |dL/dx_0| = 2.0e-02, |dL/dx_59| = 2.6e-01
```

The outputs and all four gradients equal PyTorch's to float64 rounding, so the equations, the gate order and the backward pass are right. A float64 reference is a stronger check than finite differences: it compares every entry exactly. With PyTorch's default initialisation (forget bias ≈ 0, so `f ≈ 0.5`), the gradient from a loss at step 59 shrinks to `1e-15` by step 0: halving 60 times. Adding 3 to the forget-gate bias (`f ≈ 0.95`) keeps it at `2e-2`, almost 13 orders of magnitude larger, while the gradient at the last step barely changes.

## Choosing / trade-offs
- **LSTM vs GRU.** A GRU merges the cell and hidden state and uses two gates instead of three plus a candidate, so it has 3/4 of the parameters and is a little faster. Accuracy is usually similar; try both ([[recurrent-neural-networks-lstm-gru]]).
- **LSTM vs transformer.** An LSTM has constant memory per step and natural streaming, but trains sequentially over time. Transformers train in parallel and handle long-range dependencies better; their per-token cost grows with context length ([[transformers-and-attention]]).
- **Hidden size and layers.** Parameters per layer are `4H(D + H + 1)` (plus a second bias in PyTorch). Two stacked layers with dropout between them is a common default.
- **Bidirectional.** Run a second LSTM backwards and concatenate. Only valid when the whole sequence is available at prediction time (tagging, classification), not for streaming or generation.

## Gotchas
- Gate order differs between implementations: PyTorch (`i, f, g, o`) and Keras (`i, f, c, o`) agree, but ONNX's LSTM operator uses `i, o, f, c`. Copying weights between them silently scrambles the gates unless you reorder the slices.
- PyTorch has two bias vectors (`bias_ih`, `bias_hh`); only their sum matters. When setting the forget-gate bias, change one of them, not both, or you add the offset twice.
- `torch.nn.LSTM` expects `(T, batch, D)` unless `batch_first=True`. A wrong layout runs without error and learns nonsense.
- Cache `cₜ₋₁` and `hₜ₋₁` for every step during the forward pass. Overwriting them gives a backward pass that is subtly wrong.
- Gradient clipping is still needed: the cell path does not vanish easily, but it can still explode ([[vanishing-and-exploding-gradients]]).
- Pad variable-length batches and use `pack_padded_sequence`, or the final hidden state comes from padding tokens.
- For truncated BPTT in PyTorch, detach both `h` and `c` between chunks, not just `h`.

## Related
- [[rnn-from-scratch-numpy]] - the vanilla RNN and BPTT this builds on.
- [[recurrent-neural-networks-lstm-gru]] - LSTM and GRU usage in PyTorch.
- [[vanishing-and-exploding-gradients]] - the problem the cell state solves.
- [[autograd-engine-from-scratch]] - the general machinery that derives these gradients automatically.
- [[transformer-block-from-scratch-numpy]] - the architecture that replaced LSTMs for most sequence tasks.
- [[state-space-models]] - linear recurrences that train in parallel.

## References
- PyTorch `torch.nn.LSTM` (equations and gate order): https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- Olah, Understanding LSTM Networks: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- Hochreiter and Schmidhuber, Long Short-Term Memory (1997): https://www.bioinf.jku.at/publications/older/2604.pdf
- Jozefowicz, Zaremba and Sutskever, An Empirical Exploration of Recurrent Network Architectures (ICML 2015): https://proceedings.mlr.press/v37/jozefowicz15.html
