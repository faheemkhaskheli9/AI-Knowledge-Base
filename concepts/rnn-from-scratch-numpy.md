---
title: RNN from scratch in NumPy (forward pass, backprop through time, gradient check, truncated BPTT)
category: concepts
tags: [rnn, recurrent-neural-network, bptt, backpropagation-through-time, truncated-bptt, gradient-check, gradient-clipping, char-rnn, numpy, from-scratch, deep-learning-basics]
use_cases:
  - "implement a vanilla RNN and backprop through time from scratch"
  - "verify hand-written RNN gradients with a numerical gradient check"
  - "understand truncated BPTT, hidden-state carry-over and gradient clipping"
  - "train a tiny character-level language model in pure NumPy"
status: stable
last_verified: 2026-10-04
sources:
  - https://karpathy.github.io/2015/05/21/rnn-effectiveness/
  - https://gist.github.com/karpathy/d4dee566867f8291f086
  - https://www.deeplearningbook.org/contents/rnn.html
  - https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html
---

# RNN from scratch in NumPy (forward pass, backprop through time, gradient check, truncated BPTT)

## Summary
A vanilla RNN applies the same cell at every time step: `hₜ = tanh(W_xh xₜ + W_hh hₜ₋₁ + b_h)`, then `yₜ = W_hy hₜ + b_y`. Backpropagation through time (BPTT) is ordinary backprop on this loop unrolled over the sequence: walk the steps in reverse and add each step's gradient into the shared weights. About 40 lines of NumPy give a character-level language model whose gradients agree with finite differences to `4e-8`, and which learns a short repeated text to near-zero loss.

## Key concepts
- **Weight sharing.** The same `W_xh`, `W_hh` and `W_hy` act at every step, so each weight's gradient is the **sum** of its per-step gradients.
- **Two paths into hₜ.** In the backward pass the hidden state gets gradient from its own output (`W_hyᵀ dyₜ`) and from the next step (`dh_next`). Forgetting the second path turns the RNN into a stack of independent per-step classifiers.
- **Through tanh.** `d(tanh z)/dz = 1 − tanh²(z)`, so `draw = (1 − hₜ²) ⊙ dh`. Then `dh_next = W_hhᵀ draw` carries it one step back.
- **Softmax + cross-entropy gradient.** For a one-hot target, `∂L/∂z = p − onehot(y)`. The same identity as in an MLP ([[neural-network-from-scratch-numpy]], [[loss-functions]]).
- **One-hot input without a matmul.** `W_xh @ onehot(x)` is just column `W_xh[:, x]`, and its gradient is added back into that column only.
- **Truncated BPTT.** Backprop through only the last `T` steps (16 below). Carry the hidden state forward between chunks so the model still sees long context, but stop gradients at the chunk boundary.
- **Exploding and vanishing gradients.** `dh_next` is multiplied by `W_hhᵀ diag(1 − h²)` once per step. Over many steps that product shrinks or blows up exponentially. Clipping fixes the explosion; LSTMs and GRUs fix the vanishing ([[vanishing-and-exploding-gradients]], [[recurrent-neural-networks-lstm-gru]]).

## When to use / scenarios
- Learning: the step from "backprop through layers" to "backprop through time", and why gating was invented.
- Interviews: writing the BPTT loop, explaining weight sharing, truncated BPTT and gradient clipping.
- Debugging a custom recurrent cell: a gradient check like the one below catches a missing `dh_next` or a wrong tanh derivative.
- Production: use `torch.nn.RNN`/`LSTM`/`GRU` (cuDNN-fused) or, for most sequence tasks today, a transformer or state space model ([[transformers-and-attention]], [[state-space-models]]).
- Not for: long-range dependencies (hundreds of steps). A vanilla RNN forgets; use an LSTM/GRU or attention.

## Setup & code
`pip install numpy`. Runs on CPU in about 10 seconds.

```python
import numpy as np

rng = np.random.default_rng(0)
text = "hello world, hello rnn. " * 40
chars = sorted(set(text))
V, H, T = len(chars), 32, 16
ix = {c: i for i, c in enumerate(chars)}
data = np.array([ix[c] for c in text])

P = {"Wxh": rng.normal(0, 0.1, (H, V)), "Whh": rng.normal(0, 0.1, (H, H)),
     "Why": rng.normal(0, 0.1, (V, H)), "bh": np.zeros(H), "by": np.zeros(V)}


def forward_backward(P, xs, ys, h0):
    hs, ps, loss = {-1: h0}, {}, 0.0
    for t, x in enumerate(xs):                        # forward through time
        hs[t] = np.tanh(P["Wxh"][:, x] + P["Whh"] @ hs[t - 1] + P["bh"])
        z = P["Why"] @ hs[t] + P["by"]
        ps[t] = np.exp(z - z.max()); ps[t] /= ps[t].sum()
        loss -= np.log(ps[t][ys[t]])
    g = {k: np.zeros_like(v) for k, v in P.items()}
    dh_next = np.zeros(H)
    for t in reversed(range(len(xs))):                # backprop through time
        dy = ps[t].copy(); dy[ys[t]] -= 1             # softmax + CE gradient
        g["Why"] += np.outer(dy, hs[t]); g["by"] += dy
        dh = P["Why"].T @ dy + dh_next                # from output and from step t+1
        draw = (1 - hs[t] ** 2) * dh                  # through tanh
        g["Wxh"][:, xs[t]] += draw; g["bh"] += draw
        g["Whh"] += np.outer(draw, hs[t - 1])
        dh_next = P["Whh"].T @ draw
    return loss, g, hs[len(xs) - 1]


# Gradient check: analytic BPTT vs central differences on 5 entries of Whh.
xs, ys = data[:T], data[1:T + 1]
_, g, _ = forward_backward(P, xs, ys, np.zeros(H))
errs = []
for i, j in rng.integers(H, size=(5, 2)):
    P["Whh"][i, j] += 1e-5; lp = forward_backward(P, xs, ys, np.zeros(H))[0]
    P["Whh"][i, j] -= 2e-5; lm = forward_backward(P, xs, ys, np.zeros(H))[0]
    P["Whh"][i, j] += 1e-5
    num = (lp - lm) / 2e-5
    errs.append(abs(num - g["Whh"][i, j]) / max(1e-8, abs(num) + abs(g["Whh"][i, j])))
print(f"max relative grad error: {max(errs):.1e}")

# Truncated BPTT training with Adagrad and gradient clipping.
mem = {k: np.zeros_like(v) for k, v in P.items()}
h, pos, lr = np.zeros(H), 0, 0.1
for step in range(3001):
    if pos + T + 1 > len(data):
        h, pos = np.zeros(H), 0                       # reset state at end of text
    xs, ys = data[pos:pos + T], data[pos + 1:pos + T + 1]
    loss, g, h = forward_backward(P, xs, ys, h)       # carry h, but not gradients
    for k in P:
        np.clip(g[k], -5, 5, out=g[k])                # exploding-gradient guard
        mem[k] += g[k] ** 2
        P[k] -= lr * g[k] / np.sqrt(mem[k] + 1e-8)
    pos += T
    if step % 1000 == 0:
        print(f"step {step}: loss/char={loss / T:.3f}")


def sample(P, seed_char, n):
    h, x, out = np.zeros(H), ix[seed_char], [seed_char]
    for _ in range(n):
        h = np.tanh(P["Wxh"][:, x] + P["Whh"] @ h + P["bh"])
        x = int((P["Why"] @ h + P["by"]).argmax())    # greedy decoding
        out.append(chars[x])
    return "".join(out)


print("sample:", repr(sample(P, "h", 40)))
```

Output (numpy 2.5):
```
max relative grad error: 3.7e-08
step 0: loss/char=2.413
step 1000: loss/char=0.001
step 2000: loss/char=0.001
step 3000: loss/char=0.000
sample: 'hello world, hello rnn. hello world, hell'
```

The analytic BPTT gradients match central differences to a relative error of `4e-8`, which means the backward pass is right. The starting loss of 2.41 is close to `ln(11) = 2.40`, uniform guessing over the 11 distinct characters. The model then memorises the repeated phrase almost perfectly and greedy sampling reproduces it. On real text the same code needs more hidden units and data, and its loss plateaus well above zero.

## Choosing / trade-offs
- **Truncation length T.** Longer chunks let gradients reach further back but cost memory linearly and suffer more vanishing. Typical values are 32-256. Carrying `h` across chunks gives longer *forward* context for free.
- **Clipping: element-wise vs global norm.** The code clips each element to `[-5, 5]` (as in Karpathy's min-char-rnn). Frameworks usually clip the global norm (`torch.nn.utils.clip_grad_norm_`), which keeps the gradient's direction.
- **Optimiser.** Adagrad works for this toy. Adam is the usual default for RNNs ([[optimizers]]).
- **Vanilla RNN vs LSTM/GRU vs attention.** Vanilla RNNs are for learning and tiny problems. LSTM/GRU handle longer dependencies with gates. Transformers parallelise over time and dominate most sequence tasks; linear RNNs/SSMs return for very long sequences ([[recurrent-neural-networks-lstm-gru]], [[state-space-models]]).

## Gotchas
- Gradients must **accumulate** across time steps (`+=`) because the weights are shared. Assigning (`=`) keeps only the last step's gradient.
- Snapshot `hₜ` per step. If you overwrite one `h` variable in the forward loop, the backward pass uses the wrong activations and the gradient check fails.
- Do not backprop across chunk boundaries by accident. In PyTorch, call `h = h.detach()` between chunks; without it the graph grows each step until memory runs out.
- Reset or carry the hidden state on purpose. Carrying it across unrelated sequences (different documents in a batch) leaks context between them.
- Run the gradient check in float64 with `eps ≈ 1e-5`. In float32 the finite differences are dominated by rounding and look "wrong" when they are not ([[debugging-neural-network-training]]).
- Near-zero loss on a toy repeated string only shows the code learns. Always evaluate a language model on held-out text.
- Greedy decoding loops on repeated phrases. For generation, sample from the softmax with a temperature ([[decoding-and-sampling]]).

## Related
- [[neural-network-from-scratch-numpy]] - the feed-forward MLP version of the same backprop recipe.
- [[recurrent-neural-networks-lstm-gru]] - gated RNNs and their PyTorch usage.
- [[vanishing-and-exploding-gradients]] - why gradients through long chains shrink or blow up.
- [[backpropagation-and-autograd]] - backprop as reverse-mode autodiff on any graph, including unrolled loops.
- [[convolution-layer-from-scratch-numpy]] - the other weight-sharing layer written from scratch.
- [[decoding-and-sampling]] - greedy, temperature and top-p sampling from a language model.

## References
- Karpathy, The Unreasonable Effectiveness of Recurrent Neural Networks: https://karpathy.github.io/2015/05/21/rnn-effectiveness/
- Karpathy, min-char-rnn.py: https://gist.github.com/karpathy/d4dee566867f8291f086
- Goodfellow, Bengio and Courville, Deep Learning, ch. 10 (Sequence Modeling): https://www.deeplearningbook.org/contents/rnn.html
- PyTorch `torch.nn.RNN`: https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html
