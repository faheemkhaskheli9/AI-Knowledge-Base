---
title: Recurrent neural networks (LSTM, GRU) in practice
category: concepts
tags: [rnn, lstm, gru, bptt, truncated-bptt, gradient-clipping, packed-sequences, sequence-modeling, pytorch]
use_cases:
  - "train an LSTM/GRU on variable-length sequences without padding corrupting the result"
  - "classify or forecast from sensor, log or event sequences with a small model"
  - "process a long or endless stream step by step with constant memory"
  - "fix exploding or vanishing gradients in a recurrent model"
  - "decide between a GRU, a 1D CNN and a transformer for a sequence task"
status: draft
last_verified: 2026-10-04
sources:
  - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
  - https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
  - https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
  - https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
---

# Recurrent neural networks (LSTM, GRU) in practice

## Summary
A recurrent neural network reads a sequence one step at a time and carries a hidden state forward, so the same small set of weights handles any length. Plain RNNs forget after a few dozen steps because gradients vanish or explode through the repeated multiplication; LSTM and GRU cells add gates that keep information and gradients alive over hundreds of steps. [[cnn-and-rnn-architectures]] places RNNs among other architectures; this file is the practical detail of training them: shapes, padding and packing, backpropagation through time, clipping and stateful streaming.

## Key concepts
- **Recurrence.** hₜ = f(W·xₜ + U·hₜ₋₁ + b). The output at step t depends on all earlier inputs through hₜ.
- **Backpropagation through time (BPTT).** Unroll the loop over T steps and backpropagate through the unrolled graph. Memory grows with T; gradients get multiplied by U (and the activation's derivative) T times, so they shrink to 0 or blow up ([[backpropagation-and-autograd]]).
- **LSTM.** A separate cell state cₜ updated additively: forget gate (what to drop), input gate (what to write), output gate (what to expose as hₜ). The additive path is what lets gradients survive. Returns `(output, (h_n, c_n))`.
- **GRU.** Two gates (update, reset) and no separate cell state. About 25% fewer parameters than an LSTM of the same width, and often as accurate. Returns `(output, h_n)`.
- **Shapes (PyTorch).** Input `(batch, time, features)` with `batch_first=True`. `output` is the top layer's hidden state at **every** step, `(batch, time, hidden × directions)`. `h_n` is the **last** step of every layer, `(layers × directions, batch, hidden)`. Use `output` for per-step tasks (tagging, seq2seq) and `h_n[-1]` for a whole-sequence label.
- **Packing.** `pack_padded_sequence` tells the RNN each sequence's real length, so it stops at the last real step and `h_n` is not computed over padding.
- **Truncated BPTT.** For long streams, run in chunks of k steps, pass the state to the next chunk, and `detach()` it so gradients only flow k steps back. Memory stays constant; dependencies longer than k are learned only weakly.
- **Bidirectional.** A second RNN reads the sequence backwards and outputs are concatenated. Better for offline labeling (NER, full-recording classification); impossible for streaming or forecasting, because it needs the future.

## When to use / scenarios
- Small on-device or streaming models: keyword spotting, wearable/IMU activity recognition, predictive maintenance on vibration windows, ECG beats. A GRU with 32-128 units runs one step at a time with constant memory.
- Event sequences with a running state: user session modeling, clickstream next-action, log anomaly detection.
- Time-series forecasting with covariates when data is moderate (DeepAR-style models are LSTMs) ([[time-series-forecasting]]).
- Online speech and handwriting recognition with CTC ([[sequence-to-sequence-and-ctc]]).
- NOT the default for text or large datasets: transformers train in parallel and handle long context better ([[transformers-and-attention]]). For long sequences with linear cost, consider state-space models ([[state-space-models]]). For fixed windows, a 1D CNN is often faster and just as accurate.

## Setup & code
```bash
pip install "torch>=2.2"
```
A GRU classifier on variable-length sequences with packing and gradient clipping, a check that padding does not change the result, and truncated BPTT on a long stream:
```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

torch.manual_seed(0)

# Task: label = sum of the sequence is positive; variable lengths 5..50.
def make_batch(n):
    lengths = torch.randint(5, 51, (n,))
    seqs = [torch.randn(int(L), 1) for L in lengths]
    y = torch.tensor([float(s.sum() > 0) for s in seqs])
    return pad_sequence(seqs, batch_first=True), lengths, y


class GRUClassifier(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.rnn = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)          # h_n: (num_layers, batch, hidden), last *real* step
        return self.head(h_n[-1]).squeeze(-1)


model = GRUClassifier()
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.BCEWithLogitsLoss()
for step in range(1, 401):
    x, lengths, y = make_batch(64)
    loss = loss_fn(model(x, lengths), y)
    opt.zero_grad()
    loss.backward()
    gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    if step % 100 == 0:
        print(f"step {step}: loss={loss.item():.3f} grad_norm={gnorm:.2f}")

model.eval()
with torch.no_grad():
    x, lengths, y = make_batch(1000)
    acc = ((model(x, lengths) > 0).float() == y).float().mean()
print(f"test accuracy: {acc:.3f}")

# Padding check: same sequence, extra zero padding, must give the same logit when packed
with torch.no_grad():
    s = torch.randn(1, 10, 1)
    padded = torch.cat([s, torch.zeros(1, 40, 1)], dim=1)
    a = model(s, torch.tensor([10]))
    b = model(padded, torch.tensor([10]))
    rnn_out, _ = model.rnn(padded)                        # unpacked: runs over padding
    c = model.head(rnn_out[:, -1]).squeeze(-1)
print(f"packed short={a.item():.4f} packed padded={b.item():.4f} unpacked last-step={c.item():.4f}")

# Truncated BPTT on one long stream: carry state, detach every chunk
lstm = nn.LSTM(1, 16, batch_first=True)
stream = torch.randn(1, 1000, 1)
state = None
for chunk in stream.split(100, dim=1):
    out, state = lstm(chunk, state)
    state = tuple(s.detach() for s in state)              # stop gradients flowing past the chunk
print(f"TBPTT chunks: {len(stream.split(100, dim=1))}, final h shape: {tuple(state[0].shape)}")
```
Output (PyTorch 2.13.0, CPU):
```
step 100: loss=0.134 grad_norm=0.37
step 200: loss=0.188 grad_norm=4.66
step 300: loss=0.096 grad_norm=1.76
step 400: loss=0.036 grad_norm=1.21
test accuracy: 0.976
packed short=3.2285 packed padded=3.2285 unpacked last-step=6.5177
TBPTT chunks: 10, final h shape: (1, 1, 16)
```
With packing, adding 40 padding steps leaves the logit unchanged (3.2285). Reading the last step of an unpacked run gives 6.5177: the GRU kept updating its state over the zeros. The grad-norm spike to 4.66 at step 200 is the kind of batch that clipping caps at 1.0.

In a real stream loop, also call `loss.backward()` and `opt.step()` inside each chunk; the example only shows the state handling.

## Choosing / trade-offs
- **GRU vs LSTM.** Start with a GRU (fewer parameters, faster). Try an LSTM if the task needs very long memory or the GRU underfits. Differences are usually small; tune width and depth first.
- **Hidden size and depth.** 1-2 layers of 64-256 units cover most tabular-sequence and sensor tasks. Past 3 layers, returns drop and training gets harder; add residual connections if you go deeper.
- **RNN vs 1D CNN vs transformer.** RNN: streaming, constant memory per step, small data. 1D CNN/TCN: fixed windows, fastest to train (parallel over time). Transformer: large data, long context, needs more memory. See [[cnn-and-rnn-architectures]].
- **Truncation length k.** Longer k learns longer dependencies but costs memory and time linearly. Set it a bit longer than the longest dependency you expect.
- **cuDNN.** On GPU, `nn.LSTM`/`nn.GRU` use fused cuDNN kernels. Writing your own cell loop with `nn.LSTMCell` is far slower; use it only when you need per-step custom logic.

## Gotchas
- Without packing, `h_n` (or `output[:, -1]`) for short sequences is computed over padding, as the check above shows. Pack, or gather `output` at index `length - 1`.
- `h_n` is ordered `(layers × directions, batch, hidden)` even with `batch_first=True`. For a bidirectional model, concatenate `h_n[-2]` and `h_n[-1]` (last layer's forward and backward state), not `h_n[-1]` alone.
- `pack_padded_sequence` needs lengths on the CPU (a CUDA tensor raises an error) and sorted lengths unless `enforce_sorted=False`.
- Forgetting to `detach()` the carried state makes the graph grow across the whole stream: memory climbs until OOM, or `backward` fails with "Trying to backward through the graph a second time".
- Reset the state between independent sequences (new user, new file). Carrying state across unrelated samples leaks information and hurts accuracy.
- Exploding gradients show up as sudden NaN loss: clip by global norm (~1.0) every step ([[debugging-neural-network-training]]). The `dropout` argument of `nn.LSTM`/`nn.GRU` applies only between stacked layers, not inside the recurrence.
- Scale the inputs. Unnormalized features saturate the tanh/sigmoid gates and the network stops learning.

## Related
- [[cnn-and-rnn-architectures]] - where RNNs fit next to CNNs and transformers.
- [[sequence-to-sequence-and-ctc]] - encoder-decoder and CTC built on RNNs.
- [[backpropagation-and-autograd]] - how BPTT computes gradients.
- [[deep-learning-training]] - training loop, clipping and schedules.
- [[state-space-models]] - modern linear-time recurrent alternatives (Mamba).
- [[time-series-forecasting]] - LSTM forecasting next to classical and boosted models.
- [[transformers-and-attention]] - the default replacement for RNNs on text.
- [[gru-from-scratch-numpy]] - GRU built in NumPy with BPTT, checked against PyTorch.

## References
- PyTorch `nn.LSTM`: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- PyTorch `nn.GRU`: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
- PyTorch `pack_padded_sequence`: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
- PyTorch `clip_grad_norm_`: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- Hochreiter & Schmidhuber, "Long Short-Term Memory", Neural Computation 1997.
- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (GRU), 2014: https://arxiv.org/abs/1406.1078
- Pascanu, Mikolov & Bengio, "On the difficulty of training recurrent neural networks", ICML 2013: https://arxiv.org/abs/1211.5063
