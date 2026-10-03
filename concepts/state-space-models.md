---
title: State space models (S4, Mamba) and linear-time sequence models
category: concepts
tags: [state-space-models, ssm, mamba, s4, linear-attention, rwkv, hybrid-models, long-sequences, sequence-modeling]
use_cases:
  - "model very long sequences (audio, genomics, logs) where attention's quadratic cost is too high"
  - "understand what Mamba or a hybrid Mamba-transformer model is when choosing an LLM"
  - "pick a sequence architecture with constant memory per generated token"
  - "decide between a transformer, an LSTM and an SSM for a time-series or signal task"
status: draft
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/2111.00396
  - https://arxiv.org/abs/2312.00752
  - https://arxiv.org/abs/2405.21060
  - https://github.com/state-spaces/mamba
  - https://huggingface.co/docs/transformers/en/model_doc/mamba
---

# State space models (S4, Mamba) and linear-time sequence models

## Summary
State space models (SSMs) process a sequence through a fixed-size hidden state
updated by a linear recurrence, like an RNN, but are designed so training can
run in parallel over the whole sequence. Cost grows linearly with length, and
generating each token needs constant memory, unlike attention's quadratic cost
and growing KV cache. S4 made them work on long-range tasks; Mamba added
input-dependent ("selective") parameters, which made them competitive with
transformers for language. Most current production use is in hybrids that mix
SSM layers with a few attention layers.

## Key concepts
- Continuous SSM: `h'(t) = A h(t) + B x(t)`, `y(t) = C h(t)`. Discretize with
  a step size `dt` to get `h_t = A_bar h_{t-1} + B_bar x_t`.
- Two equivalent views when A, B, C are fixed: a recurrence (cheap inference,
  O(1) per step) and a long convolution with kernel `K_k = C A_bar^k B_bar`
  (parallel training via FFT). The example below checks they match.
- Structured A: S4 used a special (HiPPO-initialized, diagonal plus low-rank)
  matrix to remember long histories stably; later work showed diagonal A is
  enough.
- Selectivity (Mamba): `dt`, `B` and `C` are computed from the current input.
  The model can choose to reset or keep its state per token, which fixed SSMs
  cannot. The convolution view no longer exists; training uses a
  hardware-aware parallel scan.
- Mamba-2 / SSD: reformulates selective SSMs as a structured form of linear
  attention, allowing faster matrix-multiply kernels.
- Relatives in the same "linear-time" family: linear attention, RetNet, RWKV,
  gated DeltaNet, xLSTM. Same trade-off: fixed-size state instead of a KV cache.
- Weakness: a fixed-size state compresses the past, so exact recall of an
  arbitrary earlier token (copying, needle-in-a-haystack, in-context lookup)
  is worse than attention. Hybrids add a few attention layers to restore it.

## When to use / scenarios
- Very long signals: raw audio, genomics (DNA sequence models), high-rate
  sensor streams, long event logs ([[manufacturing-iot]]).
- LLM serving where long outputs or many concurrent sequences make the KV
  cache the bottleneck: hybrid SSM-attention models (Jamba, Zamba, Falcon-H1,
  Nemotron-H and similar) trade a little recall for much cheaper memory.
- Edge/streaming inference that must process one step at a time with
  constant memory ([[edge-on-device]]).
- NOT for: standard-length text, vision or tabular tasks where a pretrained
  transformer exists; tooling, kernels and fine-tuning support are far more
  mature there ([[transformers-and-attention]]). For short time series,
  gradient boosting or a small CNN/LSTM is simpler ([[time-series-forecasting]],
  [[cnn-and-rnn-architectures]]).

## Setup & code
A toy diagonal SSM in plain PyTorch: shows the recurrence and convolution
views agree, and what selectivity buys.
```bash
pip install torch
```
```python
import torch

torch.manual_seed(0)
L, N = 64, 16                         # sequence length, state size
A = -torch.rand(N) * 0.5 - 0.01       # diagonal, negative -> stable decay
B, C = torch.randn(N), torch.randn(N)
dt = 0.1
Ab = torch.exp(dt * A)                # zero-order-hold discretisation (diagonal)
Bb = (Ab - 1) / A * B
x = torch.randn(L)

# 1) Recurrent mode: O(1) state per step - how inference runs
h, y_rec = torch.zeros(N), []
for t in range(L):
    h = Ab * h + Bb * x[t]
    y_rec.append((C * h).sum())
y_rec = torch.stack(y_rec)

# 2) Convolutional mode: kernel K_k = C * Ab^k * Bb - how (non-selective) training runs in parallel
K = torch.stack([(C * Ab ** k * Bb).sum() for k in range(L)])
y_conv = torch.stack([(K[: t + 1].flip(0) * x[: t + 1]).sum() for t in range(L)])
print("recurrent == convolution:", torch.allclose(y_rec, y_conv, atol=1e-5))

# 3) Selective (Mamba-style): dt and B depend on the input, so no fixed kernel exists;
#    training uses a parallel scan instead. Large dt -> reset to the input, small dt -> keep state.
def selective(x, dts):
    h, out = torch.zeros(N), []
    for t in range(len(x)):
        Ab_t = torch.exp(dts[t] * A)
        h = Ab_t * h + (Ab_t - 1) / A * B * x[t]
        out.append((C * h).sum())
    return torch.stack(out)

x2 = torch.zeros(L); x2[5] = 1.0                 # one impulse at t=5
keep = selective(x2, torch.full((L,), 0.01))     # small dt everywhere: memory persists
forget = selective(x2, torch.where(torch.arange(L) == 30, 50.0, 0.01))  # big dt at t=30 resets
print(f"output at t=40, keep: {keep[40]:.4f}  forget after t=30: {forget[40]:.4f}")
```
Output (torch 2.13.0 CPU): `recurrent == convolution: True`; at t=40 the
impulse still contributes 0.0494 when dt stays small and 0.0027 after one
large-dt step at t=30, which wiped the state. In Mamba a learned projection of
the input produces that dt, so the model decides per token what to forget.

To use real models: Hugging Face Transformers ships `MambaForCausalLM`,
`Mamba2ForCausalLM` and several hybrid architectures; the fast CUDA kernels
come from `mamba-ssm` and `causal-conv1d` (Linux + NVIDIA GPU). Without them
Transformers falls back to a slow pure-PyTorch path. See
[[huggingface-transformers]].

## Choosing / trade-offs
- Quality on general text at a given size: transformers and hybrids lead;
  pure SSMs trail on recall-heavy tasks.
- Inference memory and throughput on long sequences: SSMs and hybrids win
  (no KV cache growth).
- Ecosystem: transformers have vLLM/llama.cpp support, quantization and PEFT
  everywhere; SSM support exists but check your serving stack per model
  ([[inference-servers-vllm]], [[llama-cpp-gguf]]).
- Small-data sequence tasks: an LSTM/GRU or 1D CNN is easier to train and
  often as good.

## Gotchas
- "Linear-time" does not mean faster at short lengths; attention with
  FlashAttention is very fast below a few thousand tokens.
- Long context window on paper does not mean the model recalls details from
  it; test retrieval within your actual context ([[long-context]]).
- Fast kernels are CUDA-only; on CPU, Apple Silicon or Windows-native you may
  get the slow reference implementation.
- The recurrence can be numerically sensitive; keep A negative (stable) and
  compute in fp32 for the state where implementations recommend it.
- Benchmarks reported for a hybrid say little about a pure SSM, and vice versa.

## Related
- [[transformers-and-attention]] - the architecture SSMs compete with and hybridize with.
- [[cnn-and-rnn-architectures]] - RNN ancestors; same recurrence idea without parallel training.
- [[long-context]] - recall limits and KV cache costs that motivate SSMs.
- [[mixture-of-experts]] - often combined with hybrid SSM models.
- [[time-series-forecasting]] - simpler options for short series.

## References
- Gu et al., Efficiently Modeling Long Sequences with Structured State Spaces (S4): https://arxiv.org/abs/2111.00396
- Gu & Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces: https://arxiv.org/abs/2312.00752
- Dao & Gu, Transformers are SSMs (Mamba-2 / SSD): https://arxiv.org/abs/2405.21060
- Reference implementation: https://github.com/state-spaces/mamba
- Hugging Face Mamba docs: https://huggingface.co/docs/transformers/en/model_doc/mamba
