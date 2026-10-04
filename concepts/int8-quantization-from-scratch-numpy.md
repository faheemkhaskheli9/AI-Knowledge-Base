---
title: INT8 quantization from scratch in NumPy (absmax, zero-point, per-channel, outliers)
category: concepts
tags: [quantization, int8, absmax, zero-point, per-channel, llm-int8, outliers, inference, numpy, from-scratch]
use_cases:
  - "see exactly what int8 weight and activation quantization does to a matrix multiply"
  - "choose between per-tensor and per-channel scales, or symmetric and zero-point quantization"
  - "understand why activation outliers break naive int8 LLM inference and how LLM.int8() handles them"
  - "explain quantization error and int32 accumulation in an interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1712.05877
  - https://arxiv.org/abs/2208.07339
  - https://pytorch.org/docs/stable/quantization.html
---

# INT8 quantization from scratch in NumPy (absmax, zero-point, per-channel, outliers)

## Summary
Quantization stores a float tensor as small integers plus a scale: `q = round(t / s)`, `t ≈ q · s`. With int8 weights, a matrix takes a quarter of the fp32 memory, and integer matrix multiplies run on fast int8 hardware paths. This file builds symmetric (absmax) and asymmetric (zero-point) int8 quantization in a few lines of NumPy and measures the error. On a 256 × 512 weight matrix with a few large rows, a per-tensor scale gives 6.6% relative error and per-channel scales give 0.75%. A full int8 matmul with int32 accumulation is within 1% of fp32, until one activation feature is 50× larger than the rest, as happens in large LLMs. Then the error jumps to 6.6%. Keeping that one column in floating point, the LLM.int8() idea, brings it back to 0.6%.

## Key concepts
- **Scale and rounding.** Symmetric absmax uses `s = max|t| / 127` and `q = clip(round(t / s), -127, 127)`. The rounding error per element is at most `s / 2`, so a single large value raises the error for every other element in the tensor.
- **Zero-point (asymmetric).** `q = round(t / s) + z` with `s = (max − min) / 255`. It uses all 256 levels when the data is one-sided, such as ReLU outputs, where symmetric quantization wastes the negative half.
- **Granularity.** One scale per tensor, per output channel (row of `W`), per token (row of `x`), or per group of e.g. 128 values. Finer granularity costs a few extra floats and contains the damage from outliers.
- **Integer matmul.** `y = x Wᵀ ≈ (q_x q_Wᵀ) · s_x · s_Wᵀ`. The int8 × int8 products are accumulated in int32, so they do not overflow, and dequantized once at the end. This only works when scales factor out of the dot product, which is why activations get per-row scales and weights per-column ones.
- **Outlier features.** Large transformers develop a few hidden dimensions with values far larger than the rest. With a per-token scale, those outliers set the scale for the whole token and the other features lose almost all their precision.

## When to use / scenarios
- Learning: the arithmetic under every `load_in_8bit`, GGUF `Q8_0`, TensorRT or ONNX Runtime int8 setting.
- Practice: post-training quantization (PTQ) of weights to cut memory and bandwidth, the main cost of LLM decoding. Int8 activations as well for CPU or edge inference ([[edge-on-device]]).
- Interviews: "per-tensor vs per-channel", "why int32 accumulators", "why LLM.int8() keeps some columns in fp16".
- Not for: going below 8 bits. 4-bit and lower need group-wise scales and error-compensating methods such as GPTQ or AWQ, covered in [[quantization]]. Not for training either, see [[efficient-training-mixed-precision]].

## Setup & code
NumPy only. Runs in under a second.

```python
import numpy as np

rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, (256, 512)).astype(np.float32)   # out x in weight matrix
W[:4] *= 20                                               # a few rows with large weights
x = rng.normal(0, 1, (64, 512)).astype(np.float32)        # activations
x[:, 7] *= 50                                             # one outlier feature, as in LLMs


def quant_sym(t, axis=None):
    """Symmetric absmax int8: q = round(t / s), s = max|t| / 127."""
    s = np.abs(t).max(axis=axis, keepdims=axis is not None) / 127
    return np.clip(np.round(t / s), -127, 127).astype(np.int8), s


def quant_asym(t):
    """Asymmetric (zero-point) uint8: covers [min, max] instead of [-max, max]."""
    lo, hi = t.min(), t.max()
    s = (hi - lo) / 255
    z = np.round(-lo / s)
    return np.clip(np.round(t / s) + z, 0, 255).astype(np.uint8), s, z


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


qt, s = quant_sym(W)
qc, sc = quant_sym(W, axis=1)
print(f"weight error  per-tensor {rel_err(qt * s, W):.4f}   per-channel {rel_err(qc * sc, W):.4f}")

r = np.maximum(rng.normal(0.5, 1, 10_000), 0).astype(np.float32)   # ReLU output, all >= 0
qs, ss = quant_sym(r)
qa, sa, za = quant_asym(r)
print(f"ReLU acts     symmetric  {rel_err(qs * ss, r):.4f}   zero-point  {rel_err((qa.astype(np.float32) - za) * sa, r):.4f}")

y = x @ W.T
qx, sx = quant_sym(x, axis=1)                                         # per-token activation scale
acc = qx.astype(np.int32) @ qc.astype(np.int32).T                    # int8 x int8 -> int32 accumulate
y_q = acc * sx * sc.T                                                 # dequantize once at the end
print(f"int8 matmul   rel error {rel_err(y_q, y):.4f}  (with one outlier feature)")
xo = x.copy(); xo[:, 7] /= 50
qxo, sxo = quant_sym(xo, axis=1)
y_o = (qxo.astype(np.int32) @ qc.astype(np.int32).T) * sxo * sc.T
print(f"int8 matmul   rel error {rel_err(y_o, xo @ W.T):.4f}  (outlier removed)")
keep = np.arange(512) != 7                                            # LLM.int8()-style split
qk, sk = quant_sym(x[:, keep], axis=1)
y_mix = (qk.astype(np.int32) @ qc[:, keep].astype(np.int32).T) * sk * sc.T + x[:, ~keep] @ W[:, ~keep].T
print(f"mixed         rel error {rel_err(y_mix, y):.4f}  (outlier column kept in fp32)")
print(f"memory        fp32 {W.nbytes} B  int8 {qc.nbytes + sc.nbytes} B")
```

Output (Python 3.14, NumPy 2.5):
```
weight error  per-tensor 0.0657   per-channel 0.0075
ReLU acts     symmetric  0.0078   zero-point  0.0039
int8 matmul   rel error 0.0664  (with one outlier feature)
int8 matmul   rel error 0.0101  (outlier removed)
mixed         rel error 0.0059  (outlier column kept in fp32)
memory        fp32 524288 B  int8 132096 B
```

Four large rows were enough to make the per-tensor scale 20× too coarse for the other 252 rows. Per-channel scales fix that for 256 extra floats. Zero-point quantization halves the error on non-negative data because it spends all 256 levels on `[0, max]`. In the matmul, one outlier feature sets each token's scale, and the remaining 511 features are rounded on a grid 50× too coarse. Splitting that column out and computing it in float is the mixed-precision decomposition from LLM.int8(). Memory drops about 4× (the extra 1 KB is the scales).

## Choosing / trade-offs
- **Weights: per-channel symmetric.** It is the standard choice, and nearly free. Group-wise scales (one per 32-128 values) are the next step and are what 4-bit formats use.
- **Activations: dynamic per-token vs static.** Dynamic scales (computed at runtime, as here) adapt to each input. Static scales come from a calibration set, are cheaper, and fail on inputs unlike the calibration data.
- **Outliers.** Options are mixed precision (LLM.int8()), moving the scale from activations into weights (SmoothQuant), or quantizing weights only (W8A16, W4A16) and keeping activations in fp16. Weight-only is the common choice for LLM serving, since decoding is limited by memory bandwidth, not compute.
- **Symmetric vs zero-point.** Symmetric keeps the integer matmul simple. Zero-point adds correction terms to the matmul but fits one-sided data better. Most kernels use symmetric weights and allow asymmetric activations.
- **PTQ vs QAT.** Post-training quantization needs no training and is enough at 8 bits. Quantization-aware training simulates rounding during fine-tuning and matters most at 4 bits and below.
- **Libraries.** `bitsandbytes` (`load_in_8bit`), `torch.ao.quantization`, ONNX Runtime and TensorRT for int8. llama.cpp GGUF and vLLM for LLM weight formats.

## Gotchas
- Report relative error on the layer output or, better, the end-task metric. Small per-weight error can still add up across dozens of layers.
- Use `-127..127`, not `-128..127`, for symmetric int8. The extra level breaks the symmetry and buys almost nothing.
- Accumulate in int32. An int8 or int16 accumulator overflows within a few hundred products.
- An all-zero row gives `s = 0` and a division by zero. Guard with a small epsilon or skip the row.
- The speedup depends on hardware kernels. Quantizing in NumPy and dequantizing before a float matmul saves memory but not compute.
- Calibration data must match production inputs. A static activation scale tuned on short English prompts can clip badly on code or long contexts.
- Fused operations (layer norm, softmax, residual adds) usually stay in higher precision. Quantizing them costs more accuracy than it saves.

## Related
- [[quantization]] - formats, GPTQ/AWQ, GGUF and which to choose for LLMs.
- [[kv-cache-from-scratch-numpy]] - the other large memory cost at inference, which can also be quantized.
- [[efficient-training-mixed-precision]] - fp16/bf16 training, the training-time counterpart.
- [[knowledge-distillation-from-scratch-numpy]] - the other main way to shrink a model.
- [[edge-on-device]] - where int8 activation quantization pays off most.

## References
- Jacob et al. (2018), "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference": https://arxiv.org/abs/1712.05877
- Dettmers et al. (2022), "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale": https://arxiv.org/abs/2208.07339
- PyTorch quantization docs: https://pytorch.org/docs/stable/quantization.html
