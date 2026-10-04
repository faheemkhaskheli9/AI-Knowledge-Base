---
title: Convolution arithmetic, pooling and receptive field
category: concepts
tags: [deep-learning, cnn, convolution, stride, padding, dilation, pooling, receptive-field, depthwise-separable, transposed-convolution, pytorch]
use_cases:
  - "work out the output shape of a conv or pooling layer before a shape error"
  - "check whether a CNN's receptive field covers the objects or time span it must see"
  - "cut parameters and compute with depthwise-separable convolutions for edge devices"
  - "upsample feature maps in a segmentation decoder"
  - "make a CNN accept images of any size with global average pooling"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1603.07285
  - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
  - https://distill.pub/2019/computing-receptive-fields/
  - https://arxiv.org/abs/1704.04861
  - https://arxiv.org/abs/1511.07122
---

# Convolution arithmetic, pooling and receptive field

## Summary
Every convolution and pooling layer is defined by kernel size, stride, padding and dilation, and these four numbers fix the output size, how far each output "sees" into the input (receptive field), and the compute cost. Knowing the arithmetic lets you design a CNN that downsamples where intended, sees enough context for the task, fits a compute budget, and avoids shape errors. This file is the numeric companion to [[cnn-and-rnn-architectures]].

## Key concepts
- **Output size** (per spatial dim, PyTorch `Conv2d`/`MaxPool2d`): `out = floor((n + 2p − d·(k−1) − 1) / s) + 1` for input `n`, kernel `k`, stride `s`, padding `p`, dilation `d`.
  - "Same" size at stride 1: `p = d·(k−1)/2` (k=3 → p=1). Halve with stride 2: k=3, s=2, p=1.
- **Parameters** of `Conv2d(C_in, C_out, k)`: `C_out · C_in · k² (+ C_out bias)`, independent of image size. **Multiply-adds** ≈ params × H_out × W_out.
- **Receptive field (RF).** The input region that affects one output unit. For a stack of layers, track `r` (RF) and `j` (jump, the product of strides so far): `r += (k−1)·d·j; j *= s`. Three 3x3 convs (stride 1) see 7x7, with fewer parameters than one 7x7.
- **Effective RF** is smaller than the theoretical one: central pixels dominate. Plan for the object to fit well inside the theoretical RF.
- **Dilation** spaces kernel taps apart, growing RF without extra parameters or downsampling (segmentation, audio WaveNet-style models).
- **Pooling.** Max/avg pooling downsample with no parameters. **Global average pooling** (`AdaptiveAvgPool2d(1)`) collapses any H×W into one vector per channel, so the network accepts arbitrary input sizes.
- **Depthwise-separable conv.** A per-channel spatial conv (`groups=C_in`) followed by a 1x1 conv that mixes channels. Roughly `1/C_out + 1/k²` of the standard conv's cost; the core of MobileNet and EfficientNet.
- **1x1 conv** mixes channels at each pixel: cheap channel reduction/expansion (bottlenecks).
- **Transposed conv** upsamples: `out = (n−1)·s − 2p + d·(k−1) + output_padding + 1`. k=4, s=2, p=1 doubles the size exactly.

## When to use / scenarios
- Designing a small CNN for a custom input (spectrograms, satellite tiles, sensor images): use the formulas to place stride-2 layers so the final map is ~7x7 and the RF covers the object scale.
- Defect detection where defects are tiny: avoid early aggressive downsampling (e.g. the ResNet stem) or use higher input resolution.
- Long audio or time series with a 1D CNN: dilated convs (dilation 1, 2, 4, 8, ...) give an RF that grows exponentially with depth ([[time-series-forecasting]]).
- Phone or microcontroller deployment: depthwise-separable convs and fewer channels ([[edge-on-device]]).
- Segmentation decoders: transposed conv or `Upsample` + conv to restore resolution ([[segmentation]]).
- NOT needed when using a pretrained backbone as-is; then only input size and output stride matter.

## Setup & code
```bash
pip install torch
```
Shapes, receptive field and cost, checked against PyTorch (torch 2.13 CPU):
```python
import torch
from torch import nn

def out_size(n, k, s=1, p=0, d=1):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1

x = torch.randn(1, 3, 224, 224)
for k, s, p, d in [(3, 1, 1, 1), (3, 2, 1, 1), (7, 2, 3, 1), (3, 1, 2, 2), (2, 2, 0, 1)]:
    y = nn.Conv2d(3, 8, k, stride=s, padding=p, dilation=d)(x)
    assert y.shape[-1] == out_size(224, k, s, p, d)
    print(f"k={k} s={s} p={p} d={d}: 224 -> {y.shape[-1]}")
# 224, 112, 112, 224, 112

def receptive_field(layers):               # layers: [(kernel, stride, dilation), ...]
    r, jump = 1, 1
    for k, s, d in layers:
        r += (k - 1) * d * jump
        jump *= s
    return r, jump
print(receptive_field([(3, 1, 1)] * 3))              # (7, 1)
print(receptive_field([(7, 2, 1), (3, 2, 1)]))       # (11, 4)  ResNet stem: 7x7/2 conv + 3x3/2 maxpool

# Standard vs depthwise-separable 3x3, 64 -> 128 channels, on a 56x56 map.
std = nn.Conv2d(64, 128, 3, padding=1, bias=False)
dws = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
                    nn.Conv2d(64, 128, 1, bias=False))
n_params = lambda m: sum(p.numel() for p in m.parameters())
print(n_params(std), n_params(std) * 56 * 56)   # 73728 params, 231,211,008 MACs
print(n_params(dws), n_params(dws) * 56 * 56)   # 8768 params,   27,496,448 MACs (~8.4x less)

feats = torch.randn(2, 128, 13, 17)
print(nn.MaxPool2d(2)(feats).shape)                                    # [2, 128, 6, 8]  (floor)
print(nn.AdaptiveAvgPool2d(1)(feats).flatten(1).shape)                 # [2, 128]  any H, W
print(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)(feats).shape) # [2, 64, 26, 34]  exact 2x
```
For a full model, `torchinfo.summary(model, input_size=(1, 3, 224, 224))` (`pip install torchinfo`) prints per-layer output shapes and parameter counts.

## Choosing / trade-offs
- **Stride-2 conv vs pooling** for downsampling: strided convs learn the downsampling (most modern CNNs); max pooling is parameter-free and adds a little translation robustness.
- **Large kernel vs stacked 3x3.** Stacked 3x3s give the same RF with fewer parameters and more non-linearities; recent designs (ConvNeXt) use 7x7 depthwise convs, where large kernels are cheap.
- **Dilation vs downsampling** to grow RF: dilation keeps resolution (good for dense prediction) but can cause gridding artifacts; downsampling is cheaper.
- **Depthwise-separable vs standard.** Far fewer FLOPs and parameters, but lower arithmetic intensity: on GPUs the speedup is often much smaller than the FLOP ratio. Measure latency on the target device.
- **Transposed conv vs upsample + conv.** Transposed conv is learnable but prone to checkerboard artifacts when the kernel is not divisible by the stride; bilinear/nearest upsample followed by a 3x3 conv avoids them.

## Gotchas
- Odd input sizes with stride 2 lose a row/column (`floor`): 13 → 6 in the example. Encoder-decoder skip connections then mismatch by one pixel; pad inputs to a multiple of the total stride (32 for most backbones) or crop/interpolate before concatenating.
- `padding="same"` in PyTorch is only allowed with stride 1.
- A `Linear` layer after `flatten` hard-codes the input resolution; use global average pooling to accept other sizes.
- Parameter count is not compute: an early layer on a large map can have few parameters but dominate FLOPs and memory.
- PyTorch layouts are `(N, C, H, W)`; images loaded with PIL/NumPy/OpenCV are `(H, W, C)` and need `permute(2, 0, 1)`.
- Theoretical RF covering the whole image does not mean the model uses global context; effective RF is much smaller. Add attention or global pooling for global reasoning.

## Related
- [[cnn-and-rnn-architectures]] - CNN families and building blocks.
- [[normalization-layers]] - conv → BatchNorm → activation block structure.
- [[segmentation]] - encoder-decoder upsampling and output stride.
- [[edge-on-device]] - depthwise-separable models for edge devices.
- [[transfer-learning-and-domain-adaptation]] - reusing pretrained backbones instead of designing layers.

## References
- Dumoulin & Visin, A guide to convolution arithmetic for deep learning: https://arxiv.org/abs/1603.07285
- PyTorch `nn.Conv2d` (shape formula): https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- Araujo et al., Computing Receptive Fields of Convolutional Neural Networks (Distill): https://distill.pub/2019/computing-receptive-fields/
- Howard et al., MobileNets (depthwise-separable convolutions): https://arxiv.org/abs/1704.04861
- Yu & Koltun, Multi-Scale Context Aggregation by Dilated Convolutions: https://arxiv.org/abs/1511.07122
