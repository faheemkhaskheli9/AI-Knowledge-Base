---
title: Residual and skip connections (ResNet, U-Net, DenseNet, pre-norm)
category: concepts
tags: [deep-learning, residual-connection, skip-connection, resnet, bottleneck, u-net, densenet, highway, pre-norm, degradation-problem, stochastic-depth, pytorch]
use_cases:
  - "train a network deeper than ~20 layers without accuracy getting worse"
  - "build an encoder-decoder for segmentation or denoising that keeps fine detail"
  - "understand why every transformer block is x + f(x)"
  - "add a shortcut when input and output shapes differ"
  - "regularise a very deep network with stochastic depth"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1512.03385
  - https://arxiv.org/abs/1603.05027
  - https://arxiv.org/abs/1505.04597
  - https://arxiv.org/abs/1608.06993
  - https://arxiv.org/abs/1603.09382
  - https://pytorch.org/vision/stable/models/resnet.html
---

# Residual and skip connections (ResNet, U-Net, DenseNet, pre-norm)

## Summary
A skip connection routes a layer's input around it and combines it with the output. In a **residual** connection the combination is addition, `y = x + F(x)`, so each block learns only a correction to its input and the identity is the easy default. This gives gradients a direct path back to early layers and made 100+ layer CNNs (ResNet) and deep transformers trainable. **Concatenating** skips (U-Net, DenseNet) instead pass earlier feature maps forward unchanged, which preserves spatial detail and encourages feature reuse.

## Key concepts
- **Degradation problem.** Before ResNet, a plain 56-layer CNN had *higher training error* than a 20-layer one. That is an optimisation problem, not overfitting: stacked layers struggle to even learn the identity. With `x + F(x)`, a block that is not useful just drives `F` toward 0.
- **Gradient highway.** `d(x + F(x))/dx = I + dF/dx`. The identity term passes gradient through unchanged, so it does not vanish across many blocks ([[backpropagation-and-autograd]]).
- **Basic vs bottleneck block.** Basic: two 3x3 convs (ResNet-18/34). Bottleneck: 1x1 reduce, 3x3, 1x1 expand (ResNet-50+), cheaper at high channel counts.
- **Projection shortcut.** When stride or channel count changes, the shortcut becomes a 1x1 conv (+ norm) so shapes match; otherwise it is the identity.
- **Placement of norm and activation.**
  - Post-activation (original ResNet): `relu(x + BN(conv(...)))`.
  - Pre-activation (ResNet v2): `x + conv(relu(BN(x)))`, a cleaner identity path, better for very deep nets.
  - Transformers: pre-norm `x + f(LN(x))` is the stable default ([[normalization-layers]]).
- **Concatenation skips.** U-Net copies encoder feature maps to the decoder at the same resolution (`cat` along channels). DenseNet concatenates every earlier layer's output inside a block.
- **Residual init tricks.** Zero-init the last norm's gamma (or last layer's weights) in each residual branch so every block starts as the identity; scale residual branches by `1/sqrt(n_layers)` in deep transformers ([[weight-initialization]]).
- **Stochastic depth.** Randomly drop whole residual branches during training (keep the identity). A strong regulariser for deep ResNets and ViTs (`drop_path` in timm).

## When to use / scenarios
- Any CNN, MLP or transformer deeper than a handful of layers: use residual blocks by default.
- Segmentation, denoising, super-resolution, diffusion U-Nets: encoder-decoder with concatenating skips at every resolution ([[segmentation]]).
- Deep tabular MLPs: residual MLP blocks with LayerNorm are a strong baseline ([[deep-learning-for-tabular-data]]).
- Fine-tuning adapters (LoRA, bottleneck adapters) are themselves residual: the pretrained path is the identity and the adapter learns a small delta ([[fine-tuning-and-peft]]).
- NOT needed for shallow networks (2-4 layers), where a skip adds nothing but parameters for the projection.

## Setup & code
```bash
pip install torch
```
A residual block with a projection shortcut, and a demonstration that gradients reach the first layer of a deep stack only with the skip:
```python
import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out))
        nn.init.zeros_(self.f[-1].weight)              # block starts as identity
        self.short = (nn.Identity() if stride == 1 and c_in == c_out else
                      nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)))
    def forward(self, x):
        return torch.relu(self.short(x) + self.f(x))

print(ResBlock(16, 32, stride=2)(torch.randn(2, 16, 32, 32)).shape)   # (2, 32, 16, 16)

class Plain(nn.Module):
    def __init__(self, d, residual):
        super().__init__()
        self.lin, self.residual = nn.Linear(d, d), residual
    def forward(self, x):
        h = torch.tanh(self.lin(x))
        return x + h if self.residual else h

torch.manual_seed(0)
for residual in (False, True):
    net = nn.Sequential(*[Plain(64, residual) for _ in range(50)])
    net(torch.randn(32, 64)).pow(2).mean().backward()
    print(f"residual={residual}: first-layer grad norm {net[0].lin.weight.grad.norm():.2e}")
```
U-Net style concatenation skip (one level):
```python
class TinyUNet(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, c, 3, padding=1), nn.ReLU())
        self.down = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c, 2 * c, 3, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec = nn.Conv2d(2 * c, 1, 3, padding=1)      # 2c = upsampled c + skipped c
    def forward(self, x):
        e = self.enc(x)
        return self.dec(torch.cat([self.up(self.down(e)), e], dim=1))

print(TinyUNet()(torch.randn(1, 3, 64, 64)).shape)       # (1, 1, 64, 64)
```
Pretrained ResNets: `torchvision.models.resnet50(weights="DEFAULT")`.

## Choosing / trade-offs
- **Add vs concatenate.** Addition keeps channel count fixed and memory flat; concatenation keeps raw detail and lets later layers choose features, at the cost of growing channels and activation memory (DenseNet is memory-hungry).
- **Basic vs bottleneck.** Basic blocks for small or shallow models; bottlenecks for depth 50+ where 3x3 convs on wide channels get expensive.
- **Post- vs pre-activation / post- vs pre-norm.** Pre-variants train more stably at large depth; post-variants are needed only to match pretrained weights.
- **Depth vs width.** Residuals make depth trainable, but past a point wider or higher-resolution models give more per FLOP; see EfficientNet-style compound scaling in [[image-classification]].

## Gotchas
- Shape mismatch at the add: when stride or channels change, the shortcut needs a projection; forgetting it raises an error, while a mismatched but broadcastable shape silently does the wrong thing.
- Applying ReLU to the shortcut path, or norm after the addition in pre-norm designs, blocks the clean identity path that makes residuals work.
- Residual streams grow in variance with depth; without branch scaling or zero-init, very deep transformers can diverge early ([[debugging-neural-network-training]]).
- In-place ops (`relu(inplace=True)`) on a tensor that is also the shortcut input corrupt the saved value for backward.
- U-Net skips need matching spatial sizes; odd input sizes after pooling require padding or cropping before `cat`.
- Stochastic depth must be off at eval (`model.eval()`), and its drop rate usually increases linearly with block depth.

## Related
- [[cnn-and-rnn-architectures]] - the CNN families that use residual blocks.
- [[transformers-and-attention]] - the residual stream in every transformer block.
- [[normalization-layers]] - pre-norm vs post-norm placement around the skip.
- [[weight-initialization]] - zero-init and branch scaling for residual nets.
- [[backpropagation-and-autograd]] - why the identity path preserves gradients.
- [[regularization-in-deep-learning]] - stochastic depth alongside dropout and weight decay.

## References
- He et al., Deep Residual Learning for Image Recognition: https://arxiv.org/abs/1512.03385
- He et al., Identity Mappings in Deep Residual Networks (pre-activation): https://arxiv.org/abs/1603.05027
- Ronneberger et al., U-Net: https://arxiv.org/abs/1505.04597
- Huang et al., Densely Connected Convolutional Networks: https://arxiv.org/abs/1608.06993
- Huang et al., Deep Networks with Stochastic Depth: https://arxiv.org/abs/1603.09382
- torchvision ResNet models: https://pytorch.org/vision/stable/models/resnet.html
