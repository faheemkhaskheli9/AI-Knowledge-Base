---
title: Normalization layers (BatchNorm, LayerNorm, RMSNorm, GroupNorm)
category: concepts
tags: [deep-learning, normalization, batch-norm, layer-norm, rms-norm, group-norm, instance-norm, pre-norm, post-norm, train-eval-mode, pytorch]
use_cases:
  - "choose between BatchNorm, LayerNorm, RMSNorm and GroupNorm for a model"
  - "fix validation accuracy that collapses because BatchNorm statistics are wrong"
  - "train a vision model with batch size 1-4 per GPU"
  - "decide pre-norm vs post-norm placement in a transformer"
  - "fold BatchNorm into convolution weights for faster inference"
status: draft
last_verified: 2026-10-04
sources:
  - https://pytorch.org/docs/stable/nn.html#normalization-layers
  - https://arxiv.org/abs/1502.03167
  - https://arxiv.org/abs/1607.06450
  - https://arxiv.org/abs/1910.07467
  - https://arxiv.org/abs/1803.08494
  - https://arxiv.org/abs/2002.04745
---

# Normalization layers (BatchNorm, LayerNorm, RMSNorm, GroupNorm)

## Summary
Normalization layers re-center and re-scale activations inside the network to roughly zero mean and unit variance, then apply a learned scale (γ) and shift (β). This keeps activations in a healthy range through depth, allows higher learning rates, and makes training less sensitive to initialization. The variants differ only in *which axes* the statistics are computed over. That one choice decides whether the layer depends on the batch (BatchNorm) or works per sample (LayerNorm, RMSNorm, GroupNorm), and so where each one fits.

## Key concepts
- **Common form.** `y = γ · (x − μ) / sqrt(σ² + ε) + β`, with μ and σ² computed over some axes of the activation tensor. γ and β are learned per channel/feature so the layer can undo the normalization if useful.
- **Which axes** (image tensor N×C×H×W, sequence tensor N×T×D):
  - **BatchNorm**: over N, H, W for each channel C. The statistics depend on the other samples in the batch.
  - **LayerNorm**: over the feature dims of each sample (D for each token in a transformer). No batch dependence.
  - **RMSNorm**: like LayerNorm but no mean subtraction and no β: `x / rms(x) · γ`. Cheaper; used in LLaMA, Mistral, Gemma and most current LLMs.
  - **GroupNorm**: splits C into G groups and normalises over each group's channels and H, W per sample. G = 1 is LayerNorm over C×H×W, and G = C is InstanceNorm.
  - **InstanceNorm**: per sample, per channel over H, W. Used for style transfer and some generative models.
- **BatchNorm train vs eval.** In training it uses the current batch statistics and updates running averages (`momentum=0.1`). In eval it uses the running averages. That is why `model.train()` / `model.eval()` matter and why small or non-representative batches break it.
- **Pre-norm vs post-norm (transformers).** Post-norm `LN(x + f(x))` (original Transformer, BERT) can match pre-norm quality but needs warmup and is unstable when deep. Pre-norm `x + f(LN(x))` (GPT-2 onwards, most LLMs) trains stably at depth with less warmup; it usually adds a final LayerNorm before the output head.
- **Weight-side alternatives.** Weight normalization and spectral normalization re-parameterise weights instead of activations. Spectral norm is common in GAN discriminators ([[generative-adversarial-networks]]).

## When to use / scenarios
- CNNs with per-GPU batch ≥ ~16: BatchNorm (the ResNet default).
- Detection/segmentation at high resolution with 1-4 images per GPU: GroupNorm (32 groups), frozen BatchNorm from a pretrained backbone, or SyncBatchNorm across GPUs.
- Transformers, RNNs, any variable-length or batch-size-1 setting: LayerNorm. New LLMs from scratch: RMSNorm with pre-norm.
- Online or streaming inference where batches are size 1 and training batches mix sources: avoid BatchNorm, or freeze its statistics.
- Tabular MLPs: BatchNorm or LayerNorm both help; inputs still need standardising ([[feature-engineering]]).
- NOT a substitute for scaling the input features; normalization layers act on hidden activations.

## Setup & code
```bash
pip install torch
```
The axis difference made explicit, plus the BatchNorm train/eval trap:
```python
import torch
from torch import nn

torch.manual_seed(0)
x = torch.randn(4, 6, 5, 5) * 3 + 2            # N, C, H, W

bn, ln, gn = nn.BatchNorm2d(6), nn.LayerNorm([6, 5, 5]), nn.GroupNorm(2, 6)
print("BN  per-channel mean over N,H,W:", bn(x).mean((0, 2, 3))[:3])
print("LN  per-sample mean over C,H,W:", ln(x).mean((1, 2, 3)))
print("GN  per-sample, per-group mean:", gn(x).view(4, 2, -1).mean(-1)[0])

class RMSNorm(nn.Module):                       # torch>=2.4 also ships nn.RMSNorm
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

tok = torch.randn(2, 7, 16) * 5
print("RMSNorm rms per token:", RMSNorm(16)(tok).pow(2).mean(-1).sqrt()[0, :3])

# BatchNorm: running stats only fill up during training passes.
bn = nn.BatchNorm1d(3)
data = torch.randn(512, 3) * 10 + 50
bn.eval();  print("eval before training :", bn(data[:4]).mean().item())   # stats still (0,1): ~46, unnormalised
bn.train()
for i in range(0, 512, 32):
    bn(data[i:i + 32])
bn.eval();  print("eval after warm stats:", bn(data[:4]).mean().item())   # ~0.6: now roughly standardised
```
Pre-norm transformer block skeleton:
```python
class PreNormBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ffn(self.n2(x))
```

## Choosing / trade-offs
- **BatchNorm vs per-sample norms.** BatchNorm adds mild regularisation and folds into the preceding conv at inference (zero runtime cost), but couples samples in a batch and breaks with small batches, distributed training without sync, and sequence padding. Per-sample norms avoid all of that at a small runtime cost.
- **LayerNorm vs RMSNorm.** RMSNorm skips the mean and bias: fewer ops, similar quality in LLMs. Keep LayerNorm when fine-tuning a model built with it.
- **GroupNorm groups.** 32 is the usual default; it must divide the channel count. Quality is close to BatchNorm at large batch and much better at tiny batch.
- **Pre-norm vs post-norm.** Pre-norm for anything deep or trained from scratch. Post-norm only to reproduce or fine-tune models that use it.

## Gotchas
- Forgetting `model.eval()` at validation lets BatchNorm use (and update) batch statistics, giving noisy metrics that depend on batch size. Forgetting `model.train()` afterwards freezes them.
- Fine-tuning a pretrained CNN with tiny batches corrupts its running statistics. Freeze BatchNorm (`eval()` on those modules and `requires_grad_(False)` on γ/β) or swap to GroupNorm.
- A batch of size 1 in BatchNorm training mode raises an error for 1-D inputs (or yields zero variance). Drop the last incomplete batch (`drop_last=True`) or handle it.
- BatchNorm statistics differ between train and deployment data (domain shift). Recomputing them on target data (AdaBN) is a cheap fix ([[transfer-learning-and-domain-adaptation]]).
- A bias in the Linear/Conv right before BatchNorm is redundant (β replaces it); set `bias=False`.
- Do not apply weight decay to γ/β by default; most recipes exclude norm parameters from decay ([[deep-learning-training]]).
- In mixed precision, compute normalization statistics in fp32; PyTorch autocast does this for its built-in norm layers, but custom implementations must cast.

## Related
- [[deep-learning-training]] - where normalization fits among optimizer, schedule and regularisation choices.
- [[weight-initialization]] - normalization reduces, but does not remove, init sensitivity.
- [[activation-functions]] - normalization keeps pre-activations in the useful range.
- [[transformers-and-attention]] - LayerNorm/RMSNorm placement in blocks.
- [[cnn-and-rnn-architectures]] - BatchNorm in CNNs, LayerNorm in RNNs.
- [[debugging-neural-network-training]] - symptoms of train/eval mode bugs.

## References
- PyTorch normalization layers: https://pytorch.org/docs/stable/nn.html#normalization-layers
- Ioffe & Szegedy, Batch Normalization: https://arxiv.org/abs/1502.03167
- Ba, Kiros & Hinton, Layer Normalization: https://arxiv.org/abs/1607.06450
- Zhang & Sennrich, Root Mean Square Layer Normalization: https://arxiv.org/abs/1910.07467
- Wu & He, Group Normalization: https://arxiv.org/abs/1803.08494
- Xiong et al., On Layer Normalization in the Transformer Architecture (pre- vs post-norm): https://arxiv.org/abs/2002.04745
