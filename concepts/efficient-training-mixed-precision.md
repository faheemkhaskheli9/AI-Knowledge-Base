---
title: Efficient training (mixed precision, gradient accumulation, checkpointing, torch.compile)
category: concepts
tags: [mixed-precision, amp, bf16, fp16, gradient-scaler, gradient-accumulation, gradient-checkpointing, activation-checkpointing, torch-compile, dataloader, gpu-memory, pytorch]
use_cases:
  - "train a model that runs out of GPU memory (CUDA OOM) on a single GPU"
  - "speed up a PyTorch training loop without changing the model"
  - "use a larger effective batch size than fits in memory"
  - "choose between bf16 and fp16 mixed precision"
  - "find out whether training is GPU-bound or data-loading-bound"
status: draft
last_verified: 2026-10-04
sources:
  - https://pytorch.org/docs/stable/amp.html
  - https://pytorch.org/docs/stable/notes/amp_examples.html
  - https://pytorch.org/docs/stable/checkpoint.html
  - https://pytorch.org/docs/stable/torch.compiler.html
  - https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
  - https://huggingface.co/docs/transformers/perf_train_gpu_one
---

# Efficient training (mixed precision, gradient accumulation, checkpointing, torch.compile)

## Summary
Most single-GPU training problems are "out of memory" or "too slow", and a small set of standard techniques fixes them without touching the model's maths: mixed precision (bf16/fp16) for speed and memory, gradient accumulation for a large effective batch, activation (gradient) checkpointing to trade compute for memory, `torch.compile` for kernel fusion, and a DataLoader that keeps the GPU fed. Apply them in that order, measuring after each, before reaching for multiple GPUs ([[distributed-training]]).

## Key concepts
- **Where memory goes.** Weights + gradients + optimizer state (Adam keeps 2 extra copies, so ~16 bytes/param in fp32 training) + activations saved for backward (grows with batch size × sequence length/image size × depth). Activations usually dominate for large inputs; optimizer state dominates for large models.
- **Mixed precision (AMP).** `torch.autocast` runs matmuls/convs in 16-bit while keeping numerically sensitive ops (softmax, norms, losses) and the master weights in fp32. Typically 1.5-3× faster on tensor-core GPUs and roughly halves activation memory.
  - **bf16**: same exponent range as fp32, no loss scaling needed. Default on Ampere (A100, RTX 30xx) and newer, and on TPUs.
  - **fp16**: narrower range; small gradients underflow to zero, so it needs `torch.amp.GradScaler`. Use on older GPUs (V100, T4, RTX 20xx).
- **Gradient accumulation.** Run `k` micro-batches, divide each loss by `k`, call `backward()` each time, step the optimizer once. Effective batch = micro-batch × k at the same memory. BatchNorm still sees only the micro-batch.
- **Activation checkpointing.** Do not store intermediate activations for chosen blocks; recompute them during backward. Cuts activation memory by a large factor for ~20-40% more compute. `torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)`; in Hugging Face, `model.gradient_checkpointing_enable()`.
- **`torch.compile(model)`.** Captures the model into a graph and fuses kernels; often a solid speed-up after a one-time compile (seconds to minutes). Works best with static shapes.
- **Input pipeline.** `num_workers > 0`, `pin_memory=True`, `persistent_workers=True`, `non_blocking=True` on `.to(device)`; decode/augment off the main process. If GPU utilisation (`nvidia-smi`) sits well below ~90%, the loader is the bottleneck.
- **Other memory levers.** Smaller optimizer state (8-bit Adam via bitsandbytes, or Adafactor), freezing layers / LoRA ([[fine-tuning-and-peft]]), `optimizer.zero_grad(set_to_none=True)` (default in recent PyTorch), and `channels_last` memory format for CNNs.

## When to use / scenarios
- CUDA OOM fine-tuning a vision or language model on one 12-24 GB GPU: bf16 autocast, then gradient checkpointing, then smaller micro-batch + accumulation.
- Training feels slow and the GPU is modern: bf16 autocast + `torch.compile` + DataLoader tuning, measure throughput (samples/s) before and after.
- Recipe from a paper says batch 256 but only 32 fits: accumulation steps = 8, keep the paper's LR.
- Long sequences or high-resolution images (segmentation, video): activation checkpointing is the main lever.
- NOT needed for small tabular nets or anything that already trains in minutes on CPU; NOT a substitute for [[distributed-training]] when the model weights alone exceed one GPU (use FSDP/DeepSpeed ZeRO there).

## Setup & code
```bash
pip install torch
```
One loop with AMP (bf16 or fp16 + scaler), gradient accumulation, activation checkpointing and optional `torch.compile`. Runs on CPU too (bf16 autocast), just without the speed-up:
```python
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = device == "cpu" or torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        return x + self.net(x)

class Model(nn.Module):
    def __init__(self, d=256, depth=6, n_classes=10, ckpt=True):
        super().__init__()
        self.inp = nn.Linear(32, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(depth))
        self.head = nn.Linear(d, n_classes)
        self.ckpt = ckpt
    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            # Recompute this block's activations in backward instead of storing them.
            x = checkpoint(b, x, use_reentrant=False) if self.ckpt and self.training else b(x)
        return self.head(x)

X, y = torch.randn(4096, 32), torch.randint(0, 10, (4096,))
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True,
                    num_workers=0, pin_memory=(device == "cuda"))   # raise num_workers for real data

model = Model().to(device)
# model = torch.compile(model)    # uncomment on Linux/GPU; first steps are slow while it compiles
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler(device, enabled=(amp_dtype == torch.float16))
loss_fn = nn.CrossEntropyLoss()
accum = 4                                       # effective batch = 64 * 4 = 256

model.train()
for step, (xb, yb) in enumerate(loader):
    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
    with torch.autocast(device_type=device, dtype=amp_dtype):
        loss = loss_fn(model(xb), yb) / accum   # average over micro-batches
    scaler.scale(loss).backward()               # no-op scaling when the scaler is disabled (bf16)
    if (step + 1) % accum == 0:
        scaler.unscale_(opt)                    # so clipping sees real gradient values
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
print(f"device={device} amp={amp_dtype} last loss={loss.item() * accum:.3f}")
if device == "cuda":
    print(f"peak memory {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
```
Measure, do not guess: compare samples/s and `torch.cuda.max_memory_allocated()` with each switch on and off. Hugging Face `Trainer` exposes the same switches as `bf16=True`, `fp16=True`, `gradient_accumulation_steps`, `gradient_checkpointing=True`, `torch_compile=True`.

## Choosing / trade-offs
- **bf16 vs fp16.** bf16 whenever the hardware supports it: same speed, no scaler, no overflow surprises. fp16 + `GradScaler` only on pre-Ampere GPUs.
- **Accumulation vs checkpointing.** Accumulation saves memory by shrinking the micro-batch and costs nothing extra per sample, but does not help when even batch size 1 OOMs (long sequences); checkpointing fixes that at ~20-40% more compute.
- **What to checkpoint.** Every transformer/residual block is the usual choice; checkpointing only some blocks trades memory and speed in between.
- **`torch.compile`.** Worth it for long runs with fixed shapes; skip for quick experiments, highly dynamic shapes, or when debugging (stack traces get harder). Support on Windows is more limited than on Linux.
- **Faster hardware vs more tricks.** If the model's weights + optimizer state alone do not fit, single-GPU tricks run out; move to FSDP/ZeRO, LoRA or quantised fine-tuning instead.

## Gotchas
- Forgetting to divide the loss by `accum` multiplies the effective learning rate by `accum`.
- Stepping the LR scheduler per micro-batch instead of per optimizer step makes warmup/decay `accum`× too fast.
- Clipping gradients without `scaler.unscale_(opt)` first clips the *scaled* gradients, so the threshold is meaningless under fp16.
- Calling `.half()` on the whole model is not mixed precision; fp16 master weights lose small updates. Use autocast.
- Losses computed outside the autocast region on fp16 tensors can overflow; let autocast handle dtype or cast logits to fp32 before the loss.
- `checkpoint` with dropout or other RNG ops recomputes with the same RNG state (handled by default `preserve_rng_state=True`); custom CUDA RNG code may not be.
- GPU sitting idle between batches is a data-loading problem; more AMP or compile will not help.
- `torch.backends.cudnn.benchmark = True` speeds up fixed-shape conv nets but makes runs non-deterministic.

## Related
- [[pytorch-basics]] - the plain training loop these techniques wrap.
- [[deep-learning-training]] - LR, batch size and schedule choices that interact with accumulation.
- [[debugging-neural-network-training]] - NaNs and overflow that fp16 can introduce.
- [[distributed-training]] - the next step when one GPU is not enough.
- [[fine-tuning-and-peft]] - LoRA/QLoRA to cut weight and optimizer memory.
- [[quantization]] - low-precision weights for inference, a different goal from AMP training.

## References
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html
- PyTorch AMP examples (accumulation, clipping, scaler): https://pytorch.org/docs/stable/notes/amp_examples.html
- PyTorch activation checkpointing: https://pytorch.org/docs/stable/checkpoint.html
- torch.compile: https://pytorch.org/docs/stable/torch.compiler.html
- PyTorch performance tuning guide: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- Hugging Face, efficient training on one GPU: https://huggingface.co/docs/transformers/perf_train_gpu_one
