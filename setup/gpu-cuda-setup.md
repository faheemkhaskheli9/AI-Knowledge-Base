---
title: GPU, drivers, CUDA and PyTorch setup
category: setup
tags: [gpu, cuda, nvidia, driver, pytorch, vram, nvidia-smi]
use_cases:
  - "get PyTorch to see my NVIDIA GPU on Windows or Linux"
  - "figure out whether a 7B/13B/70B model fits in my GPU memory"
  - "fix 'CUDA out of memory' or 'torch.cuda.is_available() is False'"
  - "choose a GPU for local fine-tuning or inference on a budget"
status: draft
last_verified: 2026-10-03
sources:
  - https://pytorch.org/get-started/locally/
  - https://docs.astral.sh/uv/guides/integration/pytorch/
  - https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
---

# GPU, drivers, CUDA and PyTorch setup

## Summary
Local GPU work needs three layers to agree: the NVIDIA driver, the CUDA runtime that your framework was built against, and the framework build itself. With pip/uv wheels, PyTorch bundles its own CUDA runtime, so you normally only install the driver and pick the matching wheel; a system CUDA Toolkit is needed only to compile extensions. This file also gives VRAM sizing rules of thumb.

## Key concepts
- Driver (system) vs CUDA runtime (in the wheel) vs CUDA Toolkit (nvcc, only to compile). `nvidia-smi` shows the driver and the highest CUDA version it supports, not what is installed.
- A newer driver runs older CUDA runtimes (backward compatible); the wheel's CUDA version must be <= what the driver supports.
- PyTorch wheels are tagged `cuXXX` (e.g. cu130) and served from `https://download.pytorch.org/whl/<tag>`; the exact set of tags changes each release, so read the selector at pytorch.org/get-started/locally.
- Compute capability (architecture generation) must be supported by the build; very new GPUs may need a recent PyTorch.
- Apple Silicon uses MPS (Metal), AMD uses ROCm; both are separate wheels/builds.

## When to use / scenarios
- Setting up a workstation or laptop with an RTX card for inference or QLoRA fine-tuning.
- A server that suddenly reports "no CUDA GPUs available" after a driver update.
- Sizing: deciding between a 24 GB consumer card and renting an 80 GB datacenter GPU.
- If you only call hosted APIs, you need none of this.

## Setup & code
1. Install the latest NVIDIA driver (Windows: NVIDIA App/driver installer; Linux: distro package or NVIDIA repo). Verify:
```bash
nvidia-smi
```
2. Install PyTorch matching it (uv):
```bash
uv pip install torch torchvision --torch-backend=auto
```
or pip with an explicit index (take the exact tag/command from the PyTorch selector):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
3. Check:
```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())  # (free, total) bytes
```
WSL2: install the driver on Windows only; never install a Linux NVIDIA driver inside WSL (see [[windows-wsl-setup]]).

### VRAM rules of thumb (inference, weights only)
- Bytes per parameter: FP32 4, FP16/BF16 2, INT8 1, 4-bit ~0.5 (GGUF Q4 about 0.55-0.6 with overhead).
- Weights GB ~ params (billions) x bytes/param. 7B: ~14 GB FP16, ~4-5 GB 4-bit. 70B: ~140 GB FP16, ~40 GB 4-bit.
- Add KV cache: grows linearly with context length and batch size; budget 10-30% extra for short contexts, far more for long contexts or many concurrent users. Add ~1-2 GB for runtime/activations.
- Full fine-tuning with Adam needs roughly 12-16+ bytes/param (weights, grads, optimizer states); LoRA/QLoRA cuts it to base weights (quantized) plus small adapters and activations.
These are estimates; measure with `torch.cuda.max_memory_allocated()`.

## Choosing / trade-offs
- More VRAM beats more speed for LLMs: a model that does not fit is slower than a smaller one that does.
- Consumer card (local, cheap, 8-24 GB class) vs datacenter card (more memory, ECC, rented): see [[gpu-cloud-options]].
- Quantize to fit ([[quantization]]) at some quality cost; offload layers to CPU RAM when it will not fit (slow).
- Multiple small GPUs need tensor/pipeline parallelism; PCIe-only cards limit scaling.

## Gotchas
- `torch.cuda.is_available()` False usually means a CPU-only wheel (common on Windows with plain PyPI) or a driver too old for the wheel.
- Installing a full CUDA Toolkit does not change which CUDA PyTorch uses.
- Mixed `pip` and `conda` torch installs shadow each other; use one environment manager.
- OOM at the first batch is weights; OOM after many steps is activations/KV-cache growth or memory fragmentation.
- Another process (browser, display compositor) can hold 1-2 GB on the same GPU; check `nvidia-smi`.
- Laptop GPUs may be hybrid: ensure the app runs on the discrete GPU.

## Related
- [[python-env-uv]] - env and index handling.
- [[quantization]] - shrink the memory footprint.
- [[fine-tuning-and-peft]] - training memory needs.
- [[inference-servers-vllm]] - serving with KV-cache management.
- [[gpu-cloud-options]] - renting GPUs instead.
- [[windows-wsl-setup]] - GPU in WSL2.

## References
- PyTorch install selector: https://pytorch.org/get-started/locally/
- uv PyTorch guide: https://docs.astral.sh/uv/guides/integration/pytorch/
- NVIDIA CUDA docs: https://docs.nvidia.com/cuda/
