---
title: Model checkpointing, saving and export (state_dict, safetensors, ONNX)
category: concepts
tags: [checkpointing, state-dict, safetensors, onnx, torch-export, onnxruntime, resume-training, serialization, pytorch]
use_cases:
  - "resume training after a crash or a spot-instance preemption"
  - "save a trained model so another person or service can load it safely"
  - "export a PyTorch model to run without Python or on another runtime"
  - "keep the best and the last checkpoint during training"
status: draft
last_verified: 2026-10-04
sources:
  - https://pytorch.org/tutorials/beginner/saving_loading_models.html
  - https://pytorch.org/docs/stable/generated/torch.load.html
  - https://huggingface.co/docs/safetensors/index
  - https://pytorch.org/docs/stable/onnx.html
  - https://onnxruntime.ai/docs/
---

# Model checkpointing, saving and export (state_dict, safetensors, ONNX)

## Summary
There are three kinds of saved models, with different contents and audiences. A **training checkpoint** holds everything needed to continue training: weights, optimizer and scheduler state, epoch, and RNG state. A **weights artifact** holds only the parameters, for sharing or fine-tuning. Use `safetensors` because it cannot run code. An **exported graph** (ONNX, `torch.export`, TorchScript) holds weights plus the computation, so it runs without your Python class, in onnxruntime, TensorRT, mobile or browser runtimes. Most "my resumed run diverged" and "I can't load this model" problems come from saving the wrong kind.

## Key concepts
- **`state_dict`, not the module.** `torch.save(model)` pickles the class by import path, so it breaks when the code moves. `torch.save(model.state_dict())` saves a dict of tensors; rebuild the model from code and `load_state_dict`.
- **What a resumable checkpoint needs.** Model, optimizer (Adam moments), LR scheduler, GradScaler (AMP), epoch/step, best metric, and RNG states (torch, CUDA, numpy, Python), plus the data sampler position if you resume mid-epoch. Missing optimizer state makes the loss jump after resume.
- **Pickle is code execution.** `torch.load` of an untrusted `.pt`/`.bin` can run arbitrary code. `weights_only=True` (the default since PyTorch 2.6) restricts loading to tensors and plain containers. `safetensors` is a flat tensor format with no code at all, and loads zero-copy via memory map.
- **Atomic writes.** Write to `file.tmp`, then `os.replace` it onto the real name. A crash mid-write then never leaves a corrupt "latest" checkpoint.
- **Export = trace the graph.** `torch.export` captures the forward pass as a static graph with declared dynamic dimensions. `torch.onnx.export(..., dynamo=True)` builds on it and writes ONNX for any ONNX runtime. Python control flow that depends on data values must be expressible in the graph (`torch.cond`) or it gets specialized away.

## When to use / scenarios
- Long or preemptible training (spot GPUs, shared clusters): save a full checkpoint every N steps, keep `last` and `best` ([[gpu-cloud-options]]).
- Publishing a model (Hugging Face Hub, a teammate, a customer): `safetensors` weights + config. `save_pretrained` does this for HF models ([[huggingface-transformers]]).
- Serving from a non-Python stack (C++, C#, Java), CPU-only boxes, mobile, browser: export to ONNX and run with onnxruntime ([[edge-on-device]]).
- GPU serving with TensorRT or Triton: ONNX is the usual entry point ([[inference-servers-vllm]]).
- NOT needed: LLM serving with vLLM/llama.cpp, which load HF safetensors or GGUF directly ([[llama-cpp-gguf]]). Plain Python FastAPI serving can just load a `state_dict` ([[serving-with-fastapi]]).

## Setup & code
```bash
pip install torch safetensors onnx onnxscript onnxruntime
```
Resumable checkpoint, safetensors weights, and ONNX export:
```python
import os
import tempfile

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

d = tempfile.mkdtemp()
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10)
x = torch.randn(8, 4)
model(x).sum().backward(); opt.step(); sched.step()

# 1) Resumable training checkpoint: everything needed to continue, not just weights.
ckpt = os.path.join(d, "last.pt")
tmp = ckpt + ".tmp"
torch.save({"epoch": 1, "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(), "rng": torch.get_rng_state()}, tmp)
os.replace(tmp, ckpt)                                   # atomic: no half-written file on crash

state = torch.load(ckpt, weights_only=True)             # no arbitrary code execution
model2 = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
model2.load_state_dict(state["model"])
opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
opt2.load_state_dict(state["optimizer"])
print("resumed epoch", state["epoch"], "| same output:", torch.equal(model(x), model2(x)))

# 2) Weights-only artifact for sharing: safetensors (no pickle).
st = os.path.join(d, "model.safetensors")
save_file(model.state_dict(), st)
model3 = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
model3.load_state_dict(load_file(st))
print("safetensors keys:", list(load_file(st).keys()))

# 3) Portable graph for serving: ONNX via the torch.export-based exporter.
model.eval()
onnx_path = os.path.join(d, "model.onnx")
torch.onnx.export(model, (x,), onnx_path, dynamo=True,
                  dynamic_shapes={"input": {0: torch.export.Dim("batch")}},
                  input_names=["input"], output_names=["logits"], verbose=False)
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path)
out = sess.run(None, {"input": torch.randn(3, 4).numpy()})[0]
print("onnxruntime output shape:", out.shape)
```
Output (torch 2.13.0 CPU, safetensors 0.8.0, onnx 1.22.0, onnxruntime 1.29.0):
```
resumed epoch 1 | same output: True
safetensors keys: ['0.bias', '0.weight', '2.bias', '2.weight']
onnxruntime output shape: (3, 2)
```
The ONNX model was exported with batch 8 and run with batch 3, which works because the batch dimension was declared dynamic. On Windows, set `PYTHONIOENCODING=utf-8` if the exporter's progress log raises `UnicodeEncodeError`.

## Choosing / trade-offs
- **`.pt` checkpoint vs safetensors.** Use `.pt` (loaded with `weights_only=True`) for your own resumable checkpoints, since optimizer state contains non-tensor values. Use safetensors for anything shared or downloaded.
- **ONNX vs `torch.export` / AOTInductor vs TorchScript.** ONNX gives the widest runtime and hardware reach (onnxruntime, TensorRT, OpenVINO, mobile, web). `torch.export` + AOTInductor keeps you in the PyTorch ecosystem with compiled C++ inference. TorchScript is in maintenance mode; do not start new work on it.
- **Checkpoint frequency.** More often means less lost work but more I/O and storage. Rule of thumb: lose at most ~15-30 minutes of compute. Keep `last` plus the top-k by validation metric, and delete the rest.
- **Sharded vs single file.** Multi-GPU (FSDP) runs save sharded checkpoints with `torch.distributed.checkpoint`. They are faster to write and can be resharded to a different GPU count on load ([[distributed-training]]).
- **Precision of the artifact.** Saving fp16/bf16 halves the size; quantize after export for edge targets ([[quantization]]).

## Gotchas
- Wrapped models change key names: `DataParallel`/DDP add a `module.` prefix, `torch.compile` adds `_orig_mod.`. Save `model.module.state_dict()` (or the uncompiled model), or strip the prefix on load.
- `load_state_dict(strict=False)` silently skips mismatched keys and leaves random weights. Print the returned `missing_keys`/`unexpected_keys`.
- Call `model.eval()` before export and inference. Dropout and BatchNorm in train mode give random, batch-dependent outputs.
- Create the optimizer **after** moving the model to its device, and load the optimizer state after that. Loading on the wrong device mixes CPU and GPU tensors.
- Resuming with a fresh LR scheduler restarts warmup, which often spikes the loss. Restore the scheduler state ([[learning-rate-schedules]]).
- Checkpoints without the code version, config and data version cannot be reproduced. Log them alongside the file ([[experiment-tracking]]).
- Verify an export numerically: compare ONNX vs PyTorch outputs on real inputs (`np.allclose` with a tolerance) and on more than one batch size. Unsupported ops and shapes specialized during tracing are the usual failures.
- Do not `torch.load` downloaded files without `weights_only=True`, and do not override it to silence an error. Convert the file to safetensors instead ([[ai-security-privacy-compliance]]).

## Related
- [[pytorch-basics]] - the training loop that produces these checkpoints.
- [[deep-learning-training]] - when to checkpoint, early stopping on the best checkpoint.
- [[distributed-training]] - sharded checkpoints for FSDP/multi-GPU.
- [[edge-on-device]] - running exported ONNX models on phones and edge devices.
- [[serving-with-fastapi]] - loading a saved model in an API.
- [[quantization]] - shrinking the exported artifact.
- [[experiment-tracking]] - storing checkpoints as versioned artifacts.

## References
- PyTorch, Saving and loading models: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch, `torch.load` (`weights_only`): https://pytorch.org/docs/stable/generated/torch.load.html
- Hugging Face, safetensors: https://huggingface.co/docs/safetensors/index
- PyTorch, ONNX export (`torch.onnx.export`, `dynamo=True`): https://pytorch.org/docs/stable/onnx.html
- PyTorch, `torch.export`: https://pytorch.org/docs/stable/export.html
- ONNX Runtime docs: https://onnxruntime.ai/docs/
