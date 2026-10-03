---
title: PyTorch basics
category: ml
tags: [pytorch, deep-learning, training-loop, autograd, gpu, dataloader, torch]
use_cases:
  - "train a custom neural network on my own dataset"
  - "write a clean training loop with GPU, mixed precision and checkpointing"
  - "fine-tune a pretrained vision or text model on a small labelled set"
  - "debug a model that does not learn (loss stuck or NaN)"
  - "export a trained model for serving"
status: draft
last_verified: 2026-10-03
sources:
  - https://pytorch.org/tutorials/beginner/basics/intro.html
  - https://pytorch.org/get-started/locally/
  - https://pytorch.org/docs/stable/notes/amp_examples.html
---

# PyTorch basics

## Summary
PyTorch is the dominant deep-learning framework: tensors with autograd, `nn.Module` for models, `Dataset`/`DataLoader` for data, and an explicit training loop you control. Almost every modern open model (Hugging Face, YOLO, Whisper) is built on it. Learn the loop once and everything else is configuration.

## Key concepts
- Tensor: n-d array on CPU or GPU (`device`), dtype matters (float32, bfloat16, float16).
- Autograd: `loss.backward()` computes gradients; `optimizer.step()` updates; `optimizer.zero_grad()` clears.
- `nn.Module`: define layers in `__init__`, computation in `forward`.
- `Dataset` (`__len__`, `__getitem__`) + `DataLoader` (batching, shuffling, `num_workers`).
- `model.train()` vs `model.eval()` (dropout, batch norm) and `torch.no_grad()` / `inference_mode()` for evaluation.
- Mixed precision (`torch.autocast`) and `torch.compile` for speed; checkpointing with `state_dict`.
- Transfer learning: load pretrained weights, replace the head, freeze or fine-tune the backbone.

## When to use / scenarios
- Custom models for images, audio, text, time series or multimodal data where off-the-shelf APIs do not fit.
- Fine-tuning pretrained backbones (see [[image-classification]], [[fine-tuning-and-peft]]).
- Research or teaching: explicit control and easy debugging.
- NOT for: tabular data (see [[gradient-boosting-tabular]]); calling a hosted LLM/API (no training needed); production serving by itself (see [[serving-with-fastapi]]).

## Setup & code
Install the build matching your CUDA/CPU from https://pytorch.org/get-started/locally/ (see [[gpu-cuda-setup]]).
```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
X = torch.randn(1000, 20)
y = (X[:, 0] + X[:, 1] > 0).long()
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    model.eval()
    with torch.inference_mode():
        acc = (model(X.to(device)).argmax(1).cpu() == y).float().mean().item()
    print(epoch, round(loss.item(), 4), round(acc, 3))

torch.save(model.state_dict(), "model.pt")   # reload: model.load_state_dict(torch.load("model.pt"))
```

## Choosing / trade-offs
- Raw loop vs a wrapper (PyTorch Lightning, Hugging Face Trainer/Accelerate): wrappers cut boilerplate for multi-GPU, logging and checkpointing; the raw loop is clearer for small or unusual setups.
- Mixed precision (bfloat16 on recent GPUs) halves memory and speeds training; fp16 needs a `GradScaler`.
- Batch size vs memory: use gradient accumulation to simulate larger batches.
- Export: `torch.export`/ONNX for portable inference; check operator coverage per target.

## Gotchas
- Forgetting `model.eval()` and `no_grad` at inference gives wrong, slow results.
- Forgetting `zero_grad()` accumulates gradients across steps.
- Tensors and model on different devices raise runtime errors.
- Loss expects specific shapes/dtypes (`CrossEntropyLoss` takes raw logits and class indices, not softmax output).
- Data leakage and a missing seed make runs irreproducible; set seeds and split before augmentation.
- `torch.load` on untrusted files can execute code; prefer `weights_only=True` or safetensors.
- Overfit a tiny batch first; if it cannot reach near-zero loss, the bug is in the code, not the data.

## Related
- [[fine-tuning-and-peft]] - adapting large pretrained models.
- [[image-classification]] - transfer learning in vision.
- [[experiment-tracking]] - log loss curves and configs.
- [[gpu-cuda-setup]] - installing the right build.
- [[huggingface-transformers]] - pretrained models on top of PyTorch.

## References
- https://pytorch.org/tutorials/beginner/basics/intro.html
- https://pytorch.org/get-started/locally/
- https://pytorch.org/docs/stable/notes/amp_examples.html
