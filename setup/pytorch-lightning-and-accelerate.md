---
title: PyTorch Lightning and Hugging Face Accelerate
category: setup
tags: [pytorch-lightning, lightning-fabric, accelerate, multi-gpu, mixed-precision, ddp, fsdp, deepspeed, training-loop]
use_cases:
  - "run my existing PyTorch training script on 4 GPUs without rewriting it for DDP"
  - "add bf16 mixed precision and gradient accumulation to a training loop with a few lines"
  - "pick between Lightning, Fabric and Accelerate for a new deep learning project"
  - "move a single-GPU fine-tuning script to FSDP or DeepSpeed on a multi-node cluster"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/Lightning-AI/pytorch-lightning
  - https://lightning.ai/docs/pytorch/stable/
  - https://lightning.ai/docs/fabric/stable/
  - https://github.com/huggingface/accelerate
  - https://huggingface.co/docs/accelerate/index
---

# PyTorch Lightning and Hugging Face Accelerate

## Summary
PyTorch Lightning and Hugging Face Accelerate remove the device, distributed and mixed-precision boilerplate from PyTorch training. Lightning gives you a structured `LightningModule` plus a `Trainer` that owns the loop. Lightning Fabric and Accelerate keep your own loop and wrap only the model, optimizer and dataloaders. All three switch between CPU, one GPU, multi-GPU DDP, FSDP and DeepSpeed through configuration rather than code changes. Both projects are Apache-2.0 licensed.

## Key concepts
- **Two styles**: *framework* (Lightning `Trainer`: you fill in hooks such as `training_step`, `configure_optimizers`; the loop, logging, checkpointing and early stopping are built in) vs *thin wrapper* (Fabric, Accelerate: you keep a plain PyTorch loop and add a handful of calls).
- **Accelerate core calls**: `accelerator = Accelerator(mixed_precision="bf16")`, `model, opt, dl = accelerator.prepare(model, opt, dl)`, `accelerator.backward(loss)` in place of `loss.backward()`, `with accelerator.accumulate(model):` for gradient accumulation, `accelerator.gather_for_metrics(...)` for evaluation across processes.
- **Launchers**: `accelerate config` writes a YAML (GPUs, nodes, FSDP/DeepSpeed, precision) and `accelerate launch train.py` runs it; Lightning spawns processes itself from `Trainer(devices=..., num_nodes=...)` or under `torchrun`/SLURM.
- **Strategies**: `ddp` (replicate model, all-reduce gradients), `fsdp` (shard params, grads and optimizer state), `deepspeed` (ZeRO stages). Theory lives in [[distributed-training]].
- **Precision**: Lightning uses strings such as `"bf16-mixed"`, `"16-mixed"`, `"32-true"`; Accelerate uses `mixed_precision="bf16" | "fp16" | "no"`. fp16 needs loss scaling, bf16 usually does not ([[efficient-training-mixed-precision]]).
- **Rank awareness**: every process runs your script. Print, save and log only on the main process (`accelerator.is_main_process`, Lightning's `rank_zero_only`).

## When to use / scenarios
- A research or product team training custom vision, tabular or speech models on 1-8 GPUs that wants multi-GPU and AMP without writing DDP code: Lightning `Trainer`.
- An existing hand-written PyTorch loop (or a Hugging Face model) that must now run on several GPUs or with DeepSpeed: Accelerate or Fabric, a few changed lines.
- LLM fine-tuning: Hugging Face `Trainer` and TRL already use Accelerate under the hood, so `accelerate launch` + an FSDP/DeepSpeed config is the usual path ([[fine-tuning-and-peft]]).
- Not needed for a single-GPU notebook experiment where the plain loop in [[pytorch-basics]] is clearer, or for JAX/Keras stacks ([[jax-and-flax]], [[keras-and-tensorflow]]).
- Very large pretraining runs (thousands of GPUs, tensor/pipeline parallelism) usually move to dedicated frameworks such as Megatron-style trainers; see [[pretraining-and-scaling-laws]].

## Setup & code
```bash
uv pip install torch lightning accelerate      # torch build per gpu-cuda-setup
accelerate config                               # interactive: GPUs, precision, FSDP/DeepSpeed
accelerate launch train_accel.py                # or: accelerate launch --multi_gpu --num_processes 2 train_accel.py
python train_lightning.py                       # Lightning spawns its own processes
```

Accelerate (your loop, four changed lines):
<!-- skip-check: needs torch and ideally a GPU -->
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)
X, y = torch.randn(4096, 20), torch.randint(0, 2, (4096,))
dl = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
model = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
model, opt, dl = accelerator.prepare(model, opt, dl)   # moves to device, wraps DDP/FSDP

for epoch in range(3):
    for xb, yb in dl:
        with accelerator.accumulate(model):              # syncs grads only every 4 steps
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            accelerator.backward(loss)                   # handles scaling / sharding
            opt.step(); opt.zero_grad()
    accelerator.print(f"epoch {epoch} loss {loss.item():.3f}")   # main process only
accelerator.save_model(model, "ckpt")                     # unwraps and saves once
```

Lightning (structured module + Trainer):
<!-- skip-check: needs torch and lightning -->
```python
import torch, lightning as L
from torch.utils.data import DataLoader, TensorDataset

class Clf(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
    def training_step(self, batch, _):
        x, y = batch
        loss = torch.nn.functional.cross_entropy(self.net(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

dl = DataLoader(TensorDataset(torch.randn(4096, 20), torch.randint(0, 2, (4096,))), batch_size=64)
trainer = L.Trainer(accelerator="auto", devices="auto", strategy="auto",
                    precision="bf16-mixed", accumulate_grad_batches=4, max_epochs=3)
trainer.fit(Clf(), dl)
```

What the wrappers do for you, shown in numpy: data-parallel gradient averaging equals the full-batch gradient, and gradient accumulation over micro-batches equals one big batch. This is why switching `devices` or `accumulate_grad_batches` needs no change to the model code (only the learning rate may need retuning for the larger effective batch).
```python
import numpy as np

rng = np.random.default_rng(0)
X, y = rng.normal(size=(256, 5)), rng.normal(size=256)
w = rng.normal(size=5)

def grad(Xb, yb, w):                        # MSE gradient, mean over the batch
    return 2 * Xb.T @ (Xb @ w - yb) / len(yb)

full = grad(X, y, w)

# DDP: 4 "GPUs" each get a shard; all-reduce averages their gradients
shards = np.array_split(np.arange(256), 4)
ddp = np.mean([grad(X[s], y[s], w) for s in shards], axis=0)

# Gradient accumulation: 8 micro-batches, loss divided by the number of steps
micro = np.array_split(np.arange(256), 8)
acc = sum(grad(X[m], y[m], w) / len(micro) for m in micro)

print("max |ddp - full| =", np.abs(ddp - full).max())
print("max |acc - full| =", np.abs(acc - full).max())
assert np.allclose(ddp, full) and np.allclose(acc, full)
```

## Choosing / trade-offs
| Need | Pick |
|---|---|
| Batteries included: loop, callbacks, checkpointing, early stopping, loggers | Lightning `Trainer` |
| Keep my loop, want Lightning-style device/strategy switching | Lightning Fabric |
| Keep my loop, Hugging Face models, TRL, PEFT, DeepSpeed/FSDP via config | Accelerate |
| Full control, few GPUs, no extra dependency | plain PyTorch DDP ([[distributed-training]]) |

- Lightning's structure helps teams standardize projects but hides the loop; unusual training (GANs, RL, multiple optimizers) needs manual optimization mode or Fabric.
- Accelerate is lighter and is the default in the Hugging Face ecosystem; you write logging, checkpoint rotation and early stopping yourself (or use HF `Trainer`).
- Strategy choice (DDP vs FSDP vs DeepSpeed ZeRO) matters more for memory than the wrapper does; the wrappers just make it a config switch.
- Both add a dependency whose versions must match your torch build; pin them together.

## Gotchas
- Every process executes top-level code: guard downloads, prints and file writes with main-process checks, or N GPUs download N copies.
- Effective batch = per-device batch x devices x accumulation steps; scale the learning rate or warmup when you change any of them ([[batch-size-and-gradient-noise]]).
- `DistributedSampler` and shuffling are handled by `prepare`/Trainer; do not add your own sampler on top or each rank sees the same data.
- Metrics computed per process are partial: use `gather_for_metrics` (Accelerate) or `sync_dist=True` / torchmetrics (Lightning), or validation numbers are wrong.
- Saving the wrapped model saves DDP/FSDP wrappers; use `accelerator.unwrap_model` / `save_model` or Lightning checkpoints, and gather FSDP state dicts properly ([[model-checkpointing-and-export]]).
- fp16 without loss scaling silently underflows; prefer bf16 on GPUs that support it.
- Notebooks cannot fork CUDA after it is initialized; use `notebook_launcher` (Accelerate) or a script.

## Related
- [[pytorch-basics]] - the plain loop these libraries wrap.
- [[distributed-training]] - DDP, FSDP, ZeRO and parallelism theory.
- [[efficient-training-mixed-precision]] - AMP, loss scaling, checkpointing, accumulation.
- [[fine-tuning-and-peft]] - LoRA/QLoRA runs launched with Accelerate.
- [[huggingface-transformers]] - models and the HF Trainer built on Accelerate.
- [[experiment-tracking]] - loggers both libraries plug into.
- [[gpu-cuda-setup]] - drivers, NCCL and torch builds.
- [[gpu-cloud-options]] - renting multi-GPU machines.

## References
- PyTorch Lightning repository and README: https://github.com/Lightning-AI/pytorch-lightning
- Lightning docs (Trainer, precision, strategies): https://lightning.ai/docs/pytorch/stable/
- Lightning Fabric docs: https://lightning.ai/docs/fabric/stable/
- Hugging Face Accelerate repository: https://github.com/huggingface/accelerate
- Accelerate docs (launch, FSDP, DeepSpeed, gradient accumulation): https://huggingface.co/docs/accelerate/index
