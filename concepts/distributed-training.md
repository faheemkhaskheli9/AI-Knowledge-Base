---
title: Distributed and multi-GPU training (DDP, FSDP, DeepSpeed)
category: concepts
tags: [distributed-training, multi-gpu, ddp, fsdp, deepspeed, zero, data-parallel, model-parallel, torchrun, accelerate, pytorch]
use_cases:
  - "speed up training by using all GPUs on one machine"
  - "train or fine-tune a model too large to fit on one GPU"
  - "scale a training job across several nodes"
  - "convert a single-GPU PyTorch script to multi-GPU with minimal changes"
status: draft
last_verified: 2026-10-04
sources:
  - https://docs.pytorch.org/tutorials/beginner/ddp_series_theory.html
  - https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
  - https://www.deepspeed.ai/tutorials/zero/
  - https://huggingface.co/docs/accelerate/index
---

# Distributed and multi-GPU training (DDP, FSDP, DeepSpeed)

## Summary
Distributed training splits work across GPUs either to go faster (data
parallel: each GPU trains a full model copy on different batches and gradients
are averaged) or to fit a model that does not fit on one GPU (sharding
parameters, gradients and optimizer state with FSDP or DeepSpeed ZeRO, or
splitting layers with tensor/pipeline parallelism). In PyTorch, start with DDP;
move to FSDP/ZeRO only when memory forces you to.

## Key concepts
- Process per GPU: launched with `torchrun`; each process has a `RANK`,
  `LOCAL_RANK` and `WORLD_SIZE`, and talks over NCCL (GPUs) or Gloo (CPU).
- DDP (DistributedDataParallel): full model on every GPU; gradients all-reduced
  during backward. Near-linear speedup when the model fits per GPU.
- `DistributedSampler`: gives each rank a different shard of the data; call
  `sampler.set_epoch(epoch)` so shuffling differs per epoch.
- Effective batch size = per-GPU batch x world size x gradient accumulation
  steps; the learning rate usually needs to scale with it.
- Memory per GPU for training is dominated by parameters + gradients +
  optimizer state (Adam keeps two extra copies) + activations.
- FSDP / ZeRO: shard optimizer state (ZeRO-1), + gradients (ZeRO-2), +
  parameters (ZeRO-3 / FSDP full shard) across ranks; gather on the fly.
  Trades communication for memory. ZeRO-Offload moves state to CPU/NVMe.
- Tensor parallel (split matrices within a layer) and pipeline parallel
  (split layers across GPUs) are for very large models and fast interconnects.
- Activation checkpointing and mixed precision (bf16) cut memory on any setup.

## When to use / scenarios
- Training a CNN/transformer on one multi-GPU workstation: DDP.
- Full fine-tuning of a multi-billion-parameter LLM: FSDP or DeepSpeed ZeRO-3,
  or avoid it with LoRA/QLoRA on one GPU ([[fine-tuning-and-peft]]).
- Multi-node clusters (Slurm, Kubernetes): same code, `torchrun` with
  rendezvous settings per node.
- NOT needed when one GPU finishes in acceptable time, when the bottleneck is
  data loading, or for inference serving (see [[inference-servers-vllm]]).

## Setup & code
Minimal DDP script, `train_ddp.py`:
```python
import os, torch, torch.distributed as dist, torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

def main():
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    ds = TensorDataset(torch.randn(4096, 32), torch.randn(4096, 1))
    sampler = DistributedSampler(ds, shuffle=True)
    dl = DataLoader(ds, batch_size=64, sampler=sampler)

    model = DDP(nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)).to(device),
                device_ids=[local_rank] if device.type == "cuda" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(3):
        sampler.set_epoch(epoch)
        for x, y in dl:
            loss = nn.functional.mse_loss(model(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        if rank == 0:                       # log/save from one rank only
            print(f"epoch {epoch} loss {loss.item():.4f}")
    if rank == 0:
        torch.save(model.module.state_dict(), "model.pt")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```
```bash
torchrun --nproc_per_node=4 train_ddp.py      # 4 GPUs on one machine
torchrun --nproc_per_node=2 train_ddp.py      # also runs on CPU (gloo) for a smoke test
```
FSDP2 (recent PyTorch) shards each block, then the root:
```python
from torch.distributed.fsdp import fully_shard
for block in model.layers:      # e.g. transformer blocks
    fully_shard(block)
fully_shard(model)
```
For Hugging Face models, Accelerate runs the same script on DDP, FSDP or
DeepSpeed via `accelerate config` then `accelerate launch train.py`; the
`Trainer` takes `--deepspeed ds_config.json` or FSDP options directly.

## Choosing / trade-offs
- Model + optimizer state fit on one GPU: DDP (simplest, fastest).
- Do not fit: FSDP (native PyTorch) or DeepSpeed ZeRO (rich config, CPU/NVMe
  offload). Similar capability; pick what your framework integrates best.
- Few GPUs, huge model: parameter-efficient fine-tuning + quantization
  usually beats sharding on cost ([[quantization]]).
- Multi-node: interconnect bandwidth (NVLink, InfiniBand) decides whether
  ZeRO-3/tensor parallel scale; on slow Ethernet prefer data parallel with
  larger per-GPU batches.
- Raw PyTorch vs Accelerate/Lightning: wrappers cut boilerplate and handle
  checkpoint sharding; raw is easier to debug.

## Gotchas
- Forgetting `DistributedSampler` (or `set_epoch`) makes every GPU train on
  the same data, or the same order every epoch.
- Logging, evaluation and checkpoint saving from every rank: duplicated
  output and file write races. Guard with `rank == 0`, and use
  `dist.barrier()` before other ranks read the file.
- Ranks that take different code paths (a conditional backward, unused
  parameters) hang in all-reduce; DDP has `find_unused_parameters=True` at a cost.
- Saving a DDP model: save `model.module.state_dict()`; FSDP needs its
  distributed checkpoint APIs to save/load sharded state.
- Scaling batch size without adjusting learning rate and warmup changes
  convergence ([[deep-learning-training]]).
- BatchNorm statistics are per GPU; use `SyncBatchNorm` for small per-GPU batches.
- Speedup below expectations is usually the data loader or the network, not the model.
- Windows: NCCL is Linux-only, so multi-GPU training belongs on Linux/WSL.
  Windows PyTorch wheels lack libuv, so even a CPU/gloo smoke test needs
  `USE_LIBUV=0`, and `torchrun` itself can still fail on that error; then set
  `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK` and start
  one process per rank by hand.

## Related
- [[deep-learning-training]] - optimizers, learning rate and mixed precision.
- [[pytorch-basics]] - the single-GPU loop this extends.
- [[fine-tuning-and-peft]] - often a cheaper alternative to sharded full fine-tuning.
- [[pretraining-and-scaling-laws]] - where large-scale parallelism is required.
- [[gpu-cloud-options]] - renting multi-GPU machines.
- [[gpu-cuda-setup]] - drivers and NCCL prerequisites.

## References
- PyTorch DDP tutorial series: https://docs.pytorch.org/tutorials/beginner/ddp_series_theory.html
- PyTorch FSDP2 tutorial: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/
- Hugging Face Accelerate: https://huggingface.co/docs/accelerate/index
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models": https://arxiv.org/abs/1910.02054
