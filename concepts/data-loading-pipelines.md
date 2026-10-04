---
title: Data loading pipelines (PyTorch Dataset/DataLoader, tf.data)
category: concepts
tags: [deep-learning, data-loading, dataset, dataloader, num-workers, pin-memory, collate-fn, iterable-dataset, tf-data, gpu-utilization, pytorch]
use_cases:
  - "my GPU sits at low utilization because the data loader is the bottleneck"
  - "load a folder of images, audio clips or a large CSV into a training loop"
  - "batch variable-length sequences with padding"
  - "stream a dataset that does not fit in memory or on local disk"
  - "make data loading reproducible across runs and workers"
status: draft
last_verified: 2026-10-04
sources:
  - https://pytorch.org/docs/stable/data.html
  - https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
  - https://www.tensorflow.org/guide/data_performance
  - https://huggingface.co/docs/datasets/stream
---

# Data loading pipelines (PyTorch Dataset/DataLoader, tf.data)

## Summary
A training step only runs as fast as batches arrive. The data pipeline reads samples from disk or the network, decodes and augments them, groups them into batches and moves them to the GPU, ideally in parallel with the model's compute. In PyTorch this is a `Dataset` (how to get sample *i*) plus a `DataLoader` (batching, shuffling, worker processes, pinned memory). In TensorFlow it is a `tf.data` chain ending in `prefetch`. When GPU utilization is low and CPU is busy, the loader is usually the bottleneck, not the model.

## Key concepts
- **Map-style `Dataset`.** Implements `__len__` and `__getitem__(i)`. Supports random access, so the `DataLoader` can shuffle. Keep `__init__` light (file paths, not decoded data) and do the heavy work in `__getitem__`.
- **`IterableDataset`.** Implements `__iter__` for streams (shards in object storage, database cursors, generators). No random access: shuffle with a buffer, and split work across workers yourself via `get_worker_info()` or every worker yields the same data.
- **`DataLoader` knobs.**
  - `batch_size`, `shuffle` (or a `sampler`, e.g. `WeightedRandomSampler` for class balance, `DistributedSampler` for multi-GPU).
  - `num_workers`: subprocesses that run `__getitem__` in parallel. 0 means the main process does it all.
  - `pin_memory=True` + `.to(device, non_blocking=True)`: page-locked host memory for faster, asynchronous host-to-GPU copies.
  - `persistent_workers=True`: keep workers alive between epochs instead of re-spawning.
  - `prefetch_factor`: batches each worker prepares ahead (default 2).
  - `collate_fn`: how a list of samples becomes a batch; customise it for padding, dicts or skipping bad samples.
  - `drop_last`: drop the final short batch (useful with BatchNorm or fixed-shape compiled models).
- **Where to augment.** Random CPU augmentations go in `__getitem__` so workers parallelise them. Heavy per-batch ops can run on GPU after transfer.
- **`tf.data` equivalent.** `from_tensor_slices` / `TFRecordDataset` -> `.shuffle(buffer)` -> `.map(fn, num_parallel_calls=AUTOTUNE)` -> `.batch()` -> `.prefetch(AUTOTUNE)`; `.cache()` after expensive deterministic steps.

## When to use / scenarios
- Image classification from folders: map-style dataset reading paths, decoding with PIL/torchvision in workers.
- NLP and speech with variable lengths: custom `collate_fn` that pads per batch; bucket by length to cut padding waste.
- Terabyte-scale web or log data in S3/GCS: `IterableDataset` over shards (WebDataset, Hugging Face `datasets` streaming, MosaicML StreamingDataset).
- Tabular data that fits in memory: skip the `DataLoader` workers entirely; index tensors already on the GPU.
- NOT worth tuning when the GPU is already near 100% busy; profile first.

## Setup & code
```bash
pip install torch
```
Map-style dataset, padding `collate_fn`, and a timing check for the loader alone:
```python
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class Sequences(Dataset):
    def __init__(self, n=1000):
        g = torch.Generator().manual_seed(0)
        self.lengths = torch.randint(5, 50, (n,), generator=g)
    def __len__(self):
        return len(self.lengths)
    def __getitem__(self, i):
        time.sleep(0.005)                      # stand-in for ~5 ms decode/augment cost
        L = int(self.lengths[i])
        return torch.arange(L), torch.tensor(L % 2)

def pad_collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    return pad_sequence(seqs, batch_first=True, padding_value=0), lengths, torch.stack(labels)

if __name__ == "__main__":                     # required for num_workers>0 on Windows/macOS
    for workers in (0, 4):
        dl = DataLoader(Sequences(), batch_size=64, shuffle=True, collate_fn=pad_collate,
                        num_workers=workers, pin_memory=torch.cuda.is_available(),
                        persistent_workers=workers > 0)
        t = time.perf_counter()
        for x, lengths, y in dl:
            pass
        print(f"num_workers={workers}: {time.perf_counter() - t:.2f}s per epoch, last batch {tuple(x.shape)}")
```
Splitting an `IterableDataset` across workers:
```python
from torch.utils.data import IterableDataset, get_worker_info

class Shards(IterableDataset):
    def __init__(self, shards):
        self.shards = shards
    def __iter__(self):
        info = get_worker_info()
        mine = self.shards if info is None else self.shards[info.id::info.num_workers]
        for shard in mine:
            yield from range(shard * 100, shard * 100 + 3)   # stand-in for reading a shard

# list(DataLoader(Shards(list(range(8))), batch_size=None, num_workers=2)) -> each item once
```
The same pipeline in `tf.data`:
```python
import tensorflow as tf
ds = (tf.data.Dataset.from_tensor_slices((features, labels))
      .shuffle(10_000).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
      .batch(64).prefetch(tf.data.AUTOTUNE))
```

## Choosing / trade-offs
- **`num_workers`.** Start at the number of physical cores per GPU (often 4-8) and measure. More workers use more RAM (each holds a copy of the dataset object) and add start-up time.
- **Preprocess once vs on the fly.** Decode, resize and tokenize once to a compact format (NumPy memmap, Arrow/Parquet, TFRecord, WebDataset tar shards) when the same deterministic work repeats every epoch. Keep random augmentation on the fly.
- **Map-style vs iterable.** Map-style gives exact shuffling and easy resuming; iterable scales to remote, unbounded data at the cost of approximate (buffer) shuffling.
- **Many small files vs shards.** Millions of small files are slow on network filesystems and object storage; pack them into shards of ~100 MB-1 GB.

## Gotchas
- On Windows and macOS workers start with `spawn`: the loader must be created under `if __name__ == "__main__":`, and the dataset and `collate_fn` must be picklable (no lambdas or local classes).
- Older PyTorch versions gave every worker the same NumPy random seed, so "random" augmentations repeated across workers; other libraries' RNGs (`random` in custom code, third-party augmenters) can still do this. Seed them in `worker_init_fn` from `torch.initial_seed()`. Pass a seeded `generator=` to the `DataLoader` for reproducible shuffling.
- An `IterableDataset` that ignores `get_worker_info()` yields every sample `num_workers` times.
- Holding a Python list of millions of objects in the dataset causes copy-on-write memory growth in each worker; store data in NumPy arrays or Arrow tables instead.
- `pin_memory` helps only with `non_blocking=True` transfers and costs host RAM; it does nothing on CPU-only training.
- With `DistributedSampler`, call `sampler.set_epoch(epoch)` each epoch or every epoch uses the same order ([[distributed-training]]).
- Opening file handles or database connections in `__init__` breaks after forking; open them lazily in `__getitem__` / `__iter__`.
- Measure before tuning: time one epoch of the loader with no model. If that already matches the training time, the loader is the bottleneck.

## Related
- [[pytorch-basics]] - tensors, modules and the training loop the loader feeds.
- [[data-augmentation]] - transforms that run inside `__getitem__`.
- [[efficient-training-mixed-precision]] - the compute side once data arrives fast enough.
- [[distributed-training]] - samplers and sharding across GPUs.
- [[keras-and-tensorflow]] - `tf.data` in the Keras training loop.
- [[sequence-to-sequence-and-ctc]] - padding and masking variable-length batches.

## References
- PyTorch, torch.utils.data: https://pytorch.org/docs/stable/data.html
- PyTorch tutorial, Datasets & DataLoaders: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- TensorFlow, Better performance with the tf.data API: https://www.tensorflow.org/guide/data_performance
- Hugging Face Datasets, Stream: https://huggingface.co/docs/datasets/stream
