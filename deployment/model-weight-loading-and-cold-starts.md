---
title: Model weight loading and cold starts (safetensors, HF cache, Run:ai streamer, vLLM sleep mode)
category: deployment
tags: [cold-start, model-loading, safetensors, huggingface-cache, hf-xet, runai-model-streamer, tensorizer, fastsafetensors, vllm, sleep-mode, torch-compile-cache, scale-to-zero, ollama, keep-alive, autoscaling]
use_cases:
  - "cut the minutes a new vLLM replica takes before it can serve its first request"
  - "load model weights straight from S3, GCS or Azure Blob into GPU memory without a local copy"
  - "scale an LLM deployment to zero overnight and bring it back quickly in the morning"
  - "free GPU memory between jobs without restarting the inference server"
  - "stop every pod from re-downloading the same model from Hugging Face"
status: draft
last_verified: 2026-10-24
sources:
  - https://github.com/huggingface/safetensors/blob/main/README.md
  - https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/package_reference/environment_variables.md
  - https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/guides/download.md
  - https://github.com/vllm-project/vllm/blob/main/docs/models/extensions/runai_model_streamer.md
  - https://github.com/vllm-project/vllm/blob/main/docs/models/extensions/tensorizer.md
  - https://github.com/vllm-project/vllm/blob/main/docs/models/extensions/fastsafetensor.md
  - https://github.com/vllm-project/vllm/blob/main/docs/features/sleep_mode.md
  - https://github.com/vllm-project/vllm/blob/main/docs/design/torch_compile.md
  - https://github.com/ollama/ollama/blob/main/docs/faq.mdx
---

# Model weight loading and cold starts (safetensors, HF cache, Run:ai streamer, vLLM sleep mode)

## Summary
A self-hosted LLM replica is not ready until its weights have been downloaded, read from disk, copied to GPU memory, and the engine has compiled kernels and captured CUDA graphs. For a large model that start-up takes minutes, which makes autoscaling slow, scale-to-zero painful and rollouts long. This topic covers the parts you control: the weight format (safetensors), where the weights live (Hugging Face cache, a shared volume, or object storage streamed directly with the Run:ai Model Streamer or Tensorizer), the engine caches you can persist (vLLM's `torch.compile` cache), and how to avoid a cold start altogether by keeping the process alive and only releasing GPU memory (vLLM sleep mode, Ollama `keep_alive`).

## Key concepts
- **Cold-start phases**: (1) fetch weights to the node, (2) read them into host memory, (3) copy to GPU, (4) engine warm-up: `torch.compile`, CUDA graph capture, KV-cache allocation. Measure each from the server log before optimizing; the slowest phase differs per setup.
- **safetensors**: the format most open checkpoints ship in. Per its README: an 8-byte little-endian header length, a JSON header mapping each tensor name to `dtype`, `shape` and `data_offsets`, then one contiguous byte buffer. It is safe (no pickle code execution, unlike `.bin`/`.pt` files) and zero-copy, so a loader can memory-map or stream tensors without parsing the whole file. Large models are split into shards listed in `model.safetensors.index.json`.
- **Hugging Face cache**: `huggingface_hub` downloads into `HF_HUB_CACHE` (default `$HF_HOME/hub`, i.e. `~/.cache/huggingface/hub`). Even with a warm cache, a download call makes an HTTP request to check for a newer version; `HF_HUB_OFFLINE=1` skips it and makes no Hub calls at all. Transfers go through `hf-xet`; `HF_XET_HIGH_PERFORMANCE=1` tries to saturate network and CPU (the older `HF_HUB_ENABLE_HF_TRANSFER` is deprecated).
- **Streaming loaders in vLLM** (`--load-format`):
  - `runai_streamer`: the Run:ai Model Streamer reads safetensors tensors concurrently and streams them into GPU memory, from a local path or directly from `s3://`, `gs://` or `az://` URIs. Tunables via `--model-loader-extra-config`: `concurrency`, `memory_limit` (CPU buffer cap), and `distributed` (CUDA and ROCm only). `runai_streamer_sharded` loads pre-sharded per-rank files so each tensor-parallel worker reads only its own shard.
  - `tensorizer`: CoreWeave's Tensorizer deserializes a model you first serialized with vLLM's example script, from disk, HTTP(S) or S3, straight to the GPU; supports tensor encryption.
  - `fastsafetensors`: loads safetensors to GPU memory through GPU Direct Storage.
- **vLLM `torch.compile` cache**: compiled artifacts land in `~/.cache/vllm/torch_compile_cache`; the vLLM design doc says you can copy that directory into your deployment to skip recompilation. `VLLM_DISABLE_COMPILE_CACHE=1` disables it for debugging. `enforce_eager=True` skips CUDA graph capture (faster start, usually slower decoding).
- **vLLM sleep mode**: keeps the server process alive but releases GPU memory. Level 1 offloads weights to CPU RAM and discards the KV cache (wake up to the same model); level 2 discards weights too (wake up then reload or swap weights, as in RLHF). Enable with `--enable-sleep-mode`; the HTTP endpoints `/sleep`, `/wake_up`, `/is_sleeping` exist only with `VLLM_SERVER_DEV_MODE=1` and must not be exposed to users. Supported on CUDA and ROCm.
- **Ollama keep-alive**: Ollama unloads a model after 5 minutes idle by default; `keep_alive` per request (`-1` keeps it loaded, `0` unloads now) or `OLLAMA_KEEP_ALIVE` server-wide; an empty request preloads a model.

## When to use / scenarios
- Autoscaling a vLLM or SGLang deployment on queue depth ([[kubernetes-for-llm-serving]]): new replicas arrive after the traffic spike unless load time is cut.
- Internal assistants with office-hours traffic: scale to zero (or sleep) at night, wake in the morning without a long wait.
- A GPU shared between an inference server and batch or training jobs: sleep the server instead of killing it ([[batch-inference-and-batch-apis]]).
- RL post-training loops that update the policy weights inside a running vLLM engine (level-2 sleep plus weight reload) ([[rl-with-verifiable-rewards]]).
- Air-gapped or rate-limited clusters where every pod pulling from the Hub is not allowed or too slow.
- Not worth the effort for a single always-on replica that restarts rarely; a warm local disk and `HF_HUB_OFFLINE=1` are enough. Hosted APIs have no cold start for you to manage ([[model-selection]]).

## Setup & code
Download once, then run offline from a shared cache (paths and model ID are placeholders):
```bash
export HF_HOME=/models/hf            # a PVC, NFS share or node-local NVMe mounted in every pod
hf download <org>/<model> --include "*.safetensors" "*.json" "tokenizer*"   # skip .bin/.pt duplicates
HF_HUB_OFFLINE=1 vllm serve <org>/<model>    # no Hub request at start-up
```
Stream weights from object storage into GPU memory (from vLLM's Run:ai Model Streamer docs):
```bash
pip install "vllm[runai]"
vllm serve s3://my-bucket/models/<model> --load-format runai_streamer \
  --model-loader-extra-config '{"concurrency":16}'
# S3-compatible stores (MinIO, GCS via S3 API): set AWS_ENDPOINT_URL and RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0
```
Keep the process and free the GPU (vLLM sleep mode, from its docs):
```bash
VLLM_SERVER_DEV_MODE=1 vllm serve <model> --enable-sleep-mode --port 8000   # dev endpoints: keep them off the public network
curl -X POST 'http://localhost:8000/sleep?level=1'   # weights to CPU RAM, KV cache dropped
curl -X POST 'http://localhost:8000/wake_up'
curl http://localhost:8000/is_sleeping
```
Ollama: preload and pin, or unload immediately:
```bash
curl http://localhost:11434/api/generate -d '{"model": "<model>", "keep_alive": -1}'
curl http://localhost:11434/api/generate -d '{"model": "<model>", "keep_alive": 0}'
```
Runnable, stdlib only: read safetensors headers without loading tensors (the same trick streaming loaders use), list what a checkpoint holds, and estimate each cold-start phase from bandwidths you measure on your own hardware.
```python
import json
import struct
import tempfile
from pathlib import Path

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
               "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}

def write_safetensors(path, tensors, metadata=None):
    """tensors: {name: (dtype, shape)}; writes zero-filled data in the documented layout."""
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = DTYPE_BYTES[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)                       # header may be padded with spaces
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        f.write(b"\0" * offset)

def read_header(path):
    """Parse only the header: 8-byte little-endian length, then JSON. No tensor bytes are read."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    return header, meta

def summarize(paths):
    total, by_dtype, count = 0, {}, 0
    for p in paths:
        header, _ = read_header(p)
        for name, t in header.items():
            begin, end = t["data_offsets"]
            total += end - begin
            by_dtype[t["dtype"]] = by_dtype.get(t["dtype"], 0) + (end - begin)
            count += 1
    return total, by_dtype, count

def estimate_seconds(nbytes, download_bps=None, disk_bps=None, h2d_bps=None, warmup_s=0.0):
    """Each phase = bytes / bandwidth you measured. None = phase skipped (e.g. weights already on disk)."""
    phases = {}
    for name, bps in (("download", download_bps), ("disk_read", disk_bps), ("host_to_gpu", h2d_bps)):
        if bps:
            phases[name] = nbytes / bps
    phases["engine_warmup"] = warmup_s
    return phases

with tempfile.TemporaryDirectory() as d:
    shards = []
    for i in range(2):                                   # a tiny two-shard "checkpoint"
        p = Path(d) / f"model-0000{i + 1}-of-00002.safetensors"
        write_safetensors(p, {f"layers.{i}.attn.w": ("BF16", (256, 256)),
                              f"layers.{i}.mlp.w": ("BF16", (256, 1024))}, {"format": "pt"})
        shards.append(p)
    total, by_dtype, count = summarize(shards)
    print(f"{count} tensors, {total} bytes of weights, by dtype {by_dtype}")
    print("header of shard 1:", sorted(read_header(shards[0])[0]))

    # Plug in your own measurements; these placeholders only show the arithmetic.
    weights = 16 * 10**9                                  # e.g. bytes reported by summarize() on a real checkpoint
    scenarios = {
        "cold node, pull from Hub": estimate_seconds(weights, download_bps=0.5e9, disk_bps=2e9, h2d_bps=10e9, warmup_s=60),
        "warm local cache":         estimate_seconds(weights, disk_bps=2e9, h2d_bps=10e9, warmup_s=60),
        "warm cache + compile cache": estimate_seconds(weights, disk_bps=2e9, h2d_bps=10e9, warmup_s=15),
    }
    for name, ph in scenarios.items():
        print(f"{name:28s} total {sum(ph.values()):6.1f}s  " + ", ".join(f"{k}={v:.1f}s" for k, v in ph.items()))
```

## Choosing / trade-offs
| Situation | Approach | Cost |
|---|---|---|
| One replica, restarts rarely | Local disk cache, `HF_HUB_OFFLINE=1` | Nothing extra |
| Many pods, same model | Shared cache volume or node-local cache filled once (init job or DaemonSet) | Volume to manage; network filesystems can be slower than local NVMe |
| Weights in object storage, frequent scale-out | `--load-format runai_streamer` from `s3://`/`gs://`/`az://` | Extra dependency; tune `concurrency`/`memory_limit`; storage egress |
| Very large tensor-parallel model | `runai_streamer_sharded` with pre-sharded files | One-off resharding step per TP layout |
| Pod start dominated by compile/graph capture | Persist and ship `~/.cache/vllm/torch_compile_cache` | Cache must match model, config and vLLM version |
| GPU shared with other jobs, model returns soon | vLLM sleep level 1 | Host RAM must hold the weights |
| Swap or update weights in a live engine | vLLM sleep level 2 plus `reload_weights` | Weights must be reloaded; dev-mode endpoints |
| Local desktop or small server | Ollama `keep_alive` / `OLLAMA_KEEP_ALIVE` | Idle model holds VRAM |

- **Scale to zero vs a warm minimum**: zero replicas saves GPU hours but the first user waits for the whole cold start; one warm replica (or one sleeping engine) costs GPU time but answers at once. Decide from the latency SLO and traffic shape ([[cost-and-latency]]).
- **Smaller weights load faster**: an FP8 or 4-bit checkpoint has fewer bytes to move, on top of fitting more KV cache ([[quantizing-llms-for-serving]]).
- **Many fine-tunes**: serve LoRA adapters on one warm base model instead of cold-starting one deployment per fine-tune ([[multi-lora-adapter-serving]]).

## Gotchas
- `.bin`/`.pt` checkpoints are pickles and can execute code when loaded; prefer `.safetensors` and filter downloads with `--include`/`allow_patterns` so you do not pull both formats.
- A warm Hugging Face cache still does a network round-trip per file unless `HF_HUB_OFFLINE=1` is set; in an air-gapped cluster that check can hang until a timeout.
- `hf download --local-dir` creates its own `.cache/huggingface/` metadata folder; mixing `local_dir` copies and the shared cache duplicates multi-GB files.
- `HF_HUB_ENABLE_HF_TRANSFER=1` no longer does anything useful; the Hub now transfers through `hf-xet` (`HF_XET_HIGH_PERFORMANCE=1` for maximum throughput).
- Run:ai streamer distributed mode works on CUDA and ROCm only; `memory_limit` caps the CPU buffer, and setting it too low slows loading.
- Tensorizer needs a serialization step first; a checkpoint serialized with one vLLM version or config is not guaranteed to load in another.
- Sleep level 1 needs enough CPU RAM for the full weights; on a node running several engines that can trigger the OOM killer. The sleep endpoints require `VLLM_SERVER_DEV_MODE=1`, which also exposes other development endpoints.
- Kubernetes readiness probes that fire before weights load mark the pod as failed and restart it in a loop; set a startup probe long enough for the measured cold start ([[kubernetes-for-llm-serving]]).
- A copied `torch.compile` cache is only reused when the factors vLLM hashes match (model, config, code); after an upgrade expect one slow start.

## Related
- [[inference-servers-vllm]] - the engine whose start-up this topic shortens.
- [[vllm-local-setup]] - vLLM install and first-start errors.
- [[kubernetes-for-llm-serving]] - autoscaling, probes and caches on a cluster.
- [[docker-for-ml]] - model cache volumes instead of baking weights into images.
- [[huggingface-transformers]] - the Hub, cache and download basics.
- [[quantizing-llms-for-serving]] - fewer bytes to load per replica.
- [[multi-lora-adapter-serving]] - one warm base model for many fine-tunes.
- [[ollama]] - `keep_alive` and model preloading on a local runner.
- [[gpu-cloud-options]] - serverless GPU offers where cold start is the main cost.
- [[llm-load-testing-and-benchmarking]] - measure time-to-ready alongside TTFT.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/capacity-planning-and-autoscaling.md - scaling policies that must allow for replica start-up time.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/health-checks-and-graceful-shutdown.md - startup vs readiness probes for slow-starting services.

## References
- safetensors format and safety rationale (README): https://github.com/huggingface/safetensors/blob/main/README.md
- huggingface_hub environment variables (`HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE`, `HF_XET_HIGH_PERFORMANCE`, deprecated `HF_HUB_ENABLE_HF_TRANSFER`): https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/package_reference/environment_variables.md
- huggingface_hub download guide (`hf download`, `allow_patterns`, `local_dir`): https://github.com/huggingface/huggingface_hub/blob/main/docs/source/en/guides/download.md
- vLLM, Loading models with Run:ai Model Streamer: https://github.com/vllm-project/vllm/blob/main/docs/models/extensions/runai_model_streamer.md
- vLLM, Loading models with CoreWeave's Tensorizer: https://github.com/vllm-project/vllm/blob/main/docs/models/extensions/tensorizer.md
- vLLM, Loading model weights with fastsafetensors: https://github.com/vllm-project/vllm/blob/main/docs/models/extensions/fastsafetensor.md
- vLLM Sleep Mode: https://github.com/vllm-project/vllm/blob/main/docs/features/sleep_mode.md
- vLLM torch.compile integration (compile cache directory): https://github.com/vllm-project/vllm/blob/main/docs/design/torch_compile.md
- Ollama FAQ (preloading, `keep_alive`, `OLLAMA_KEEP_ALIVE`): https://github.com/ollama/ollama/blob/main/docs/faq.mdx
