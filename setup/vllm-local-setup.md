---
title: vLLM local setup (Linux, WSL2, Docker) and common errors
category: setup
tags: [vllm, local-setup, wsl2, docker, openai-compatible, cuda, kv-cache, troubleshooting]
use_cases:
  - "install vLLM on my Linux GPU box or under WSL2 and serve a model locally"
  - "run vLLM in Docker with my Hugging Face cache mounted"
  - "fix vLLM failing at start-up with a KV cache or CUDA out-of-memory error"
  - "check whether a model and context length fit my GPU before downloading it"
status: draft
last_verified: 2026-10-15
sources:
  - https://docs.vllm.ai/en/latest/getting_started/quickstart/
  - https://docs.vllm.ai/en/latest/getting_started/installation/
  - https://docs.vllm.ai/en/latest/deployment/docker.html
  - https://docs.vllm.ai/en/latest/usage/troubleshooting.html
  - https://docs.vllm.ai/en/latest/configuration/optimization/
  - https://learn.microsoft.com/en-us/windows/wsl/install
  - https://docs.nvidia.com/cuda/wsl-user-guide/index.html
---

# vLLM local setup (Linux, WSL2, Docker) and common errors

## Summary
This file is the hands-on path to a working vLLM server on one machine: install it on Linux or WSL2 (or run the official Docker image), start `vllm serve`, call it with the OpenAI SDK, and fix the start-up errors people hit first. What vLLM is, why it is fast, and how it compares with SGLang/TGI live in [[inference-servers-vllm]]; this file does not repeat them.

## Key concepts
- Platform: vLLM targets Linux with NVIDIA GPUs first (other accelerators have separate install pages). There is no native Windows build; use WSL2 with the Windows NVIDIA driver, or Docker Desktop's WSL2 backend.
- Wheel and PyTorch pairing: a vLLM release pins a PyTorch/CUDA combination. Install it into a fresh environment (`uv pip install vllm --torch-backend=auto` per the quickstart) rather than on top of an existing torch.
- Memory budget at start-up: vLLM claims `--gpu-memory-utilization` of the GPU (a fraction), loads the weights, then gives what is left to the KV cache. It refuses to start if the KV cache cannot hold one sequence of `--max-model-len` tokens.
- KV cache per token = 2 (K and V) x layers x KV heads x head dim x bytes per element. Grouped-query attention (fewer KV heads) and an FP8 KV cache shrink it ([[kv-cache-from-scratch-numpy]]).
- Model access: gated Hugging Face models need `HF_TOKEN` and an accepted licence; weights are cached under the Hugging Face cache ([[huggingface-transformers]]).

## When to use / scenarios
- A developer with an RTX workstation wants an OpenAI-compatible endpoint that handles several concurrent requests better than desktop runners.
- A team validates a fine-tuned model locally before deploying the same image to a GPU cloud ([[gpu-cloud-options]]).
- A Windows user with an NVIDIA GPU wants vLLM: WSL2 route.
- Not for: CPU-only laptops or Apple Silicon (use [[ollama]], [[llama-cpp-gguf]] or [[lm-studio-and-jan]]); GPUs too small for the model even quantized ([[quantization]]).

## Setup & code
Native Linux or WSL2 (inside the Ubuntu distro; install only the Windows NVIDIA driver, not a Linux driver, per NVIDIA's WSL guide):
```bash
nvidia-smi                                   # GPU visible?
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install vllm --torch-backend=auto
export HF_TOKEN=<your token>                 # only for gated models
vllm serve <hf-org>/<model> --max-model-len 8192 --gpu-memory-utilization 0.90
```
Docker (needs the NVIDIA Container Toolkit, see [[docker-for-ml]]; pin a tag instead of `latest`):
```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN -p 127.0.0.1:8000:8000 --ipc=host \
  vllm/vllm-openai:<tag> --model <hf-org>/<model> --max-model-len 8192
```
Smoke test and client:
```bash
curl http://localhost:8000/v1/models
```
<!-- skip-check: needs a running vLLM server and the openai package -->
```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")   # or the --api-key you set
r = c.chat.completions.create(model="<hf-org>/<model>",
                              messages=[{"role": "user", "content": "Reply with OK"}])
print(r.choices[0].message.content)
```
Will it fit? A runnable estimator (stdlib only). Read the numbers from the model's `config.json` (`num_hidden_layers`, `num_key_value_heads`, `head_dim` or `hidden_size / num_attention_heads`); the config below is illustrative, not a specific model.
```python
GiB = 1024 ** 3

def kv_bytes_per_token(layers, kv_heads, head_dim, kv_dtype_bytes=2):
    return 2 * layers * kv_heads * head_dim * kv_dtype_bytes   # K and V

def plan(gpu_gib, util, params_b, weight_bytes, layers, kv_heads, head_dim,
         max_model_len, kv_dtype_bytes=2, overhead_gib=1.5):
    budget = gpu_gib * util
    weights = params_b * 1e9 * weight_bytes / GiB
    kv_free = budget - weights - overhead_gib            # activations, CUDA graphs, etc. (rough)
    per_tok = kv_bytes_per_token(layers, kv_heads, head_dim, kv_dtype_bytes)
    tokens = max(0, int(kv_free * GiB // per_tok))
    return {"weights_gib": round(weights, 1), "kv_gib": round(kv_free, 1),
            "kv_tokens": tokens, "fits_one_seq": tokens >= max_model_len,
            "max_len_that_fits": tokens, "full_len_seqs": tokens // max_model_len}

cfg = dict(layers=32, kv_heads=8, head_dim=128)           # illustrative GQA config
print("KiB per token:", kv_bytes_per_token(**cfg) / 1024)
for weight_bytes, label in [(2, "bf16 weights"), (0.5, "4-bit weights")]:
    p = plan(gpu_gib=24, util=0.90, params_b=8, weight_bytes=weight_bytes,
             max_model_len=32768, **cfg)
    print(label, p)
assert kv_bytes_per_token(**cfg) == 131072
```
If `fits_one_seq` is False you will get the KV cache start-up error below; lower `--max-model-len`, use a quantized checkpoint, or add GPUs.

## Choosing / trade-offs
- Native venv vs Docker: venv is easier to debug and to mix with your own code; the Docker image pins vLLM, PyTorch and CUDA together and matches production ([[docker-for-ml]]).
- WSL2 vs dual-boot Linux: WSL2 is convenient; native Linux avoids WSL-specific memory limits and filesystem slowness when reading weights from `/mnt/c`.
- Context length vs concurrency: a smaller `--max-model-len` leaves room for more simultaneous sequences.
- Quantized checkpoint (AWQ/GPTQ/FP8) vs full precision: fits a larger model on one GPU at some quality cost; check the format is supported by your vLLM version.

## Gotchas
Common errors and fixes (wording varies by version; confirm with the troubleshooting page and `vllm serve --help`):

| Symptom | Likely cause | Fix |
|---|---|---|
| "max seq len ... is larger than the maximum number of tokens that can be stored in KV cache" / "KV cache is needed, which is larger than the available KV cache memory" | Model's default context is longer than the memory left after weights | Lower `--max-model-len`, raise `--gpu-memory-utilization` a little, quantize, or use tensor parallelism |
| CUDA out of memory during warm-up or CUDA graph capture | Another process holds VRAM, or batch limits too high | Free the GPU (`nvidia-smi`), lower `--max-num-seqs`, try `--enforce-eager` |
| `401`/`403` or "gated repo" when downloading | Missing `HF_TOKEN` or licence not accepted on the Hub | Accept the licence, export `HF_TOKEN` (pass `-e HF_TOKEN` to Docker) |
| `torch.cuda.is_available()` False in WSL2 | Linux NVIDIA driver installed inside WSL, or old Windows driver | Remove the in-distro driver, update the Windows driver ([[windows-wsl-setup]]) |
| Import errors / undefined symbols after install | vLLM installed over a different torch build | Fresh venv, reinstall vLLM so it brings its own torch |
| Bus error or crash with many workers in Docker | Container shared memory too small | Add `--ipc=host` or a larger `--shm-size` |
| Garbled or never-ending output | Chat template or stop tokens missing for a base/custom model | Use the instruct model, or pass `--chat-template` |
| Slow first start | Weight download and compilation/graph capture | Pre-download into the mounted cache; health-check `/v1/models` before sending traffic |

- Check which interface the server listens on (`--host`) and assume it is reachable from the network unless you bound it to `127.0.0.1`; set `--api-key`, publish Docker ports on `127.0.0.1`, or firewall the port.
- Running several vLLM processes on one GPU needs each `--gpu-memory-utilization` to add up below 1.0.

## Related
- [[inference-servers-vllm]] - what vLLM is, production flags and alternatives; read it before scaling out.
- [[docker-for-ml]] - GPU containers and model cache volumes.
- [[windows-wsl-setup]] - WSL2 and the GPU on Windows.
- [[gpu-cuda-setup]] - driver/CUDA checks and VRAM sizing rules.
- [[kv-cache-from-scratch-numpy]] - why the KV cache dominates memory.
- [[quantization]] - fitting bigger models.
- [[lm-studio-and-jan]] - simpler desktop route for single users.
- [[ollama]] - CPU/Apple-friendly local server.
- [[serving-with-fastapi]] - the app layer in front of vLLM.

## References
- vLLM quickstart: https://docs.vllm.ai/en/latest/getting_started/quickstart/
- vLLM installation: https://docs.vllm.ai/en/latest/getting_started/installation/
- vLLM Docker: https://docs.vllm.ai/en/latest/deployment/docker.html
- vLLM troubleshooting: https://docs.vllm.ai/en/latest/usage/troubleshooting.html
- vLLM optimization and tuning: https://docs.vllm.ai/en/latest/configuration/optimization/
- WSL install: https://learn.microsoft.com/en-us/windows/wsl/install
- CUDA on WSL: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
