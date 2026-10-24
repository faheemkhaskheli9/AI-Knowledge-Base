---
title: AMD GPUs with ROCm (PyTorch, vLLM, llama.cpp, Ollama)
category: setup
tags: [amd, rocm, hip, radeon, instinct, mi300x, gfx, pytorch, vllm, llama-cpp, ollama, vulkan, docker, hsa-override]
use_cases:
  - "get PyTorch to see my AMD Radeon or Instinct GPU on Linux"
  - "serve an open model with vLLM on AMD MI300X or a Radeon RX 7900"
  - "run llama.cpp or Ollama on an AMD GPU that ROCm does not officially list"
  - "fix vLLM failing with libcudart.so errors on an AMD machine"
status: draft
last_verified: 2026-10-24
sources:
  - https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
  - https://pytorch.org/docs/stable/notes/hip.html
  - https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
  - https://github.com/ollama/ollama/blob/main/docs/gpu.mdx
  - https://github.com/ollama/ollama/blob/main/docs/docker.mdx
---

# AMD GPUs with ROCm (PyTorch, vLLM, llama.cpp, Ollama)

## Summary
ROCm is AMD's open GPU compute stack and HIP is its CUDA-like C++ dialect; PyTorch, vLLM, llama.cpp and Ollama all run on AMD GPUs through it. The pieces that differ from NVIDIA are the driver/ROCm install, the wheel or image you pick (ROCm builds are separate from CUDA builds), and the GPU's LLVM target name (`gfx942` for MI300, `gfx1100` for Radeon RX 7900 and so on), which decides whether prebuilt kernels exist for your card. Code barely changes: PyTorch on ROCm deliberately reuses the `torch.cuda` API, so `device="cuda"` works on an AMD GPU. Officially supported targets work out of the box; other consumer cards often run llama.cpp and Ollama with an `HSA_OVERRIDE_GFX_VERSION` override or through Vulkan instead.

## Key concepts
- **ROCm and HIP**: ROCm is the driver plus runtime and math libraries (rocBLAS, hipBLASLt, MIOpen); HIP is the programming layer that ports CUDA code. Install on Linux from AMD's install guide (`amdgpu-install` or distro packages).
- **LLVM target (`gfx...`)**: every AMD GPU has one, shown by `rocminfo`. Prebuilt kernels exist only for listed targets. vLLM's docs list MI200 (`gfx90a`), MI300 (`gfx942`), MI350 (`gfx950`), Radeon RX 7900 (`gfx1100/1101`), Radeon RX 9000 (`gfx1200/1201`) and Ryzen AI MAX / AI 300 (`gfx1151/1150`).
- **`torch.cuda` on ROCm**: the PyTorch HIP notes state that HIP reuses `torch.cuda` interfaces; `torch.device("cuda")` is the AMD GPU ("rocm" or "hip" are not valid device names). `torch.version.hip` is set on ROCm builds and `torch.version.cuda` on CUDA builds.
- **Separate builds**: a CUDA wheel installed on an AMD box imports fine and then fails or sees no GPU. Every tool has its own ROCm channel: the PyTorch ROCm index URL from pytorch.org, vLLM's ROCm wheel index or `vllm/vllm-openai-rocm` images, llama.cpp built with `-DGGML_HIP=ON`, Ollama's ROCm build or `ollama/ollama:rocm` image.
- **Override for unlisted cards**: `HSA_OVERRIDE_GFX_VERSION=x.y.z` makes the runtime treat the GPU as a nearby supported target (llama.cpp docs: `10.3.0` for RDNA2 such as gfx1030/1031/1035, `11.0.0` for RDNA3). Unsupported, sometimes unstable, and not available on Windows.
- **Vulkan backend**: llama.cpp and Ollama also run on AMD GPUs through Vulkan, which covers many cards ROCm does not list and works on Windows with standard drivers.
- **Device selection and monitoring**: `HIP_VISIBLE_DEVICES` (llama.cpp, PyTorch) or `ROCR_VISIBLE_DEVICES` (Ollama) pick GPUs; `rocminfo` lists agents and targets; `amd-smi` shows memory and utilization (the PyTorch notes point out cached allocator memory shows as used there).

## When to use / scenarios
- An on-prem server or cloud instance with AMD Instinct (MI300X and later) for serving open-weight models with vLLM ([[inference-servers-vllm]]); large per-GPU memory lets bigger models fit on fewer GPUs.
- A workstation with a Radeon RX 7900 or RX 9000 card for local LLMs (Ollama, llama.cpp) or PyTorch experiments.
- A Ryzen AI MAX / AI 300 laptop or mini-PC with a large shared memory pool for local inference ([[small-language-models]]).
- An older or unlisted Radeon: try llama.cpp/Ollama with Vulkan or the gfx override; do not plan training or vLLM serving on it.
- Not when your stack depends on CUDA-only pieces (some custom kernels, TensorRT-LLM, NVIDIA-only quantization formats); check each library's ROCm support first, or use NVIDIA ([[gpu-cuda-setup]], [[gpu-cloud-options]]).

## Setup & code
Linux driver and ROCm: follow AMD's install guide for your distro, add your user to the `render` and `video` groups as it describes, reboot, then check:
```bash
rocminfo | grep -m1 -o 'gfx[0-9a-f]*'    # your LLVM target, e.g. gfx1100
amd-smi                                   # GPUs, memory, utilization
```
PyTorch: copy the ROCm install line from the selector on pytorch.org (it is a separate `--index-url` ending in `rocm<version>`), or use a ROCm PyTorch container. Then the usual CUDA-style code runs:
<!-- skip-check: needs a ROCm build of PyTorch and an AMD GPU -->
```python
import torch

print("torch", torch.__version__, "hip", torch.version.hip, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "no GPU: CUDA wheel on an AMD box, missing driver, or user not in render/video groups"
print(torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)   # "cuda" is the AMD GPU here
print((x @ x).float().abs().mean().item())
```
vLLM (from the vLLM ROCm install docs): ROCm wheels are Python 3.12 only, and on another Python version the installer silently falls back to the CUDA wheel, which then fails with `libcudart.so` errors. The vLLM wheel bundles its own PyTorch; use that one.
```bash
uv venv --python 3.12 --seed --managed-python && source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/ --upgrade
vllm serve <model>
# Or the official ROCm image (tags: latest, nightly, and legacy-ROCm variants; see Docker Hub)
docker run --rm --group-add=video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/dri --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN" \
  vllm/vllm-openai-rocm:<tag> --model <model>
```
llama.cpp (from its build docs) and Ollama (from its GPU and Docker docs):
```bash
# llama.cpp with HIP; GPU_TARGETS is optional (omit to build for the GPUs in this machine)
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --config Release -- -j 16
# Unlisted RDNA2 card: pretend to be gfx1030 (Linux only, unsupported)
HSA_OVERRIDE_GFX_VERSION=10.3.0 ./build/bin/llama-server -m model.gguf -ngl 99
# Ollama in Docker on an AMD GPU
docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:rocm
```
Runnable, stdlib only: a preflight script that finds `rocminfo`, reads the GPU's LLVM target, and says which route fits (vLLM-listed target, llama.cpp/Ollama override, or Vulkan). The sample text stands in for `rocminfo` output when ROCm is not installed.
```python
import re
import shutil
import subprocess

VLLM_TARGETS = {"gfx90a": "MI200", "gfx942": "MI300", "gfx950": "MI350", "gfx1100": "RX 7900",
                "gfx1101": "RX 7900 series", "gfx1200": "RX 9000", "gfx1201": "RX 9000",
                "gfx1150": "Ryzen AI 300", "gfx1151": "Ryzen AI MAX"}       # vLLM ROCm requirements list
OVERRIDE = {"gfx103": "10.3.0", "gfx110": "11.0.0"}   # llama.cpp docs: RDNA2 -> 10.3.0, RDNA3 -> 11.0.0

SAMPLE_ROCMINFO = """
*******
Agent 2
*******
  Name:                    gfx1031
  Marketing Name:          AMD Radeon RX 6700 XT
  Device Type:             GPU
"""

def gpu_targets(text):
    return sorted(set(re.findall(r"Name:\s+(gfx[0-9a-f]+)", text)))

def advise(target):
    if target in VLLM_TARGETS:
        return f"{target} ({VLLM_TARGETS[target]}): listed by vLLM; prebuilt ROCm wheels and images should work"
    prefix = target[:6]
    if prefix in OVERRIDE:
        return (f"{target}: not in vLLM's list; for llama.cpp/Ollama on Linux try "
                f"HSA_OVERRIDE_GFX_VERSION={OVERRIDE[prefix]} (unsupported), or use the Vulkan backend")
    return f"{target}: no known override; use llama.cpp/Ollama with Vulkan, or CPU"

if shutil.which("rocminfo"):
    out = subprocess.run(["rocminfo"], capture_output=True, text=True).stdout
    source = "rocminfo"
else:
    out, source = SAMPLE_ROCMINFO, "sample output (rocminfo not found: install ROCm first)"
print("source:", source)
targets = gpu_targets(out)
print("GPU targets:", targets or "none found (driver missing, or user not in render/video groups)")
for t in targets:
    print(" -", advise(t))
for t in ["gfx942", "gfx1100", "gfx1031", "gfx906"]:
    print("example", advise(t))
```

## Choosing / trade-offs
| Goal | AMD route | Watch out for |
|---|---|---|
| Serve many users | vLLM ROCm image on Instinct or listed Radeon | Python 3.12 wheels only; check feature support per release |
| Local chat on a desktop | Ollama (ROCm or Vulkan) or llama.cpp | unlisted cards need override or Vulkan |
| Training / fine-tuning | ROCm PyTorch container, listed target | CUDA-only extensions; check library support |
| Windows laptop or desktop | Ollama (ROCm on its listed cards, else Vulkan), llama.cpp HIP or Vulkan build | `HSA_OVERRIDE_GFX_VERSION` does not work on Windows; WSL2 support depends on the ROCm release |

- **ROCm vs Vulkan in llama.cpp/Ollama**: ROCm (HIP) is the vendor path for listed GPUs; Vulkan reaches more cards and Windows without ROCm. Benchmark both on your card and model ([[llm-load-testing-and-benchmarking]]).
- **Cost and availability**: AMD Instinct instances exist on several GPU clouds; compare cost per served token, not list price ([[gpu-cloud-options]], [[cost-and-latency]]).
- **Prebuilt vs source build**: prebuilt vLLM wheels compile kernels for a validated ROCm and PyTorch pair and may not match other builds; a different ROCm or an existing PyTorch means building vLLM from source with `PYTORCH_ROCM_ARCH` set to your target.

## Gotchas
- Installing `vllm` or `torch` from default PyPI on an AMD machine gets the CUDA build. Symptoms: `libcudart.so: cannot open shared object file`, or `torch.cuda.is_available()` is False with `torch.version.hip` None.
- With `pip`, extra indexes are merged with PyPI and the newest version wins, which can pull the CUDA wheel; vLLM's docs recommend `uv`, which gives the extra index priority, or pinning the exact `+rocm` version.
- Containers need `--device /dev/kfd --device /dev/dri` (and the `video` group); on SELinux hosts Ollama's docs say to run `sudo setsebool container_use_devices=1`.
- `HSA_OVERRIDE_GFX_VERSION` makes the runtime load kernels built for another target; it can crash or give wrong results. With mixed GPUs set it per device (`HSA_OVERRIDE_GFX_VERSION_0=...`).
- Memory shown in `amd-smi` includes PyTorch's cached allocator memory, so a process can look full while tensors use less; check `torch.cuda.memory_allocated()`.
- Quantization formats and custom kernels can be NVIDIA-only; a checkpoint that serves on CUDA may not on ROCm. Check the engine's ROCm feature matrix before choosing a format ([[quantizing-llms-for-serving]]).
- ROCm, PyTorch and vLLM versions move fast and the supported-GPU lists change; re-check the install pages on each upgrade rather than copying a version from a blog.

## Related
- [[gpu-cuda-setup]] - the NVIDIA equivalent and VRAM sizing rules.
- [[vllm-local-setup]] - vLLM install and start-up errors in general.
- [[inference-servers-vllm]] - production serving once the GPU works.
- [[llama-cpp-gguf]] - GGUF models and llama.cpp flags.
- [[ollama]] - the simplest local runner, ROCm or Vulkan underneath.
- [[docker-for-ml]] - GPU containers, image size and model cache volumes.
- [[python-env-uv]] - uv, Python versions and extra index handling.
- [[gpu-cloud-options]] - renting AMD Instinct instead of buying.
- [[disaggregated-prefill-and-kv-cache-offloading]] - vLLM KV connectors, including the ROCm-only MoRIIOConnector.
- [[intel-gpu-and-openvino-setup]] - the Intel GPU/NPU equivalent.

## References
- ROCm installation on Linux (AMD): https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
- PyTorch HIP (ROCm) semantics (`torch.cuda` reuse, `torch.version.hip`, memory management): https://pytorch.org/docs/stable/notes/hip.html
- vLLM GPU installation, ROCm tab (supported targets, Python 3.12 wheels, CUDA-wheel fallback, Docker images): https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
- llama.cpp build docs, HIP and Vulkan sections (`GGML_HIP`, `GPU_TARGETS`, `HSA_OVERRIDE_GFX_VERSION`): https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- Ollama GPU docs (AMD Radeon support, overrides, Vulkan, container permissions): https://github.com/ollama/ollama/blob/main/docs/gpu.mdx
- Ollama Docker docs (`ollama/ollama:rocm`): https://github.com/ollama/ollama/blob/main/docs/docker.mdx
