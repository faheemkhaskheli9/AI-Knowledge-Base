---
title: Intel GPUs and NPUs (PyTorch XPU, OpenVINO GenAI, vLLM XPU, llama.cpp SYCL, Ollama Vulkan)
category: setup
tags: [intel, xpu, arc, core-ultra, npu, openvino, openvino-genai, optimum-intel, oneapi, sycl, level-zero, vllm, llama-cpp, ollama, vulkan, ipex]
use_cases:
  - "get PyTorch to see my Intel Arc GPU or Core Ultra integrated graphics"
  - "run a local LLM on an Intel laptop's GPU or NPU with OpenVINO"
  - "serve an open model with vLLM on Intel Arc or Data Center GPU Max"
  - "build llama.cpp for an Intel GPU, or run Ollama on one"
  - "migrate off Intel Extension for PyTorch or IPEX-LLM now that they are archived"
status: draft
last_verified: 2026-10-24
sources:
  - https://github.com/pytorch/pytorch/blob/main/docs/source/notes/get_start_xpu.md
  - https://github.com/openvinotoolkit/openvino.genai/blob/master/README.md
  - https://github.com/openvinotoolkit/openvino.genai/blob/master/site/docs/getting-started/installation.mdx
  - https://github.com/openvinotoolkit/openvino.genai/blob/master/site/docs/guides/model-preparation/convert-to-openvino.mdx
  - https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst
  - https://github.com/vllm-project/vllm/blob/main/docs/getting_started/installation/gpu.xpu.inc.md
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md
  - https://github.com/ollama/ollama/blob/main/docs/gpu.mdx
  - https://github.com/intel/intel-extension-for-pytorch/blob/main/README.md
---

# Intel GPUs and NPUs (PyTorch XPU, OpenVINO GenAI, vLLM XPU, llama.cpp SYCL, Ollama Vulkan)

## Summary
Intel GPUs (Arc A/B-series discrete cards, the Arc graphics in Core Ultra laptops, and Data Center GPU Max) and the NPU in Core Ultra processors can run AI workloads, but through different stacks than NVIDIA: PyTorch exposes Intel GPUs as the `xpu` device, OpenVINO (and its OpenVINO GenAI library) runs exported models on CPU, GPU or NPU, vLLM has an XPU build, llama.cpp has a SYCL backend, and Ollama reaches Intel GPUs through Vulkan. The older Intel Extension for PyTorch (IPEX) and IPEX-LLM are archived; their features were upstreamed into PyTorch, so new projects should use plain PyTorch XPU or OpenVINO.

## Key concepts
- **`xpu` device in PyTorch**: Intel GPU support is in official PyTorch (prototype since 2.5, for client GPUs and Data Center GPU Max on Linux and Windows). Install wheels from the `xpu` index URL; check with `torch.xpu.is_available()`; port CUDA code by replacing `"cuda"` with `"xpu"`. Training, inference, `torch.compile`, FP32/BF16/FP16 and AMP are supported per the PyTorch notes.
- **Driver and runtime**: Intel's GPU driver plus the Level Zero runtime; oneAPI components (SYCL runtime, compilers) ship inside the PyTorch XPU wheels, but building llama.cpp's SYCL backend needs the oneAPI Base Toolkit (`source /opt/intel/oneapi/setvars.sh`, check devices with `sycl-ls`). The NPU needs its own NPU driver (Windows driver download or `linux-npu-driver` releases).
- **OpenVINO and OpenVINO GenAI**: OpenVINO Runtime runs models in its IR format on `"CPU"`, `"GPU"` or `"NPU"` with no code change beyond the device string. OpenVINO GenAI adds ready pipelines (LLM, VLM, Whisper, image generation, embeddings, rerank) with tokenization built in. Models are exported from Hugging Face with `optimum-cli export openvino` (Optimum Intel), optionally compressed with `--weight-format int4|int8|fp16`, or downloaded pre-converted from the `OpenVINO` organization on Hugging Face (repos with an `-ov` suffix). GGUF files can be loaded directly as a preview feature.
- **Version lock**: `openvino-genai`, `openvino` and `openvino-tokenizers` must share major.minor.patch; mixing a nightly `openvino` with a released GenAI wheel breaks with `ImportError: libopenvino.so...`.
- **vLLM on XPU**: supports Intel Data Center GPU and Arc GPUs, needs Python 3.12 (the `vllm-xpu-kernels` wheel is 3.12-specific), installs from the vLLM XPU wheel index plus the PyTorch XPU index, or runs from `vllm/vllm-openai-xpu` images with `--device /dev/dri`. Tensor parallel is supported; pipeline parallel is beta.
- **llama.cpp SYCL**: build with `-DGGML_SYCL=ON` and the oneAPI compilers (`icx`/`icpx`); `-DGGML_SYCL_F16=ON` is an optional FP16 build. Pick devices with `ONEAPI_DEVICE_SELECTOR="level_zero:0"` (or `level_zero:0;level_zero:1`) or `--main-gpu`. The Vulkan backend is the alternative that needs no oneAPI.
- **Ollama**: uses Vulkan for Intel GPUs (enabled by default when the backend is installed; Linux needs Intel's Vulkan driver packages). Select GPUs with `GGML_VK_VISIBLE_DEVICES`; disable with `OLLAMA_VULKAN=0`.
- **Archived IPEX and IPEX-LLM**: both repositories are archived; Intel states it will not provide further support and that the projects have known security issues. Do not start new work on them.

## When to use / scenarios
- A Windows or Linux laptop with Core Ultra (Meteor Lake, Lunar Lake, Arrow Lake, Panther Lake) running a private assistant, Whisper transcription or embeddings locally: OpenVINO GenAI on GPU or NPU ([[small-language-models]], [[edge-on-device]]).
- A desktop with an Arc A770 or B580 for local LLMs: llama.cpp SYCL or Vulkan, Ollama via Vulkan, or OpenVINO GenAI.
- PyTorch experiments or small fine-tunes on an Arc card: PyTorch XPU wheels.
- Serving on Intel Data Center GPU Max or Arc in a server: vLLM XPU image ([[inference-servers-vllm]]).
- CPU-only Intel servers: OpenVINO on `"CPU"` or llama.cpp; both are often enough for embeddings, rerankers and small models ([[embedding-and-reranker-serving]]).
- Not when your stack needs CUDA-only pieces (TensorRT-LLM, many custom kernels, some quantization formats) or you need maximum throughput per server: use NVIDIA ([[gpu-cuda-setup]]) or AMD ([[amd-rocm-setup]]) and check library support first.

## Setup & code
PyTorch on an Intel GPU (after Intel's GPU driver prerequisites), from the PyTorch XPU notes:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```
<!-- skip-check: needs an Intel GPU, its driver and the PyTorch XPU wheel -->
```python
import torch

assert torch.xpu.is_available(), "no XPU: CPU/CUDA wheel installed, or Intel GPU driver missing"
print(torch.xpu.get_device_name(0))
x = torch.randn(2048, 2048, device="xpu")            # "xpu", not "cuda"
with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
    print((x @ x).float().abs().mean().item())
```
OpenVINO GenAI: export once, then run on CPU, GPU or NPU (from the OpenVINO GenAI README):
```bash
pip install openvino-genai
optimum-cli export openvino --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weight-format int4 TinyLlama_1_1b_v1_ov
# or download a pre-converted model: hf download OpenVINO/<model>-ov --local-dir model_ov
```
<!-- skip-check: needs openvino-genai and an exported model directory -->
```python
import openvino_genai as ov_genai

pipe = ov_genai.LLMPipeline("TinyLlama_1_1b_v1_ov", "GPU")   # "CPU", "GPU" or "NPU"
print(pipe.generate("What is OpenVINO?", max_new_tokens=100))
```
vLLM on XPU (Docker, from vLLM's XPU install docs) and llama.cpp with SYCL (from its SYCL backend docs):
```bash
docker run --rm --network=host --device /dev/dri:/dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN" --ipc=host --privileged \
  vllm/vllm-openai-xpu:<tag> --model <model>

source /opt/intel/oneapi/setvars.sh && sycl-ls         # expect a [level_zero:gpu] device
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON
cmake --build build --config Release -j
ONEAPI_DEVICE_SELECTOR="level_zero:0" ./build/bin/llama-server -m model.gguf -ngl 99
```
Runnable, stdlib only: a preflight that checks which Intel stacks are importable on this machine and which devices they report, then suggests a route. It imports optional packages by name at runtime, so it runs (and prints "not installed") on any machine.
```python
import importlib
import importlib.util
import platform

def probe(module):
    if importlib.util.find_spec(module) is None:
        return None
    try:
        return importlib.import_module(module)
    except Exception as e:                       # e.g. ABI mismatch between openvino packages
        print(f"{module}: import failed: {e}")
        return None

report = {"os": platform.system(), "torch_xpu": None, "openvino_devices": None,
          "openvino_genai": probe("openvino_genai") is not None}

torch = probe("torch")
if torch is not None and hasattr(torch, "xpu"):
    report["torch_xpu"] = bool(torch.xpu.is_available())
ov = probe("openvino")
if ov is not None:
    report["openvino_devices"] = list(ov.Core().available_devices)

def advise(r):
    tips = []
    devices = r["openvino_devices"] or []
    if r["torch_xpu"]:
        tips.append("PyTorch XPU works: use device='xpu' for training and experiments")
    elif r["torch_xpu"] is False:
        tips.append("torch installed but no XPU: reinstall from the download.pytorch.org/whl/xpu index and check the driver")
    if any(d.startswith("NPU") for d in devices):
        tips.append("NPU visible to OpenVINO: run small LLMs/Whisper with OpenVINO GenAI on 'NPU'")
    if any(d.startswith("GPU") for d in devices):
        tips.append("Intel GPU visible to OpenVINO: OpenVINO GenAI on 'GPU' for local inference")
    if not tips:
        tips.append("nothing Intel-specific found: install the GPU driver, then torch (xpu index) or openvino-genai; "
                    "llama.cpp (SYCL or Vulkan) and Ollama (Vulkan) are alternatives")
    return tips

for k, v in report.items():
    print(f"{k:18s} {v}")
for t in advise(report):
    print("-", t)
# Examples of the advice for other machines
print(advise({"torch_xpu": None, "openvino_devices": ["CPU", "GPU", "NPU"]}))
print(advise({"torch_xpu": False, "openvino_devices": ["CPU"]}))
```

## Choosing / trade-offs
| Goal on Intel hardware | Route | Watch out for |
|---|---|---|
| Local LLM on a Core Ultra laptop | OpenVINO GenAI on GPU or NPU | export step; NPU driver; model must be supported by the NPU plugin |
| Local chat with GGUF files | llama.cpp SYCL or Vulkan; Ollama (Vulkan) | SYCL needs oneAPI; compare both backends on your card |
| PyTorch training/experiments | PyTorch XPU wheels | CUDA-only extensions; prototype status for some features |
| Serving many users | vLLM XPU image on Data Center GPU Max or Arc | Python 3.12 only; feature support lags CUDA |
| CPU-only Intel server | OpenVINO on CPU or llama.cpp | throughput per server; batch where possible |

- **GPU vs NPU**: the NPU is a low-power accelerator for always-on or battery-sensitive features; the integrated or discrete GPU is usually the faster choice for interactive LLM chat. Measure tokens per second and power on your model ([[llm-load-testing-and-benchmarking]]).
- **OpenVINO vs PyTorch**: OpenVINO is for deployment (export once, small runtime, CPU/GPU/NPU); PyTorch XPU is for development and training with the same code as on NVIDIA.
- **Weight format**: INT4 weight compression in the export cuts memory and usually speeds decoding at some quality cost; evaluate on your task ([[quantization]], [[llm-evaluation]]).

## Gotchas
- Default PyPI `torch` is not the XPU build; `torch.xpu.is_available()` returns False until you install from the `whl/xpu` index.
- Guides that still use `import intel_extension_for_pytorch` or `ipex-llm` point at archived projects with known security issues; use upstream PyTorch XPU, OpenVINO, llama.cpp or vLLM instead.
- vLLM XPU wheels require Python 3.12, and the install needs both the vLLM XPU and PyTorch XPU indexes with `--index-strategy unsafe-best-match` under `uv`.
- In containers, pass `/dev/dri` (vLLM's docs also mount `/dev/dri/by-path`); the user also needs access to the render device on the host.
- On machines with an iGPU and a discrete Arc card, tools may pick the iGPU; select the device explicitly (`ONEAPI_DEVICE_SELECTOR`, `--main-gpu`, `GGML_VK_VISIBLE_DEVICES`, or the OpenVINO device name such as `GPU.1`).
- OpenVINO wheels on PyPI and the C++ archives use different C++ ABIs; do not mix them in one build.
- A model exported with one OpenVINO release may need re-export after a major upgrade; keep the original Hugging Face checkpoint.
- Supported-hardware lists change per release (the PyTorch notes list validated Arc and Core Ultra generations); check them before buying hardware for a specific stack.

## Related
- [[gpu-cuda-setup]] - the NVIDIA equivalent, and VRAM sizing rules that also apply here.
- [[amd-rocm-setup]] - the AMD equivalent, with the same "separate builds" pattern.
- [[windows-wsl-setup]] - Windows and WSL2 environment basics.
- [[llama-cpp-gguf]] - GGUF models and llama.cpp flags.
- [[ollama]] - the simplest local runner, Vulkan underneath on Intel.
- [[vllm-local-setup]] - vLLM install and start-up errors in general.
- [[onnx-and-triton-inference-server]] - ONNX export and other runtimes, including OpenVINO for non-LLM models.
- [[edge-on-device]] - on-device inference trade-offs across vendors.
- [[small-language-models]] - models that fit a laptop GPU or NPU.
- [[docker-for-ml]] - device passthrough and model cache volumes.

## References
- PyTorch, Getting Started on Intel GPU (hardware list, `whl/xpu` index, `torch.xpu`, supported features): https://github.com/pytorch/pytorch/blob/main/docs/source/notes/get_start_xpu.md
- OpenVINO GenAI README (pipelines, quick start, devices): https://github.com/openvinotoolkit/openvino.genai/blob/master/README.md
- OpenVINO GenAI installation (package version compatibility, ABI notes): https://github.com/openvinotoolkit/openvino.genai/blob/master/site/docs/getting-started/installation.mdx
- OpenVINO GenAI, Convert models to OpenVINO format (`optimum-cli export openvino`, weight formats): https://github.com/openvinotoolkit/openvino.genai/blob/master/site/docs/guides/model-preparation/convert-to-openvino.mdx
- OpenVINO NPU device (Core Ultra NPU, drivers): https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.rst
- vLLM installation on Intel XPU (requirements, wheels, Docker image, parallelism): https://github.com/vllm-project/vllm/blob/main/docs/getting_started/installation/gpu.xpu.inc.md
- llama.cpp for SYCL (oneAPI build, `GGML_SYCL`, `ONEAPI_DEVICE_SELECTOR`): https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md
- Ollama GPU docs (Vulkan support, Intel driver link, GPU selection): https://github.com/ollama/ollama/blob/main/docs/gpu.mdx
- Intel Extension for PyTorch (archived; retirement plan): https://github.com/intel/intel-extension-for-pytorch/blob/main/README.md
