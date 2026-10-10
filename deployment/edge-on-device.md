---
title: Edge and on-device inference (ONNX, TensorRT, Core ML, mobile)
category: deployment
tags: [edge, on-device, onnx, onnxruntime, tensorrt, coreml, tflite, mobile, jetson]
use_cases:
  - "run an image model on a phone or camera without sending data to the cloud"
  - "export a PyTorch model to ONNX and speed it up with ONNX Runtime"
  - "deploy a detector on an NVIDIA Jetson or similar edge box"
  - "run a small LLM on a laptop or phone offline"
status: draft
last_verified: 2026-10-03
sources:
  - https://onnxruntime.ai/docs/
  - https://pytorch.org/docs/stable/onnx.html
  - https://docs.nvidia.com/deeplearning/tensorrt/
  - https://apple.github.io/coremltools/
---

# Edge and on-device inference (ONNX, TensorRT, Core ML, mobile)

## Summary
Edge deployment runs models on the device that produces the data (phone, camera, robot, laptop, factory gateway). It cuts latency and bandwidth, works offline and keeps data local. The cost is tight memory, power and compute, so models are exported to portable formats and optimized (quantized, fused) per runtime.

## Key concepts
- ONNX: open interchange format; ONNX Runtime executes it on CPU and via execution providers (CUDA, TensorRT, DirectML, Core ML, others).
- TensorRT: NVIDIA's compiler/runtime producing optimized engines for a specific GPU; best on NVIDIA GPUs and Jetson.
- Core ML: Apple's format and runtime (Neural Engine, GPU, CPU); convert with `coremltools`.
- TensorFlow Lite / LiteRT and ExecuTorch: Google's and PyTorch's mobile runtimes (check current docs for naming and status).
- llama.cpp / MLX for LLMs on laptops and Apple Silicon ([[llama-cpp-gguf]]).
- Optimizations: INT8/FP16 quantization, pruning, distillation, smaller architectures ([[quantization]], [[small-language-models]]).

## When to use / scenarios
- Retail/factory cameras running detection locally ([[object-detection]], [[video-analytics]]).
- Privacy-critical apps (health, personal photos) where raw data must not leave the device.
- Poor-connectivity field tools, voice commands on a handheld.
- Not worth it when the model is huge and traffic is low: call a server instead.

## Setup & code
Export PyTorch to ONNX and run it:
```bash
uv add torch onnx onnxruntime
```
```python
import torch, numpy as np, onnxruntime as ort
model = torch.nn.Sequential(torch.nn.Linear(4, 2)).eval()      # your model here
x = torch.randn(1, 4)
torch.onnx.export(model, x, "m.onnx", input_names=["x"], output_names=["y"],
                  dynamic_axes={"x": {0: "batch"}})
sess = ort.InferenceSession("m.onnx", providers=["CPUExecutionProvider"])
print(sess.run(None, {"x": x.numpy()})[0])
```
GPU: install the GPU build of ONNX Runtime and list providers, e.g. `["CUDAExecutionProvider", "CPUExecutionProvider"]`; package names and CUDA pairings are in the ONNX Runtime install docs, so check them for your versions.

TensorRT (NVIDIA GPU/Jetson): build an engine from the ONNX file with the `trtexec` tool shipped with TensorRT, then run via the TensorRT runtime or ONNX Runtime's TensorRT provider; see NVIDIA docs for exact flags.

Core ML: `coremltools` converts PyTorch models (`ct.convert(traced_model, inputs=[...])`) into `.mlpackage` files for Xcode; see its docs for current API.

Validate after every conversion: compare outputs to the original on a test set (tolerance, e.g. 1e-3 for FP32; measure task metrics for INT8).

## Choosing / trade-offs
- Portability (ONNX Runtime) vs peak speed (TensorRT, Core ML native): vendor runtimes are faster but lock you in per platform.
- Quantization: INT8 gives big speed/size gains with accuracy risk; calibrate on representative data.
- On-device vs server: device wins on latency/privacy/offline; server wins on model size, easy updates and centralized monitoring.
- Edge LLMs: pick the smallest model that passes your eval; expect limited context and speed.

## Gotchas
- Not all PyTorch ops export; dynamic control flow and custom ops need rewriting.
- TensorRT engines are tied to GPU architecture and TensorRT version; rebuild per target.
- Preprocessing (resize, normalization) must be identical on-device; most accuracy bugs live there.
- Thermal throttling and battery limits make benchmarks on a cold device optimistic.
- Shipping weights in an app exposes them; weigh model-theft risk and license terms ([[model-licenses]]).
- Updating models on thousands of devices needs a rollout and rollback mechanism ([[mlops-lifecycle]]).

## Related
- [[quantization]] - shrinking models.
- [[small-language-models]] - models for constrained hardware.
- [[llama-cpp-gguf]] - LLMs on CPUs and phones.
- [[object-detection]], [[image-classification]] - typical edge workloads.
- [[gpu-cuda-setup]] - NVIDIA basics.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/deployment-strategies.md - staged rollouts of new models to devices.

## References
- ONNX Runtime: https://onnxruntime.ai/docs/
- PyTorch ONNX export: https://pytorch.org/docs/stable/onnx.html
- TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/
- coremltools: https://apple.github.io/coremltools/
