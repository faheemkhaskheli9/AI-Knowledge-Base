---
title: llama.cpp and GGUF
category: setup
tags: [llama.cpp, gguf, quantization, cpu-inference, apple-silicon, llama-server]
use_cases:
  - "run a quantized open model on CPU or a small GPU"
  - "convert a Hugging Face model to GGUF and quantize it to 4-bit"
  - "serve a local OpenAI-compatible endpoint from a laptop or edge box"
  - "split a model between GPU and CPU RAM when it does not fit in VRAM"
status: draft
last_verified: 2026-10-03
sources:
  - https://github.com/ggml-org/llama.cpp
  - https://github.com/ggml-org/llama.cpp/blob/master/README.md
---

# llama.cpp and GGUF

## Summary
llama.cpp is a dependency-light C/C++ inference engine for LLMs supporting CPUs, Apple Silicon, NVIDIA, AMD and others. GGUF is its single-file model format holding weights (often quantized), tokenizer and metadata. Together they let you run billion-parameter models on ordinary hardware, with fine control over quantization and GPU offload.

## Key concepts
- GGUF: one file per model/quant; name suffixes like `Q4_K_M`, `Q5_K_M`, `Q8_0` denote quantization schemes (lower = smaller, lossier).
- `-ngl` (number of GPU layers): how many transformer layers to offload to the GPU; the rest run on CPU.
- Context size and batch settings drive KV-cache memory.
- Tools: `llama-cli` (chat/completion), `llama-server` (OpenAI-compatible HTTP server), `llama-quantize`, and `convert_hf_to_gguf.py` (Hugging Face to GGUF).
- `-hf <repo>` loads a GGUF directly from Hugging Face.

## When to use / scenarios
- Edge/offline: a Raspberry-Pi-class or laptop deployment of a small model ([[edge-on-device]]).
- Squeezing a 13B-70B model onto a 12-24 GB GPU plus system RAM.
- Reproducible local inference without Python.
- Prefer [[ollama]] for convenience, [[inference-servers-vllm]] for high-throughput GPU serving.

## Setup & code
Install options (check the README for current installer): prebuilt binaries from the GitHub releases page, Docker images, or build from source:
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build            # add the backend flag you need, e.g. CUDA, see docs/build.md
cmake --build build --config Release -j
```
Run a model straight from Hugging Face (binaries land in `build/bin`):
```bash
llama-cli -hf <hf-user>/<repo-GGUF> -ngl 99
llama-server -hf <hf-user>/<repo-GGUF> -ngl 99 --port 8080
```
Call the server (OpenAI-compatible):
```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
print(c.chat.completions.create(model="local",
      messages=[{"role": "user", "content": "Hi"}]).choices[0].message.content)
```
Convert and quantize your own model:
```bash
python convert_hf_to_gguf.py /path/to/hf-model --outfile model-f16.gguf
llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
```
Verify flag spellings with `llama-cli --help` / `llama-server --help` for your build; flags evolve.

Python bindings: `pip install llama-cpp-python` (compiles llama.cpp; GPU builds need backend CMake args per its README).

## Choosing / trade-offs
- Quant choice: Q4_K_M is a common size/quality balance; Q5/Q6/Q8 for quality, Q2/Q3 only when forced by memory ([[quantization]]).
- Full GPU offload (high -ngl) is fastest; partial offload trades speed for fit.
- GGUF vs GPTQ/AWQ (GPU-oriented formats served by vLLM): GGUF for CPU/mixed, AWQ/GPTQ for pure GPU serving.
- Importance-matrix (imatrix) quants improve low-bit quality at the cost of a calibration step.

## Gotchas
- Chat template mismatches degrade output; GGUF usually embeds the template, but custom conversions may not.
- Architectures need support in llama.cpp; brand-new models can lag.
- Big context + many parallel slots in `llama-server` multiplies KV-cache memory.
- Downloaded GGUFs from unknown uploaders are unverified weights; prefer official or well-known repos.
- Always match the build backend (CUDA/Metal/Vulkan) to hardware; a CPU-only build ignores `-ngl`.

## Related
- [[ollama]] - friendlier wrapper.
- [[quantization]] - theory behind the formats.
- [[gpu-cuda-setup]] - VRAM sizing.
- [[huggingface-transformers]] - source of models to convert.
- [[edge-on-device]] - mobile and embedded options.

## References
- https://github.com/ggml-org/llama.cpp (README, docs/build.md)
