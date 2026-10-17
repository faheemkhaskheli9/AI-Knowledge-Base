---
title: MLX on Apple silicon
category: setup
tags: [mlx, mlx-lm, apple-silicon, mac, metal, unified-memory, lora, qlora, quantization, mlx-community, mlx-whisper, mlx-vlm]
use_cases:
  - "run an open LLM locally on a MacBook or Mac Studio with good speed"
  - "LoRA or QLoRA fine-tune a small model on a Mac without an NVIDIA GPU"
  - "convert and 4-bit quantize a Hugging Face model for Apple silicon"
  - "serve a local OpenAI-style chat endpoint from a Mac for development"
  - "transcribe audio with Whisper on a Mac"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/ml-explore/mlx
  - https://ml-explore.github.io/mlx/build/html/index.html
  - https://github.com/ml-explore/mlx-lm
  - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
  - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md
  - https://github.com/ml-explore/mlx-examples
  - https://github.com/Blaizzy/mlx-vlm
---

# MLX on Apple silicon

## Summary
MLX is Apple machine learning research's open-source array framework for Apple silicon. It has a NumPy-like Python API (plus C++, C and Swift APIs), lazy evaluation, composable autodiff, and a **unified memory** model: arrays live in memory shared by the CPU and GPU, so nothing is copied between devices. On top of it, **mlx-lm** (MIT licence) generates text, converts and quantizes Hugging Face models, LoRA/QLoRA/DoRA fine-tunes them, and serves an HTTP API. Sibling packages cover vision-language models (mlx-vlm) and Whisper speech recognition (mlx-whisper). It is the native, Python-friendly way to run and adapt open models on a Mac.

## Key concepts
- **Unified memory**: model size is limited by the Mac's RAM (minus what macOS reserves for the GPU), not by a separate VRAM pool. For large models on macOS 15+, mlx-lm documents raising the wired-memory limit (`sysctl iogpu.wired_limit_mb`).
- **Lazy evaluation**: operations build a graph that runs only when a result is needed (`mx.eval`, printing, or converting to NumPy). Benchmarks must force evaluation, or they time nothing.
- **Devices**: CPU and GPU (Metal). The same code runs on both with no transfers. There is also a CUDA backend (`pip install mlx[cuda12]`) and a CPU-only Linux build, but Apple silicon is the main target.
- **mlx-community**: a Hugging Face organisation with thousands of pre-converted MLX models (for example `...-4bit` repos), loadable by name.
- **Conversion and quantization**: `mlx_lm.convert --model <hf-repo> -q` writes a quantized MLX model, 4-bit in the README example. `--upload-repo` publishes it.
- **Fine-tuning**: `mlx_lm.lora` trains LoRA (default), DoRA or full weights (`--fine-tune-type`). If `--model` points at a quantized model, training is QLoRA. `mlx_lm.fuse` merges adapters into the base and can export GGUF, but only for Mistral, Mixtral and Llama-style models in fp16.
- **Data format**: a folder with `train.jsonl` (plus optional `valid.jsonl` and `test.jsonl`). Each line is in one of three formats, detected automatically: `chat` (`messages`), `completions` (`prompt`/`completion`) or `text`. `--mask-prompt` computes loss on completions only.

## When to use / scenarios
- Developers on MacBooks: local chat, coding and RAG prototypes with private data and no cloud bill ([[small-language-models]]).
- Mac Studio or Mac mini with large unified memory: run models too big for a single consumer NVIDIA card, at lower tokens per second than a datacenter GPU.
- Small-team fine-tuning: adapt a 1B-8B model to a format or style on a Mac ([[fine-tuning-and-peft]]).
- On-device Apple apps: mlx-swift brings the same models into Swift apps ([[edge-on-device]]).
- Audio and vision locally: mlx-whisper for transcription ([[speech-to-text]]), mlx-vlm for image Q&A ([[vision-language-models]]).
- Not for: production multi-user serving (the mlx-lm server docs say it is not recommended for production; use [[inference-servers-vllm]] on GPUs), CUDA-only libraries, or large training jobs (use [[unsloth-finetuning]] or [[distributed-training]] on NVIDIA GPUs).

## Setup & code
Requirements for the PyPI wheels: Apple silicon, a native (arm64, not Rosetta) Python 3.10 or later, and macOS 14.0 or later ([[python-env-uv]] for environments).
```bash
pip install mlx-lm                       # pulls in mlx; use "mlx-lm[train]" for fine-tuning
mlx_lm.generate --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --prompt "Explain unified memory in one line"
mlx_lm.chat     --model mlx-community/Mistral-7B-Instruct-v0.3-4bit     # interactive REPL
mlx_lm.convert  --model mistralai/Mistral-7B-Instruct-v0.3 -q           # quantize to ./mlx_model
mlx_lm.server   --model mlx-community/Mistral-7B-Instruct-v0.3-4bit     # OpenAI-like API on localhost:8080
```
Python API (from the mlx-lm README):
<!-- skip-check: needs Apple silicon and a model download -->
```python
from mlx_lm import load, generate, stream_generate

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
messages = [{"role": "user", "content": "Write a haiku about unified memory"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=100))
for r in stream_generate(model, tokenizer, prompt, max_tokens=100):
    print(r.text, end="", flush=True)
```
Core MLX looks like NumPy but is lazy:
<!-- skip-check: needs Apple silicon (or an mlx CPU/CUDA build) -->
```python
import mlx.core as mx

a = mx.random.normal((4096, 4096))
b = a @ a.T                # nothing computed yet
mx.eval(b)                 # runs on the GPU by default; no host/device copies
grad = mx.grad(lambda x: (x ** 2).sum())(mx.array([1.0, 2.0]))
print(b.shape, grad)
```
LoRA fine-tuning. First write the data folder; this stdlib block runs anywhere:
```python
import json, os, random

pairs = [(f"Classify the ticket: 'My card {i} was charged twice'", "billing") for i in range(40)] + \
        [(f"Classify the ticket: 'App crash number {i} on login'", "bug") for i in range(40)]
random.Random(0).shuffle(pairs)
os.makedirs("data", exist_ok=True)
splits = {"train": pairs[:64], "valid": pairs[64:72], "test": pairs[72:]}
for name, rows in splits.items():
    with open(f"data/{name}.jsonl", "w", encoding="utf-8") as f:
        for p, c in rows:                     # "completions" format; one JSON object per line
            f.write(json.dumps({"prompt": p, "completion": c}) + "\n")
print({k: len(v) for k, v in splits.items()})
```
Then train, test, use and fuse (commands from mlx-lm's LORA.md):
```bash
pip install "mlx-lm[train]"
mlx_lm.lora --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --train --data ./data \
            --iters 600 --batch-size 1 --num-layers 8 --mask-prompt --grad-checkpoint   # 4-bit base = QLoRA
mlx_lm.lora --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --adapter-path adapters --data ./data --test
mlx_lm.generate --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --adapter-path adapters \
            --prompt "Classify the ticket: 'Refund not received'"
mlx_lm.fuse --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --adapter-path adapters   # -> fused_model/
```
Lower memory with `--batch-size 1`, fewer `--num-layers`, `--grad-checkpoint`, `--grad-accumulation-steps`, shorter examples, or a quantized base.

## Choosing / trade-offs
| Option on a Mac | Best for | Trade-off |
|---|---|---|
| mlx-lm | Python control, fine-tuning, conversion, newest MLX models | CLI/library, not a GUI; server is for development only |
| [[llama-cpp-gguf]] (Metal) | GGUF ecosystem, portable files, cross-platform deployment | Fine-tuning is not its job |
| [[ollama]] | Easiest local model runner and API | Less control over conversion and training |
| [[lm-studio-and-jan]] | GUI chat and local server; LM Studio has an MLX engine (`lmstudio-ai/mlx-engine`) | Desktop app, less scriptable |
| [[unsloth-finetuning]] on a cloud GPU | Faster or bigger training runs | Costs money, data leaves the Mac |

- MLX format vs GGUF: MLX models (safetensors plus config) are for the MLX stack, and GGUF is for llama.cpp and Ollama. Pick the format your serving target needs. `mlx_lm.fuse --export-gguf` only covers some architectures in fp16.
- Quantization bits: 4-bit is the common default for fitting larger models in RAM. Use 8-bit or 16-bit when quality matters and memory allows ([[quantization]]).
- RAM sizing: weights plus KV cache must fit in unified memory, with headroom for macOS and your apps. Long contexts grow the KV cache, and the server's `--kv-bits` option quantizes it.

## Gotchas
- An x86 Python under Rosetta cannot use MLX. Check that `python -c "import platform; print(platform.machine())"` prints `arm64`.
- Lazy evaluation hides cost. Call `mx.eval` before timing, and remember that NumPy conversion or printing forces computation.
- Swap kills throughput. A model that "fits" but forces swapping runs many times slower, so choose a smaller model or lower-bit quant instead.
- mlx-lm releases move fast, and new architectures need a recent version. Upgrade before assuming a model is unsupported.
- Chat templates: train and infer with the same template. When fine-tuning chat models, use the `chat` data format or `--mask-prompt`.
- The `mlx_lm.server` binds to localhost and has only basic security checks. Do not expose it to a network without a proper proxy and auth.
- Community conversions in `mlx-community` are uploads by many people. Check the source model and licence ([[model-licenses]]).
- GGUF export from `mlx_lm.fuse` is limited (Mistral, Mixtral and Llama-style models, fp16). For other architectures, fuse to safetensors and convert with llama.cpp's `convert_hf_to_gguf.py`.

## Related
- [[llama-cpp-gguf]] - the other main Mac inference stack, and GGUF conversion.
- [[ollama]], [[lm-studio-and-jan]] - friendlier local runners.
- [[unsloth-finetuning]] - the NVIDIA-GPU counterpart for fine-tuning.
- [[fine-tuning-and-peft]], [[lora-from-scratch-numpy]] - LoRA/QLoRA concepts.
- [[quantization]] - what 4-bit and 8-bit mean.
- [[small-language-models]] - models that suit laptop memory.
- [[edge-on-device]] - shipping models inside apps.
- [[python-env-uv]] - native arm64 Python environments.

## References
- MLX: https://github.com/ml-explore/mlx; docs (install requirements) https://ml-explore.github.io/mlx/build/html/index.html
- mlx-lm (CLI, Python API, convert): https://github.com/ml-explore/mlx-lm
- LoRA guide (data formats, QLoRA, fuse, memory tips): https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- Server: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md
- mlx-examples (mlx-whisper and more): https://github.com/ml-explore/mlx-examples
- mlx-vlm: https://github.com/Blaizzy/mlx-vlm
- LM Studio MLX engine: https://github.com/lmstudio-ai/mlx-engine
