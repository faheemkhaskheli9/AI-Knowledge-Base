---
title: Quantization (int8, int4, GPTQ, AWQ, GGUF)
category: concepts
tags: [quantization, int8, int4, gptq, awq, gguf, bitsandbytes, llama-cpp]
use_cases:
  - "run a 7B-13B model on a laptop or a single consumer GPU"
  - "reduce GPU memory and serving cost of an LLM endpoint"
  - "ship an offline on-device assistant for field workers"
  - "fit a larger base model for QLoRA fine-tuning in 24GB VRAM"
  - "choose between GGUF, AWQ and GPTQ for a production deployment"
status: draft
last_verified: 2026-10-03
sources:
  - https://huggingface.co/docs/transformers/main/en/quantization/overview
  - https://huggingface.co/docs/bitsandbytes
  - https://arxiv.org/abs/2210.17323
  - https://arxiv.org/abs/2306.00978
  - https://github.com/ggml-org/llama.cpp
---

# Quantization (int8, int4, GPTQ, AWQ, GGUF)

## Summary
Quantization stores model weights (and sometimes activations or the KV cache) in fewer bits, for example 8-bit or 4-bit instead of 16-bit. This cuts memory roughly 2-4x and often speeds up memory-bound inference, with a small quality loss that depends on the method and bit width.

## Key concepts
- **Weight memory**: params x bytes. A 7B model is ~14 GB in 16-bit, ~7 GB in int8, ~3.5-4.5 GB in 4-bit (plus overhead). Add KV cache and activations on top.
- **PTQ vs QAT**: post-training quantization (no retraining, most common) vs quantization-aware training (better quality, costly).
- **Groupwise scales**: weights are quantized in blocks (e.g. 32-128 values) each with a scale, balancing accuracy and overhead.
- **bitsandbytes (LLM.int8, NF4)**: on-the-fly quantization at load time in Transformers; easy, used for QLoRA; NVIDIA-centred.
- **GPTQ**: one-shot, layer-wise PTQ using second-order information; GPU inference formats.
- **AWQ**: activation-aware; protects the most salient weight channels; good 4-bit GPU quality.
- **GGUF (llama.cpp)**: single-file format with k-quants (Q4_K_M, Q5_K_M, Q8_0...) for CPU/Apple/consumer GPU inference; used by llama.cpp and Ollama.
- **FP8 / activation quantization**: used on newer data-centre GPUs for throughput; supported by servers such as vLLM (check hardware support).
- **KV-cache quantization**: shrinks memory for long contexts.

## When to use / scenarios
- Local/offline/edge: GGUF with llama.cpp or Ollama ([[llama-cpp-gguf]], [[ollama]]).
- GPU serving at scale: AWQ/GPTQ/FP8 weights in vLLM ([[inference-servers-vllm]]).
- Fine-tuning big models cheaply: NF4 base + LoRA ([[fine-tuning-and-peft]]).
- Real-world: a hospital runs an 8B model on-prem on one GPU; a retailer cuts endpoint cost by serving 4-bit weights.
- Avoid or be careful: tiny models (<3B) degrade more; tasks needing exact numerics or rare-language fidelity; below 4-bit without testing.

## Setup & code
`pip install transformers accelerate bitsandbytes`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

name = "Qwen/Qwen2.5-1.5B-Instruct"
cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=cfg, device_map="auto")
tok = AutoTokenizer.from_pretrained(name)
out = model.generate(**tok("Quantization is", return_tensors="pt").to(model.device), max_new_tokens=20)
print(tok.decode(out[0]), model.get_memory_footprint() / 1e9, "GB")
```

GGUF route (llama.cpp): download a prebuilt `*-Q4_K_M.gguf` or convert with the repo's `convert_hf_to_gguf.py` and `llama-quantize`, then run `llama-cli -m model.gguf -p "..."`. Verify exact tool names against the current llama.cpp README.

## Choosing / trade-offs
- Bits: 8-bit is near-lossless; 4-bit (Q4_K_M, AWQ, NF4) is the usual sweet spot; 3-bit and below shows visible loss.
- Bigger model at 4-bit usually beats a smaller model at 16-bit for the same memory.
- GGUF (portable, CPU-friendly) vs AWQ/GPTQ (GPU throughput) vs bitsandbytes (convenience, slower inference).
- Calibration-based methods (GPTQ/AWQ) need a representative calibration set; domain mismatch costs quality.

## Gotchas
- Quantized weights may not speed things up if the kernel is not optimised (bitsandbytes 4-bit is often slower than 16-bit on throughput).
- Format and backend mismatch: a GPTQ checkpoint will not load in llama.cpp, a GGUF will not load in vLLM the same way as HF weights; match format to server.
- Memory estimates forget KV cache; long contexts can dominate.
- Always re-run your own eval after quantizing; perplexity alone hides task regressions.
- Quantized models from random uploaders may be mislabelled or poorly calibrated; prefer known sources.

## Related
- [[fine-tuning-and-peft]] - QLoRA.
- [[llama-cpp-gguf]] - practical GGUF workflow.
- [[ollama]] - one-command local serving of quantized models.
- [[inference-servers-vllm]] - GPU serving formats.
- [[edge-on-device]] - on-device constraints.
- [[small-language-models]] - alternative to quantizing a big one.

## References
- Transformers quantization overview: https://huggingface.co/docs/transformers/main/en/quantization/overview
- bitsandbytes: https://huggingface.co/docs/bitsandbytes
- Frantar et al., GPTQ: https://arxiv.org/abs/2210.17323
- Lin et al., AWQ: https://arxiv.org/abs/2306.00978
- llama.cpp: https://github.com/ggml-org/llama.cpp
