---
title: Quantizing LLMs for GPU serving (FP8, W4A16, INT8 with llm-compressor and vLLM)
category: deployment
tags: [quantization, fp8, w4a16, int8, gptq, awq, llm-compressor, compressed-tensors, vllm, kv-cache, gpu-memory]
use_cases:
  - "fit a larger open-weight model, or more concurrent users, on the GPUs we already have"
  - "produce an FP8 or 4-bit checkpoint of our fine-tuned model for vLLM"
  - "choose between FP8, INT8 and 4-bit weight-only quantization for production serving"
  - "check that a quantized model has not lost accuracy on our own tasks before rollout"
status: draft
last_verified: 2026-10-24
sources:
  - https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/
  - https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int4/
  - https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int8_w8a8/
  - https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
  - https://docs.vllm.ai/en/latest/features/quantization/online/
  - https://github.com/vllm-project/llm-compressor
  - https://github.com/EleutherAI/lm-evaluation-harness
  - https://arxiv.org/abs/2209.05433
  - https://arxiv.org/abs/2210.17323
  - https://arxiv.org/abs/2211.10438
---

# Quantizing LLMs for GPU serving (FP8, W4A16, INT8 with llm-compressor and vLLM)

## Summary
Serving cost is set by GPU memory: the weights plus the KV cache for every concurrent request. Quantizing weights (and optionally activations and the KV cache) to 8 or 4 bits frees memory for more concurrent tokens or lets a bigger model fit, and on GPUs with low-precision tensor cores it also speeds up the matmuls. The practical path for vLLM is: try on-the-fly FP8 (`--quantization fp8_per_tensor`) for a quick check, produce a calibrated checkpoint with llm-compressor (FP8 dynamic, W4A16 with GPTQ, or INT8 W8A8) saved in the compressed-tensors format vLLM loads directly, then compare it with the 16-bit model on your own eval set and under load before rollout. This topic is the serving workflow; [[quantization]] covers the methods themselves.

## Key concepts
- **Scheme names**: `WxAy` means x-bit weights, y-bit activations. W8A8 (FP8 or INT8) quantizes both, so the matmul itself runs in 8 bits; W4A16 stores 4-bit weights and dequantizes them to 16-bit for the matmul.
- **Memory-bound vs compute-bound**: at small batch sizes decoding is limited by reading weights from memory, so fewer weight bits help most (W4A16). At high concurrency the GPU becomes compute-bound and 8-bit activation formats (W8A8) help more, because weight-only schemes still compute in 16 bits.
- **Hardware support**: vLLM's FP8 W8A8 path needs FP8 tensor cores (NVIDIA compute capability 8.9 or newer: Ada Lovelace, Hopper and later). On Ampere, vLLM runs FP8 checkpoints as weight-only W8A16 through Marlin kernels, which saves memory but not compute. Newer 4-bit float formats target newer GPU generations; check the vLLM quantization support table for your GPU.
- **Dynamic vs static scales**: FP8 dynamic computes activation scales per token at run time and needs no calibration data. GPTQ (W4A16) and SmoothQuant + GPTQ (INT8 W8A8) need a few hundred representative calibration samples.
- **Granularity**: one scale per tensor, per output channel, or per group of input columns (for example 128). Finer granularity costs a few extra bits per weight for scales but lowers error, especially with outlier channels (demonstrated below).
- **What stays in 16-bit**: recipes usually skip `lm_head` (and, for mixture-of-experts models, the router gates); embeddings and norms are not quantized.
- **KV cache quantization**: `--kv-cache-dtype fp8` halves KV memory per token relative to BF16, roughly doubling the tokens that fit, independently of the weight scheme ([[kv-cache-from-scratch-numpy]]).
- **compressed-tensors**: llm-compressor writes the quantization config into the checkpoint's `config.json`; vLLM reads it and picks kernels, so no extra serving flag is needed.

## When to use / scenarios
- A 70B-class model needs more GPUs than budgeted in BF16: 8-bit or 4-bit weights cut the GPU count ([[gpu-cloud-options]]).
- Concurrency is limited by KV-cache room, visible as queueing or preemption at load: FP8 weights plus an FP8 KV cache free memory for more sequences ([[llm-load-testing-and-benchmarking]]).
- A fine-tuned model you own must ship as an efficient checkpoint: quantize after merging any LoRA into the base ([[fine-tuning-and-peft]]).
- Not for CPU, laptop or Apple-silicon serving: use GGUF with llama.cpp or Ollama ([[llama-cpp-gguf]], [[ollama]], [[mlx-on-apple-silicon]]).
- Not without an eval: accuracy-sensitive tasks (math, code, long-context extraction) can degrade under 4-bit schemes; measure on your tasks first ([[llm-evaluation]]).
- Often not needed for hosted APIs: you pay per token and the provider chooses precision ([[cost-and-latency]]).

## Setup & code
Quick check with no new checkpoint: vLLM quantizes BF16/FP16 weights to FP8 at load time ("online quantization"); add an FP8 KV cache. Current docs name the scheme `fp8_per_tensor` (also `fp8_per_block`); releases from before these shorthands use `fp8`, which current releases still accept:
```bash
vllm serve <hf-model-id> --quantization fp8_per_tensor --kv-cache-dtype fp8
```
Calibrated checkpoints with llm-compressor (`pip install llmcompressor`). FP8 dynamic needs no calibration data:
<!-- skip-check: downloads model weights and needs a GPU -->
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot                  # older releases: from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "<hf-model-id>"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")   # older transformers: torch_dtype=
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
oneshot(model=model, recipe=recipe)

SAVE_DIR = MODEL_ID.split("/")[-1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```
W4A16 with GPTQ needs calibration data in your traffic's format (chat template applied):
<!-- skip-check: downloads model weights and a dataset, needs a GPU -->
```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID, N, MAX_LEN = "<hf-model-id>", 512, 2048
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")   # older transformers: torch_dtype=
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

ds = load_dataset("<chat-dataset-like-your-traffic>", split="train").shuffle(seed=0).select(range(N))
ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)})
ds = ds.map(lambda ex: tokenizer(ex["text"], max_length=MAX_LEN, truncation=True,
                                 padding=False, add_special_tokens=False),
            remove_columns=ds.column_names)

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=MAX_LEN, num_calibration_samples=N)
model.save_pretrained(MODEL_ID.split("/")[-1] + "-W4A16", save_compressed=True)
tokenizer.save_pretrained(MODEL_ID.split("/")[-1] + "-W4A16")
```
Serve the result and compare with the original on the same benchmark (lm-evaluation-harness has a vLLM backend), then on your own eval set. The vLLM docs note that quantized models can be sensitive to a missing BOS token and `lm_eval` does not add one by default, hence `add_bos_token=True`:
```bash
vllm serve ./<model>-W4A16
lm_eval --model vllm --model_args pretrained=./<model>-W4A16,add_bos_token=True --tasks gsm8k --num_fewshot 5 --batch_size auto
lm_eval --model vllm --model_args pretrained=<hf-model-id>,add_bos_token=True --tasks gsm8k --num_fewshot 5 --batch_size auto
```
Runnable with NumPy only: (1) a memory plan of weights against KV-cache room for an 8B-class model at different precisions on 24 GB and 80 GB GPUs; (2) int4 output error on a weight matrix with outlier input channels, per-tensor vs per-channel vs group-wise scales.
```python
import numpy as np

# 1) Memory plan: does the model plus enough KV cache fit on the GPU?
# Shape of a Llama-3.1-8B-style config (from its config.json); swap in your model's numbers.
# All weights at one precision; real checkpoints keep lm_head/embeddings in 16-bit, a bit more.
cfg = {"params": 8.03e9, "layers": 32, "kv_heads": 8, "head_dim": 128}

def plan(gpu_gb, weight_bits, kv_bytes, util=0.90, overhead_gb=2.0):
    weights_gb = cfg["params"] * weight_bits / 8 / 1e9
    kv_per_token = 2 * cfg["layers"] * cfg["kv_heads"] * cfg["head_dim"] * kv_bytes   # K and V
    free_gb = gpu_gb * util - weights_gb - overhead_gb       # overhead: activations, CUDA graphs
    tokens = int(max(free_gb, 0) * 1e9 // kv_per_token)
    return weights_gb, tokens

for name, bits, kv in [("BF16 weights, BF16 KV", 16, 2), ("FP8 weights, BF16 KV", 8, 2),
                       ("FP8 weights, FP8 KV", 8, 1), ("W4A16 weights, BF16 KV", 4, 2)]:
    for gpu in (24, 80):
        w, t = plan(gpu, bits, kv)
        print(f"{name:24s} {gpu:2d} GB GPU: weights {w:5.1f} GB, KV room {t:>9,} tokens "
              f"(~{t // 8192:>3} concurrent 8k-token requests)")

# 2) Why scale granularity matters: int4 error on a weight matrix with outlier input channels.
rng = np.random.default_rng(0)
W = rng.standard_normal((1024, 1024)).astype(np.float32) * 0.02
W[:, rng.choice(1024, 8, replace=False)] *= 25               # a few large-magnitude channels

def fake_quant(w, bits, granularity):
    """Symmetric round-to-nearest; one scale per tensor, per output row, or per group of columns."""
    qmax = 2 ** (bits - 1) - 1
    if granularity == "tensor":
        s = np.abs(w).max() / qmax
        return np.clip(np.round(w / s), -qmax - 1, qmax) * s
    if granularity == "row":
        s = np.abs(w).max(1, keepdims=True) / qmax
        return np.clip(np.round(w / s), -qmax - 1, qmax) * s
    g = w.reshape(w.shape[0], -1, granularity)               # group-wise along the input dim
    s = np.abs(g).max(2, keepdims=True) / qmax
    return (np.clip(np.round(g / s), -qmax - 1, qmax) * s).reshape(w.shape)

x = rng.standard_normal((256, 1024)).astype(np.float32)      # stand-in calibration activations
ref = x @ W.T
for label, gran in [("per-tensor", "tensor"), ("per-channel", "row"), ("group 128", 128), ("group 32", 32)]:
    err = np.linalg.norm(x @ fake_quant(W, 4, gran).T - ref) / np.linalg.norm(ref)
    extra_bits = 0 if gran in ("tensor", "row") else 16 / gran   # one fp16 scale per group
    print(f"int4 {label:11s}: relative output error {err:.3f}, ~{4 + extra_bits:.2f} bits/weight")
```
GPTQ and AWQ go beyond this round-to-nearest baseline by using calibration activations to choose roundings or protect salient channels ([[int8-quantization-from-scratch-numpy]]).

## Choosing / trade-offs
| Scheme | Pick when | Watch out for |
|---|---|---|
| FP8 W8A8 (dynamic) | Ada/Hopper or newer GPUs; want the smallest quality risk and a speedup at high concurrency | on older GPUs only the memory saving remains |
| INT8 W8A8 (SmoothQuant + GPTQ) | GPUs with INT8 but no FP8 tensor cores (Ampere, Turing) | needs calibration data; more quality risk than FP8; vLLM does not support INT8 on Blackwell (compute capability 10.0+) |
| W4A16 (GPTQ, AWQ) | memory-bound: single-GPU fits, low-batch latency, large models | largest quality risk; less benefit at high batch sizes |
| BF16, no quantization | the model fits with enough KV room and quality is critical | highest GPU cost |
| GGUF (llama.cpp) | CPU, Apple silicon, desktop apps | different stack; see [[llama-cpp-gguf]] |

- **Pre-quantized vs your own**: many model publishers and community repos ship FP8 or 4-bit checkpoints; they save work but check who made them, which calibration data they used and the licence. A fine-tuned model has to be quantized by you.
- **FP8 KV cache**: often the cheapest concurrency gain; validate on long-context tasks, where small errors accumulate.
- **Calibration data**: use samples that look like production prompts (same language, chat template, length). Generic web text can cost quality on domain or non-English traffic.
- **Quantize or pick a smaller model**: a quantized larger model is often better than a 16-bit model of the same memory footprint, but not always; compare both on your eval ([[small-language-models]], [[model-selection]]).

## Gotchas
- Benchmarks averaged over many tasks hide regressions on the one task you care about; always run your own eval set, at the sampling settings you serve with.
- Quantize after merging LoRA adapters; an adapter trained on the 16-bit base and applied to a quantized base drifts. For multi-LoRA serving on a quantized base, test the adapters on that exact base ([[multi-lora-adapter-serving]]).
- Calibration samples without the chat template, or truncated far shorter than production prompts, give scales tuned to the wrong distribution.
- `--quantization fp8_per_tensor` at load time quantizes per tensor without calibration; a calibrated checkpoint can be more accurate and loads faster.
- Kernel support differs by GPU, scheme and vLLM release; a checkpoint that loads on H100 may fall back to slow kernels or fail on another GPU. Check startup logs for the kernel chosen.
- Throughput is not just memory: measure TTFT, TPOT and goodput under your load before and after ([[llm-load-testing-and-benchmarking]]).
- Record precision, scheme and calibration set in the model registry; "the same model" at two precisions behaves differently ([[model-registry-and-versioning]]).

## Related
- [[quantization]] - the methods (GPTQ, AWQ, GGUF, bitsandbytes) and formats.
- [[int8-quantization-from-scratch-numpy]] - absmax, per-channel and outlier handling from scratch.
- [[inference-servers-vllm]] - the server that loads compressed-tensors checkpoints.
- [[vllm-local-setup]] - single-GPU vLLM install and KV-cache sizing.
- [[kv-cache-from-scratch-numpy]] - why KV cache memory grows with concurrent tokens.
- [[sglang-and-tensorrt-llm]] - other servers with their own quantized formats.
- [[llm-load-testing-and-benchmarking]] - measuring the speedup under realistic traffic.
- [[gpu-cloud-options]] - what GPU generation you rent decides which schemes are fast.
- [[multi-lora-adapter-serving]] - adapters on a quantized base model.
- [[llama-cpp-gguf]] - the CPU and desktop alternative.
- [[lm-evaluation-harness-and-lighteval]] - lm-eval backends, settings and pitfalls for before/after comparisons.

## References
- vLLM FP8 W8A8 (llm-compressor recipe, hardware support, `add_bos_token` note): https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/
- vLLM INT4 W4A16 (GPTQ with llm-compressor, calibration data): https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int4/
- vLLM INT8 W8A8 (SmoothQuant + GPTQ, supported GPUs): https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/int8_w8a8/
- vLLM online quantization (`fp8_per_tensor`, `fp8_per_block` and other load-time schemes): https://docs.vllm.ai/en/latest/features/quantization/online/
- vLLM quantized KV cache: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- llm-compressor: https://github.com/vllm-project/llm-compressor
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
- Micikevicius et al., FP8 Formats for Deep Learning (2022): https://arxiv.org/abs/2209.05433
- Frantar et al., GPTQ (2022): https://arxiv.org/abs/2210.17323
- Xiao et al., SmoothQuant (2022): https://arxiv.org/abs/2211.10438
