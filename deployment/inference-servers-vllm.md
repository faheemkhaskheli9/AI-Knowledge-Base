---
title: LLM inference servers (vLLM, SGLang, TGI)
category: deployment
tags: [vllm, sglang, tgi, inference-server, openai-compatible, continuous-batching, kv-cache, tensor-parallel]
use_cases:
  - "serve an open-weight LLM to many concurrent users on my own GPUs"
  - "expose a self-hosted model through an OpenAI-compatible endpoint so existing clients just work"
  - "pick between vLLM, SGLang and llama.cpp for a production deployment"
  - "run a private LLM for an internal assistant with high throughput"
status: draft
last_verified: 2026-10-03
sources:
  - https://docs.vllm.ai/en/latest/getting_started/quickstart/
  - https://docs.sglang.io/get_started/install.html
  - https://huggingface.co/docs/text-generation-inference/index
---

# LLM inference servers (vLLM, SGLang, TGI)

## Summary
Dedicated inference servers run LLMs on GPUs with continuous batching, paged KV-cache memory and optimized kernels, giving far higher throughput under concurrency than a plain `transformers.generate` loop. vLLM and SGLang are the main open-source choices; both expose OpenAI-compatible HTTP APIs, so clients only change `base_url`.

## Key concepts
- Continuous batching: new requests join the running batch every step instead of waiting for a batch to finish.
- PagedAttention / KV-cache management: the cache is allocated in blocks, allowing many concurrent sequences; KV memory, not weights, usually limits concurrency.
- Prefix caching: reuse the KV of shared prompt prefixes (system prompts, RAG templates).
- Tensor parallelism: split a model across GPUs when it does not fit on one.
- OpenAI-compatible API: `/v1/chat/completions`, `/v1/models` and others.
- Quantized models (AWQ, GPTQ, FP8) and speculative decoding cut memory or latency; check each server's docs for supported formats.

## When to use / scenarios
- Internal assistant or RAG backend for a company with a few GPUs and data that cannot leave the network.
- Batch processing millions of documents (offline engine, no HTTP).
- A fine-tuned model that must be served with low latency to a product.
- Not needed for single-user local use ([[ollama]], [[llama-cpp-gguf]]), or low volume where a hosted API is cheaper ([[cost-and-latency]]).

## Setup & code
vLLM (Linux; Python 3.10-3.13 per the quickstart; on Windows use [[windows-wsl-setup]]):
```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
vllm serve <hf-org>/<model>            # serves on http://localhost:8000
curl http://localhost:8000/v1/models
```
Tuning flags commonly used: `--max-model-len`, `--gpu-memory-utilization`, `--tensor-parallel-size`, `--api-key`. These were not re-verified here; confirm with `vllm serve --help` for your version.

Client:
```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="<key-or-dummy>")
r = c.chat.completions.create(model="<hf-org>/<model>",
                              messages=[{"role": "user", "content": "Hello"}], stream=True)
for ch in r:
    print(ch.choices[0].delta.content or "", end="")
```
Offline batch:
```python
from vllm import LLM, SamplingParams
llm = LLM(model="<hf-org>/<model>")
for o in llm.generate(["Summarize: ..."], SamplingParams(temperature=0.2, max_tokens=200)):
    print(o.outputs[0].text)
```
SGLang (docs require Python 3.10+ and a CUDA 13 environment for current releases; verify for yours):
```bash
uv pip install --prerelease=allow sglang
sglang serve <hf-org>/<model> --host 0.0.0.0 --port 30000
```
TGI: Hugging Face states it is in maintenance mode and recommends vLLM, SGLang, llama.cpp or MLX going forward; prefer those for new projects.

## Choosing / trade-offs
- vLLM: broad model/hardware support and community; the default pick.
- SGLang: strong on prefix-heavy and structured/agentic workloads; tracks newer CUDA requirements.
- llama.cpp server: CPU/mixed or Apple hardware, smaller scale.
- Managed alternatives (hosted open-model APIs, [[gpu-cloud-options]]) trade control for no ops.
- Quantization lets a model fit fewer GPUs at some quality cost ([[quantization]]).

## Gotchas
- Setting max context too high reserves KV memory and reduces concurrency.
- Never expose the port publicly without an API key plus reverse proxy/TLS.
- First start downloads weights and warms up; health-check before routing traffic.
- Chat template and stop tokens must match the model.
- Major versions change flags and CUDA/PyTorch pairs; pin versions in Docker images.
- Throughput figures from blogs depend on prompt/output length mix; benchmark your own traffic.

## Related
- [[gpu-cuda-setup]] - VRAM sizing and drivers.
- [[serving-with-fastapi]] - application layer in front of the server.
- [[cost-and-latency]] - when self-hosting pays off.
- [[llm-observability]] - metrics and tracing.
- [[quantization]], [[long-context]] - memory levers.

## References
- vLLM quickstart: https://docs.vllm.ai/en/latest/getting_started/quickstart/
- SGLang install: https://docs.sglang.io/get_started/install.html
- TGI: https://huggingface.co/docs/text-generation-inference/index
