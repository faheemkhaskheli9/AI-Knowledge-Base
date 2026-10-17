---
title: SGLang and TensorRT-LLM (vs vLLM)
category: deployment
tags: [sglang, tensorrt-llm, trtllm-serve, vllm, radixattention, structured-output, xgrammar, inference-server, nvidia, amd]
use_cases:
  - "choose between vLLM, SGLang and TensorRT-LLM for serving an open-weight LLM"
  - "serve an agent workload with long shared prefixes and JSON-constrained outputs at high throughput"
  - "squeeze maximum tokens per second out of NVIDIA H100/B200 GPUs for a production LLM"
  - "run an LLM server on AMD, TPU or Ascend hardware"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/sgl-project/sglang
  - https://docs.sglang.io/
  - https://arxiv.org/abs/2312.07104
  - https://github.com/NVIDIA/TensorRT-LLM
  - https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
  - https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/guided-decoding.md
  - https://docs.vllm.ai/en/latest/
---

# SGLang and TensorRT-LLM (vs vLLM)

## Summary
SGLang and TensorRT-LLM are the two main alternatives to vLLM for high-throughput LLM serving. All three do continuous batching, paged KV cache, quantization, speculative decoding, tensor/expert parallelism and expose an OpenAI-compatible server, so the choice rests on hardware, workload shape and operational fit, not on features alone. SGLang (Apache-2.0, LMSYS) is built around RadixAttention prefix reuse and runs on many accelerators. TensorRT-LLM (Apache-2.0, NVIDIA) uses NVIDIA-specific kernels and runs only on NVIDIA GPUs. vLLM stays the broad default ([[inference-servers-vllm]]).

## Key concepts
- **RadixAttention (SGLang)**: KV cache of finished requests is kept in a radix tree keyed by token prefix and evicted LRU. A new request reuses the longest cached prefix automatically: system prompts, few-shot examples, multi-turn chat history, agent loops, tree-of-thought branches. Introduced in the SGLang paper (Zheng et al., 2023). vLLM has automatic prefix caching with a similar goal.
- **Structured output**: all three support grammar-constrained decoding. SGLang has `--grammar-backend` with XGrammar (default), Outlines and LLGuidance. TensorRT-LLM supports XGrammar and LLGuidance through `guided_decoding_backend` in a YAML passed to `trtllm-serve --config`. vLLM also has pluggable backends. Clients pass `response_format` with a JSON schema via the OpenAI API. How it works: [[grammar-constrained-decoding-from-scratch]], [[structured-output]].
- **TensorRT-LLM architecture**: current releases are built on PyTorch with a Python `LLM` API (`from tensorrt_llm import LLM, SamplingParams`) and `trtllm-serve` for OpenAI-compatible HTTP. Older releases required compiling a TensorRT engine per model/GPU/config; check the docs for your version before following old tutorials.
- **NVIDIA-specific optimizations**: FP8/FP4 checkpoints from NVIDIA Model Optimizer, prefill/decode disaggregation, wide expert parallelism for MoE, multi-node NVLink deployments.
- **SGLang hardware reach**: per its README, NVIDIA (consumer to GB300), AMD Instinct, Google TPU, Intel GPU/CPU, Apple Silicon, Huawei Ascend and others.
- **Telemetry**: TensorRT-LLM collects anonymous usage telemetry by default. Opt out with `TRTLLM_NO_USAGE_STATS=1` or `--no-telemetry` (per its README).

## When to use / scenarios
- **Agents, RAG and chat with heavy prefix sharing** (customer-support bots with a long system prompt, coding agents re-sending history, evaluation runs with shared few-shot prompts): SGLang's radix cache shines; benchmark against vLLM prefix caching on your traces.
- **Large NVIDIA fleet, one or a few fixed models, cost per token is the KPI** (a high-volume consumer chat product, an MoE model on B200/GB200 racks): TensorRT-LLM, often behind NVIDIA Dynamo or Triton.
- **Non-NVIDIA hardware** (AMD MI300-class, TPU, Ascend): SGLang or vLLM; not TensorRT-LLM.
- **RL post-training rollouts**: SGLang is integrated in several RL frameworks (verl, slime, AReaL per its README); vLLM is also common there.
- **Many models, frequent swaps, broad community support, general default**: vLLM.
- Not for laptops, CPUs or single-user use: [[ollama]], [[llama-cpp-gguf]]. Not for low volume where a hosted API is cheaper: [[cost-and-latency]].

## Setup & code
SGLang (Docker image or uv; see [[inference-servers-vllm]] for version caveats):
```bash
docker pull lmsysorg/sglang:latest
# or: uv pip install --prerelease=allow sglang
sglang serve <hf-org>/<model> --host 0.0.0.0 --port 30000 --grammar-backend xgrammar
```
TensorRT-LLM (run inside the NGC release container; see the installation guide):
```bash
trtllm-serve "<hf-org>/<model>"                 # OpenAI-compatible on :8000
echo "guided_decoding_backend: xgrammar" > config.yaml
trtllm-serve "<hf-org>/<model>" --config config.yaml   # enable JSON-schema output
```
The same client works against any of the three servers; only `base_url` changes:
<!-- skip-check: needs a running SGLang, TensorRT-LLM or vLLM server -->
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:30000/v1", api_key="dummy")  # :8000 for trtllm-serve/vLLM
schema = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
          "required": ["city", "population"]}
r = client.chat.completions.create(
    model="<hf-org>/<model>",
    messages=[{"role": "user", "content": "Largest city in Japan, as JSON."}],
    response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": schema}},
)
print(r.choices[0].message.content)
```
TensorRT-LLM offline Python API (same shape as vLLM's `LLM`):
<!-- skip-check: needs tensorrt_llm and an NVIDIA GPU -->
```python
from tensorrt_llm import LLM, SamplingParams
llm = LLM(model="<hf-org>/<model>")
for out in llm.generate(["The capital of France is"], SamplingParams(temperature=0.8, top_p=0.95)):
    print(out.outputs[0].text)
```
Why prefix reuse matters: a stdlib simulation of a radix-style prefix cache over an agent-like workload (shared system prompt, growing history). It counts how many prompt tokens must be prefilled with and without reuse.
```python
import random

random.seed(0)
SYSTEM = list(range(1000, 1800))                  # 800-token shared system prompt

class PrefixCache:                                # trie of token ids, like a radix tree
    def __init__(self):
        self.root = {}
    def match_and_insert(self, toks):
        node, hit = self.root, 0
        for t in toks:
            if t in node:
                hit += 1
            node = node.setdefault(t, {})
        return hit

cache, total, prefilled = PrefixCache(), 0, 0
for user in range(20):                            # 20 conversations
    history = SYSTEM[:]
    for turn in range(5):                         # 5 turns each, history re-sent every turn
        history += [random.randrange(50_000) for _ in range(60)]
        hit = cache.match_and_insert(history)
        total += len(history)
        prefilled += len(history) - hit
        history += [random.randrange(50_000) for _ in range(80)]   # model reply joins history

print(f"prompt tokens sent: {total}, prefilled with cache: {prefilled}, "
      f"saved {1 - prefilled / total:.0%}")
assert prefilled < total * 0.5
```
In this workload the shared system prompt and re-sent history make up most of the prompt tokens, so most prefill work is skipped. Real savings depend on cache size, eviction and traffic mix, so measure them on your own traces.

## Choosing / trade-offs
| Factor | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| Hardware | NVIDIA, AMD, TPU, others via plugins | NVIDIA, AMD, TPU, Intel, Apple, Ascend, ... | NVIDIA GPUs only |
| Prefix reuse | automatic prefix caching | RadixAttention (core design) | KV cache reuse supported; check docs |
| Structured output | pluggable backends | XGrammar (default), Outlines, LLGuidance | XGrammar, LLGuidance |
| Strength | breadth of models, community, default choice | prefix-heavy, agentic, RL rollouts | peak NVIDIA performance, FP8/FP4, MoE at rack scale |
| Ops cost | low | low to medium | higher: NVIDIA containers, tighter CUDA/driver pairing |

- **Throughput claims**: each project publishes benchmarks where it wins. Results swing with model, GPU, quantization, prompt/output length and concurrency, so run `vllm bench`, SGLang's bench scripts or `trtllm-bench` on your own traffic before switching. No numbers are quoted here on purpose.
- **Lock-in**: all three speak the OpenAI API, so switching servers is mostly a deployment change. TensorRT-LLM ties you to NVIDIA hardware.
- **Day-0 model support** differs per release; check each project's supported-models list for the model you need.

## Gotchas
- Mixing CUDA, driver and PyTorch versions breaks all three; use the official containers and pin image tags.
- Structured output can slow decoding when grammars are compiled per request; reuse schemas so the backend's grammar cache hits.
- Old TensorRT-LLM guides that build engines with `trtllm-build` may not match current releases; follow the docs for your version.
- Prefix caching only helps if the shared part is byte-identical and at the start: put timestamps and user IDs after the static prompt ([[prompt-caching-and-cost]]).
- Benchmarks at one concurrency level mislead; measure TTFT and inter-token latency at your p95 load ([[llm-observability]]).
- Do not expose any of these servers without auth and TLS in front.

## Related
- [[inference-servers-vllm]] - the default server and the shared concepts (continuous batching, paged KV).
- [[kubernetes-for-llm-serving]] - running these servers on a cluster.
- [[onnx-and-triton-inference-server]] - Triton, which can also host TensorRT-LLM.
- [[batch-inference-and-batch-apis]] - offline jobs on the same engines.
- [[structured-output]], [[grammar-constrained-decoding-from-scratch]] - constrained decoding.
- [[kv-cache-from-scratch-numpy]] - what is being cached.
- [[speculative-decoding-from-scratch-numpy]] - a latency feature all three ship.
- [[quantization]] - FP8/FP4/INT4 formats these servers load.
- [[gpu-cloud-options]] - where to get the GPUs.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/load-balancing.md - routing across replicas (prefix-aware routing extends this).

## References
- SGLang repository (features, supported hardware, licence): https://github.com/sgl-project/sglang
- SGLang docs (structured outputs, server arguments): https://docs.sglang.io/
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (RadixAttention): https://arxiv.org/abs/2312.07104
- TensorRT-LLM repository (architecture, telemetry, licence): https://github.com/NVIDIA/TensorRT-LLM
- TensorRT-LLM quick start (`trtllm-serve`, LLM API): https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html
- TensorRT-LLM guided decoding: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/guided-decoding.md
- vLLM docs: https://docs.vllm.ai/en/latest/
