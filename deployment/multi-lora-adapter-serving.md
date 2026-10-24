---
title: Multi-LoRA serving (many fine-tuned adapters on one base model)
category: deployment
tags: [lora, multi-lora, adapters, vllm, lorax, s-lora, punica, multi-tenant, fine-tuning, peft]
use_cases:
  - "serve a separate fine-tuned model per customer without one GPU deployment each"
  - "host dozens of task-specific LoRA adapters behind one OpenAI-compatible endpoint"
  - "load a newly trained adapter into a running vLLM server without a restart"
  - "decide between merging a LoRA into the base weights and serving it as an adapter"
status: draft
last_verified: 2026-10-24
sources:
  - https://docs.vllm.ai/en/latest/features/lora/
  - https://arxiv.org/abs/2311.03285
  - https://arxiv.org/abs/2310.18547
  - https://github.com/predibase/lorax
  - https://huggingface.co/docs/peft/developer_guides/lora
---

# Multi-LoRA serving (many fine-tuned adapters on one base model)

## Summary
A LoRA fine-tune changes only small low-rank matrices next to frozen base weights, so many fine-tunes of the same base can share one copy of the base model in GPU memory. A multi-LoRA server (vLLM with `--enable-lora`, LoRAX, or the S-LoRA/Punica designs they build on) keeps the base loaded once, holds a set of adapters on the GPU, and batches requests for *different* adapters together in one forward pass. That turns "one GPU deployment per customer or task" into "one deployment, many adapters", with the request's `model` field choosing the adapter.

## Key concepts
- **Shared base, per-request adapter**: every adapter must be trained on exactly the same base checkpoint the server loads. The request names the adapter; requests naming the base model get the base.
- **Heterogeneous batching**: requests for different adapters run in the same batch. The base matmul is shared; each token adds its own adapter's low-rank product `x @ A @ B`, computed by gathered kernels (Punica's SGMV, S-LoRA's custom kernels) instead of one matmul per adapter.
- **Adapter slots**: the server reserves GPU memory for a fixed number of adapters at once (vLLM `--max-loras`) at a fixed maximum rank (`--max-lora-rank`), and can keep more in host memory (`--max-cpu-loras`). A request whose adapter is not resident waits for a load, so slot count against tenant count is the main sizing question.
- **Adapter size**: each targeted linear layer of shape `d_out x d_in` adds `r x (d_in + d_out)` parameters. For an 8B-class model that is single-digit MB at rank 8 on two projections and tens to hundreds of MB at higher ranks on all linear layers (computed below).
- **Static vs dynamic registration**: adapters can be listed at startup (`--lora-modules name=path`) or loaded at runtime through an admin endpoint; runtime loading needs an explicit opt-in in vLLM because it lets a caller make the server read arbitrary paths.
- **Merge vs serve**: `merge_and_unload()` in PEFT folds an adapter into the base weights, giving base-model speed but one full model per fine-tune. Serving unmerged costs some per-token overhead but shares the base.

## When to use / scenarios
- SaaS with per-customer fine-tunes (tone, product names, formats): one base, an adapter per tenant, the tenant ID mapped to an adapter name at the gateway ([[llm-gateways-and-routing]]).
- Several narrow task adapters (SQL generation, classification, extraction) that each beat prompting but do not justify their own GPU ([[fine-tuning-and-peft]]).
- A/B or canary of a new adapter version next to the old one on the same server ([[ab-testing-and-canary-for-models]]).
- Not when you have one fine-tune and high traffic: merge it and serve it as a plain model for the best latency.
- Not for full fine-tunes or adapters on different bases: those are separate models and need separate deployments ([[inference-servers-vllm]]).
- Not needed if the provider hosts your fine-tune: hosted tuning serves the adapter for you ([[hosted-fine-tuning-apis]]).

## Setup & code
vLLM OpenAI-compatible server with adapters registered at startup (`<base-model>` is the Hugging Face repo the adapters were trained on; set the limits from your largest adapter and expected concurrency):
```bash
vllm serve <base-model> --enable-lora \
  --lora-modules support-tone=/adapters/support-tone sql=/adapters/sql \
  --max-loras 8 --max-lora-rank 16 --max-cpu-loras 32
curl -s localhost:8000/v1/models          # lists the base and each adapter name
```
Load and unload adapters on a running server (only on a trusted network; the endpoint lets callers point the server at any path):
```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve <base-model> --enable-lora --max-loras 8
curl -s localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' \
  -d '{"lora_name": "tenant-42", "lora_path": "/adapters/tenant-42"}'
curl -s localhost:8000/v1/unload_lora_adapter -H 'Content-Type: application/json' \
  -d '{"lora_name": "tenant-42"}'
```
Calling an adapter: the `model` field is the adapter name.
<!-- skip-check: needs a running vLLM server with the adapters loaded -->
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused-locally")
resp = client.chat.completions.create(
    model="sql",                                  # adapter name, not the base model ID
    messages=[{"role": "user", "content": "Monthly revenue by region for 2025"}],
    max_tokens=200,
)
print(resp.choices[0].message.content)
```
Offline batch with vLLM's Python API (one `LoRARequest` per adapter; the integer ID must be unique per adapter):
<!-- skip-check: downloads base weights and needs a GPU -->
```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="<base-model>", enable_lora=True, max_loras=4, max_lora_rank=16)
params = SamplingParams(temperature=0, max_tokens=128)
out = llm.generate(["Monthly revenue by region for 2025"], params,
                   lora_request=LoRARequest("sql", 1, "/adapters/sql"))
print(out[0].outputs[0].text)
```
Runnable, stdlib only: adapter size per rank and target modules, a pre-deployment check of an adapter's `adapter_config.json` fields against the server's limits, and an LRU simulation of how many requests find their adapter already on the GPU for a given slot count under skewed tenant traffic.
```python
import random
from collections import OrderedDict

# 1) Adapter size: each targeted linear layer W (d_out x d_in) adds r * (d_in + d_out) params.
# Shapes of a Llama-3.1-8B-style model (hidden 4096, MLP 14336, 8 KV heads x 128, 32 layers).
H, MLP, KV, LAYERS = 4096, 14336, 8 * 128, 32
SHAPES = {"q_proj": (H, H), "k_proj": (H, KV), "v_proj": (H, KV), "o_proj": (H, H),
          "gate_proj": (H, MLP), "up_proj": (H, MLP), "down_proj": (MLP, H)}   # (d_in, d_out)

def adapter_mb(rank, targets, bytes_per_param=2):
    params = LAYERS * sum(rank * (SHAPES[t][0] + SHAPES[t][1]) for t in targets)
    return params * bytes_per_param / 1e6

for rank, targets in [(8, ["q_proj", "v_proj"]), (16, list(SHAPES)), (64, list(SHAPES))]:
    print(f"rank {rank:2d}, {len(targets)} target modules: {adapter_mb(rank, targets):7.1f} MB (bf16)")

# 2) Check an adapter against the server before routing traffic to it.
SERVER = {"base_model": "org/base-8b-instruct", "max_lora_rank": 16,
          "supported_modules": set(SHAPES)}

def validate(adapter_config):
    errors = []
    if adapter_config["base_model_name_or_path"] != SERVER["base_model"]:
        errors.append("trained on a different base model")
    if adapter_config["r"] > SERVER["max_lora_rank"]:
        errors.append(f"rank {adapter_config['r']} > server max_lora_rank {SERVER['max_lora_rank']}")
    unknown = set(adapter_config["target_modules"]) - SERVER["supported_modules"]
    if unknown:
        errors.append(f"unsupported target modules {sorted(unknown)}")
    return errors

print(validate({"base_model_name_or_path": "org/base-8b-instruct", "r": 16, "target_modules": ["q_proj", "v_proj"]}))
print(validate({"base_model_name_or_path": "org/other-7b", "r": 64, "target_modules": ["q_proj", "lm_head"]}))

# 3) Sizing GPU adapter slots: an LRU over tenants with skewed (Zipf-like) traffic.
def simulate(n_tenants, gpu_slots, cpu_slots, n_requests=50_000, skew=1.1, seed=0):
    rnd = random.Random(seed)
    weights = [1 / (i + 1) ** skew for i in range(n_tenants)]
    gpu, cpu = OrderedDict(), OrderedDict()
    disk_loads = cpu_to_gpu = 0
    for tenant in rnd.choices(range(n_tenants), weights, k=n_requests):
        if tenant in cpu:
            cpu.move_to_end(tenant)                        # CPU cache holds every recently used adapter
        if tenant in gpu:
            gpu.move_to_end(tenant); continue
        if tenant in cpu:
            cpu_to_gpu += 1                                # fast host-to-device copy
        else:
            disk_loads += 1                                # slow: read from disk or object store
            cpu[tenant] = True
            if len(cpu) > cpu_slots:
                cpu.popitem(last=False)
        gpu[tenant] = True
        if len(gpu) > gpu_slots:
            gpu.popitem(last=False)
    return (disk_loads + cpu_to_gpu) / n_requests, disk_loads / n_requests

for gpu_slots, cpu_slots in [(4, 32), (16, 32), (16, 256), (64, 256)]:
    miss, disk = simulate(n_tenants=500, gpu_slots=gpu_slots, cpu_slots=cpu_slots)
    print(f"GPU slots {gpu_slots:2d}, CPU cache {cpu_slots:3d}: adapter not on GPU for {miss:5.1%} "
          f"of requests, loaded from disk for {disk:5.1%}")
```

## Choosing / trade-offs
| Option | Pick when | Watch out for |
|---|---|---|
| Merge each adapter, one deployment per model | one or two fine-tunes with steady high traffic | GPU cost grows with model count; idle models still hold GPUs |
| vLLM `--enable-lora` | you already run vLLM; tens of adapters with known names | slot and rank limits are fixed at startup; runtime loading is an opt-in, unauthenticated admin surface |
| LoRAX | many adapters fetched on demand from a hub or object store by name | separate project with its own release cadence; check model support |
| Hosted fine-tuning | no GPU operations wanted | adapter and serving locked to that provider ([[hosted-fine-tuning-apis]]) |

- **Slots vs memory**: every GPU adapter slot and a higher `--max-lora-rank` take memory that would otherwise hold KV cache, so over-provisioning slots lowers concurrency. Size slots from the active-tenant distribution, not the total tenant count (simulation above), then load test ([[llm-load-testing-and-benchmarking]]).
- **Rank and targets**: higher rank and all-linear targeting can help quality but enlarge adapters and per-token overhead; keep one maximum rank across tenants so they fit the same server.
- **Latency**: a batch mixing many adapters is slower than a base-only batch; measure TTFT and TPOT with your real tenant mix, not with one adapter.

## Gotchas
- An adapter trained on a different revision of the base (or on the quantized base for QLoRA) still loads but quietly degrades; record the base model ID and revision with every adapter and reject mismatches before registration ([[model-registry-and-versioning]]).
- Tokenizer or chat-template changes made during fine-tuning (added special tokens, resized embeddings) do not carry over through an adapter that targets only attention and MLP layers, and the server may not support adapters that change embeddings; check that the server's tokenizer is the one the adapter was trained with ([[post-training-datasets-and-chat-templates]]).
- `--max-lora-rank` smaller than an adapter's `r` makes loading fail; a much larger value wastes memory on every slot.
- Runtime load endpoints read server-side paths; expose them only to an internal deploy job, never through the public gateway ([[ai-security-privacy-compliance]]).
- Tenant isolation is by request field: a gateway bug that lets a caller pick another tenant's adapter name leaks that tenant's fine-tune behaviour. Map tenant identity to adapter name server-side.
- A cold adapter load on the request path shows up as tail latency; preload adapters for active tenants and alert on load counts.
- Flags and endpoint names change between vLLM releases; check the LoRA page of the docs for your version.

## Related
- [[fine-tuning-and-peft]] - how LoRA adapters are trained and when fine-tuning is worth it.
- [[lora-from-scratch-numpy]] - the low-rank update itself, implemented in NumPy.
- [[inference-servers-vllm]] - the server, its batching and KV cache that adapters share.
- [[vllm-local-setup]] - installing and sizing vLLM on one GPU.
- [[unsloth-finetuning]] - producing LoRA adapters cheaply on one GPU.
- [[hosted-fine-tuning-apis]] - letting a provider train and serve the adapter instead.
- [[model-merging]] - combining adapters into one set of weights instead of serving them side by side.
- [[llm-gateways-and-routing]] - mapping tenants and tasks to adapter names.
- [[llm-load-testing-and-benchmarking]] - measuring latency with a realistic adapter mix.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/caching-strategies.md - LRU eviction and hit rates behind adapter slot sizing.

## References
- vLLM LoRA adapters (serving flags, `LoRARequest`, runtime loading): https://docs.vllm.ai/en/latest/features/lora/
- Sheng et al., S-LoRA: Serving Thousands of Concurrent LoRA Adapters (2023): https://arxiv.org/abs/2311.03285
- Chen et al., Punica: Multi-Tenant LoRA Serving (2023): https://arxiv.org/abs/2310.18547
- LoRAX (multi-LoRA inference server): https://github.com/predibase/lorax
- Hugging Face PEFT LoRA guide (`merge_and_unload`, adapter config): https://huggingface.co/docs/peft/developer_guides/lora
