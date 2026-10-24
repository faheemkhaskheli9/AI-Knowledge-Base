---
title: Disaggregated prefill/decode and KV cache offloading (vLLM connectors, LMCache, llm-d, Dynamo)
category: deployment
tags: [disaggregated-serving, prefill-decode-disaggregation, pd-disaggregation, kv-cache, kv-offloading, lmcache, nixl, llm-d, dynamo, vllm, prefix-caching, ttft, itl]
use_cases:
  - "stop long prompts from causing inter-token latency spikes for users already streaming"
  - "tune time to first token and inter-token latency separately on a self-hosted LLM"
  - "keep the KV cache of long multi-turn chats or agent sessions so the next turn skips prefill"
  - "decide whether we need prefill/decode disaggregation or just chunked prefill and prefix caching"
status: draft
last_verified: 2026-10-24
sources:
  - https://docs.vllm.ai/en/latest/features/disagg_prefill/
  - https://docs.vllm.ai/en/latest/features/kv_offloading_usage/
  - https://docs.vllm.ai/en/latest/features/nixl_connector_usage/
  - https://github.com/LMCache/LMCache
  - https://github.com/llm-d/llm-d
  - https://github.com/ai-dynamo/dynamo
  - https://arxiv.org/abs/2401.09670
  - https://arxiv.org/abs/2311.18677
  - https://arxiv.org/abs/2407.00079
---

# Disaggregated prefill/decode and KV cache offloading (vLLM connectors, LMCache, llm-d, Dynamo)

## Summary
LLM inference has two phases with different bottlenecks: prefill processes the whole prompt at once (compute-bound, sets time to first token) and decode emits one token per step (memory-bandwidth-bound, sets inter-token latency). On one server they compete: a long prefill scheduled between decode steps stalls every user who is already streaming. Disaggregated serving runs prefill and decode on separate instances and ships the KV cache from one to the other through a connector; KV cache offloading moves finished KV blocks from GPU memory to CPU, disk or remote storage so a later request with the same prefix loads them instead of recomputing. In vLLM both are `--kv-transfer-config` connectors (NixlConnector, LMCache, Mooncake, OffloadingConnector and others); llm-d and NVIDIA Dynamo add the routing and orchestration layer on top. vLLM's own docs mark disaggregated prefill experimental and state that it does not improve throughput: it is a latency-control tool.

## Key concepts
- **Prefill vs decode**: prefill cost grows with prompt length and is compute-bound; decode cost per step is dominated by reading weights and KV cache ([[kv-cache-from-scratch-numpy]]). TTFT is mostly prefill; TPOT/ITL is decode ([[llm-load-testing-and-benchmarking]]).
- **Interference**: in a co-located server the scheduler mixes new prefills into running decode batches, so tail ITL rises when long prompts arrive. Chunked prefill limits this by splitting prefills; vLLM's docs note the right chunk size is hard to find in practice, which is why they call disaggregation the more reliable way to control tail ITL.
- **P/D disaggregation**: a prefill (producer, `kv_role: kv_producer`) instance computes the prompt's KV cache and the first token; a decode (consumer, `kv_consumer`) instance receives the KV and continues. A proxy or router sends each request to both. Each side can use its own parallelism (`tp`, `pp`) and its own replica count: tune TTFT on the prefill pool, ITL on the decode pool.
- **Connector**: vLLM's plug-in that moves KV blocks between instances or tiers. The docs list NixlConnector (async send/recv over NIXL with UCX, LIBFABRIC, GDS and other backends), LMCacheConnectorV1 and LMCacheMPConnector, MooncakeConnector, MoRIIOConnector (ROCm only), FlexKVConnectorV1, OffloadingConnector, MultiConnector (chains several) and an ExampleConnector.
- **KV transfer cost**: KV bytes per token = 2 (K and V) x layers x KV heads x head dim x bytes per element. GQA, MLA and FP8 KV caches shrink it. A long prompt's KV cache is large, so disaggregation needs a fast interconnect (NVLink, RDMA); over a slow network the transfer can cost more than it saves.
- **KV cache offloading**: the GPU prefix cache ([[prompt-caching-and-cost]]) only holds what fits beside the weights. vLLM's OffloadingConnector copies completed KV blocks to pinned CPU memory (and optionally a filesystem, S3-compatible object store or peer-to-peer tier); a later hit is loaded back to the GPU instead of recomputed. It supports CUDA, ROCm and XPU.
- **Cache-aware routing**: offloaded or prefix-cached KV only helps if the next request with that prefix lands where the KV is. llm-d (Kubernetes, CNCF sandbox) and Dynamo route on prefix-cache state and load; plain round-robin load balancing throws most reuse away.
- **Engine-independent cache**: LMCache runs as a KV cache layer (optionally a separate process) across CPU, disk and remote backends, can share KV between engine instances and also does P/D KV transfer over NIXL.

## When to use / scenarios
- Chat or agent products with an ITL/TPOT SLO where p99 ITL jumps whenever long prompts (RAG context, documents, agent histories) arrive, and chunked prefill tuning has not fixed it.
- Very large models served across many GPUs where prefill and decode want different parallelism (for example wide expert parallelism for MoE decode), the setting llm-d and Dynamo target.
- Multi-turn chat, coding agents and long-running agents ([[long-running-agents]]) whose context grows each turn: KV offloading to CPU or disk keeps earlier turns' KV past GPU eviction, so the next turn's TTFT drops.
- Many requests sharing long system prompts or documents across replicas: a shared offload tier or LMCache lets replicas reuse each other's prefill.
- Not for raising throughput on a saturated cluster: vLLM's docs say disaggregated prefill does not improve throughput. Quantize, add replicas or batch first ([[quantizing-llms-for-serving]], [[batch-inference-and-batch-apis]]).
- Not for one GPU or short prompts: keep one co-located vLLM with prefix caching and chunked prefill ([[inference-servers-vllm]]).
- Not for hosted APIs: providers do this internally; use their prompt caching instead ([[prompt-caching-and-cost]]).

## Setup & code
vLLM 1 prefill + 1 decode on one host with NixlConnector (from the vLLM NixlConnector usage guide; needs two GPUs and NIXL installed):
```bash
# Prefiller (producer) on GPU 0
CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve <model> --port 8100 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
# Decoder (consumer) on GPU 1
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
vllm serve <model> --port 8200 --enforce-eager \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
# Toy proxy from the vLLM repo: sends each request to the prefiller, then the decoder
python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192 \
  --prefiller-hosts localhost --prefiller-ports 8100 --decoder-hosts localhost --decoder-ports 8200
```
KV cache offloading to CPU memory on a single (non-disaggregated) instance; `cpu_bytes_to_use` is the total across workers, and the vLLM tuning tips say to make it larger than the aggregate GPU KV cache or it only mirrors what the GPU holds:
```bash
vllm serve <model> --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"block_size":64,"cpu_bytes_to_use":1000000000}}'
# Multi-tier: CPU primary tier plus a filesystem tier (shared PVC or local NVMe)
vllm serve <model> --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":10737418240,
  "secondary_tiers":[{"type":"fs","root_dir":"/mnt/kv_cache"}]}}'
```
Runnable, stdlib only: (1) KV cache size per token and the time to ship one prompt's KV at a given link speed, (2) a toy scheduler showing why co-located prefills raise decode tail latency and what disaggregation changes, (3) an LRU prefix cache with and without a CPU offload tier on a multi-turn workload. Model shapes, step costs and link speeds below are illustrative assumptions, not measurements.
```python
# 1) KV bytes per token and transfer time for one prompt (illustrative shapes, not a specific model).
def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_elem):
    return 2 * layers * kv_heads * head_dim * bytes_per_elem      # K and V

shapes = {"MHA, 32 layers, 32 KV heads, fp16": (32, 32, 128, 2),
          "GQA, 32 layers, 8 KV heads, fp16":  (32, 8, 128, 2),
          "GQA, 32 layers, 8 KV heads, fp8":   (32, 8, 128, 1)}
prompt = 8000
for name, s in shapes.items():
    per_tok = kv_bytes_per_token(*s)
    total = per_tok * prompt
    for link, gbytes_s in [("fast RDMA/NVLink-class link", 50.0), ("slow TCP link", 1.0)]:
        ms = total / (gbytes_s * 1e9) * 1e3
        print(f"{name:36s} {per_tok/1024:6.0f} KiB/token  {total/2**20:7.0f} MiB per {prompt}-token prompt"
              f"  {ms:7.1f} ms over {link} ({gbytes_s:g} GB/s)")

# 2) Toy scheduler: decode steps for 8 streaming users while long prompts arrive.
#    Assumed costs (arbitrary units): one decode step 1.0; a prefill adds 0.002 per prompt token to the step it joins.
import random
random.seed(0)
arrivals = {t: random.choice([500, 4000, 16000]) for t in random.sample(range(400), 40)}   # step -> prompt length

def itl_samples(colocated, chunk=None):
    out, pending = [], []
    for step in range(400):
        if step in arrivals:
            pending.append(arrivals[step])
        cost = 1.0
        if colocated and pending:
            take = pending[0] if chunk is None else min(chunk, pending[0])  # whole prefill or one chunk per step
            cost += 0.002 * take
            pending[0] -= take
            if pending[0] <= 0:
                pending.pop(0)
        out.append(cost)                           # every streaming user waits this long for the next token
    return sorted(out)

def p(xs, q):
    return xs[min(len(xs) - 1, int(q * len(xs)))]

for label, xs in [("co-located, whole prefills", itl_samples(True)),
                  ("co-located, chunked prefill 2048", itl_samples(True, 2048)),
                  ("co-located, chunked prefill 512", itl_samples(True, 512)),
                  ("disaggregated (prefill elsewhere)", itl_samples(False))]:
    print(f"{label:36s} ITL p50={p(xs, .5):5.2f}  p99={p(xs, .99):5.2f}  max={xs[-1]:5.2f}")
# Smaller chunks cut the tail but spread prefill work over more steps, so the median rises once
# prefills queue up: the chunk-size tuning problem the vLLM docs describe.

# 3) Multi-turn sessions: GPU-only prefix cache vs GPU + CPU offload tier (LRU, capacity in blocks).
from collections import OrderedDict

class LRU:
    def __init__(self, cap):
        self.cap, self.d = cap, OrderedDict()
    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k); return True
        return False
    def put(self, k):
        self.d[k] = True; self.d.move_to_end(k)
        if len(self.d) > self.cap:
            return self.d.popitem(last=False)[0]    # evicted key
        return None

def run(gpu_blocks, cpu_blocks):
    gpu, cpu = LRU(gpu_blocks), LRU(cpu_blocks) if cpu_blocks else None
    hit = miss = 0
    rng = random.Random(1)
    turns = {s: 0 for s in range(60)}              # 60 concurrent chat sessions
    for _ in range(3000):
        s = rng.randrange(60)
        turns[s] += 1
        for b in range(turns[s] * 4):              # each turn's prompt = whole history, 4 blocks per turn
            key = (s, b)
            if gpu.get(key) or (cpu and cpu.get(key)):
                hit += 1
            else:
                miss += 1
            ev = gpu.put(key)
            if ev is not None and cpu:
                cpu.put(ev)                        # offload instead of dropping
    return hit / (hit + miss)

print(f"prefix-cache hit rate, GPU only (2000 blocks):        {run(2000, 0):.2f}")
print(f"prefix-cache hit rate, GPU 2000 + CPU tier 20000:     {run(2000, 20000):.2f}")
```

## Choosing / trade-offs
| Situation | Start with | Move to |
|---|---|---|
| One node, ITL spikes from long prompts | chunked prefill + prefix caching in one vLLM | P/D disaggregation on the same node (NixlConnector) |
| Multi-turn or agent traffic evicting the GPU prefix cache | OffloadingConnector to CPU | multi-tier (fs/object store) or LMCache shared across replicas |
| Many replicas, shared long prefixes | prefix-cache-aware routing (llm-d, Dynamo, or gateway affinity) | shared KV tier plus cache-aware routing |
| Very large MoE across nodes, strict TTFT and ITL SLOs | llm-d or Dynamo recipes for your engine and hardware | custom prefill/decode ratios per workload |

- **Ratio of prefill to decode instances**: depends on prompt and output lengths of your traffic; measure TTFT and ITL per pool and rebalance. Long prompts with short answers need more prefill; long generations need more decode.
- **Interconnect**: disaggregation and P2P tiers assume RDMA or NVLink-class links; on plain TCP the KV transfer can dominate TTFT. Run the arithmetic in the code above with your model's shapes.
- **Complexity**: two pools, a proxy or router, NIXL/UCX networking and connector versions that move fast, against one `vllm serve`. Take it on when an SLO, not a benchmark headline, needs it.
- **Vendor stack**: Dynamo is NVIDIA's orchestration layer over vLLM, SGLang and TensorRT-LLM ([[sglang-and-tensorrt-llm]]); llm-d is a Kubernetes-native stack built around vLLM with tested guides across accelerators ([[kubernetes-for-llm-serving]]).

## Gotchas
- Disaggregated prefill in vLLM is experimental and its flags change between releases; pin the vLLM and connector versions together and re-read the docs on upgrade.
- It does not raise throughput (vLLM docs). Teams that adopt it to "serve more users" often get the same tokens per second with more GPUs and more moving parts.
- Vendor and project benchmark headlines are measured on specific models, hardware and traffic; replay your own prompts at production concurrency before believing a gain ([[llm-load-testing-and-benchmarking]]).
- A CPU offload tier smaller than the GPU KV cache adds no hit rate, because offloading is immediate and the CPU tier just mirrors the GPU (vLLM tuning tips).
- Sharing an offload directory or bucket across instances needs identical block hashes: with the `xxhash` prefix-caching hash algorithms the seed is random per process, so set the same `PYTHONHASHSEED` on every instance; changing model, block size, parallelism or dtype writes to a new subdirectory.
- The P2P tier's control socket binds `localhost` by default; cross-host P2P fails until `VLLM_P2P_SIDE_CHANNEL_HOST` is set to the node's routable IP.
- Without cache-aware routing, offloaded KV sits on a replica the next turn never reaches; check hit rates per replica, not just globally.
- Persisted KV is derived from user prompts: treat offload disks and buckets as holding the same sensitive data as the prompts (encryption, retention, tenant isolation; [[ai-security-privacy-compliance]]).

## Related
- [[inference-servers-vllm]] - the server whose connectors implement this.
- [[kv-cache-from-scratch-numpy]] - what the KV cache is and why decode is memory-bound.
- [[prompt-caching-and-cost]] - prefix caching, the single-instance form of KV reuse.
- [[llm-load-testing-and-benchmarking]] - measuring TTFT, ITL and goodput before and after.
- [[kubernetes-for-llm-serving]] - llm-d and other Kubernetes serving stacks.
- [[sglang-and-tensorrt-llm]] - engines Dynamo orchestrates, with their own disaggregation support.
- [[speculative-decoding-in-serving]] - the other main lever for decode latency.
- [[cost-and-latency]] - where this sits among latency optimizations.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/caching-strategies.md - cache tiers, eviction and hit-rate thinking.

## References
- vLLM disaggregated prefilling (motivation, "does not improve throughput", connector list): https://docs.vllm.ai/en/latest/features/disagg_prefill/
- vLLM KV offloading usage guide (OffloadingConnector, tiers, tuning tips, cross-process sharing): https://docs.vllm.ai/en/latest/features/kv_offloading_usage/
- vLLM NixlConnector usage guide (1P1D example, proxy): https://docs.vllm.ai/en/latest/features/nixl_connector_usage/
- LMCache (KV cache layer, tiered offloading, P/D transfer): https://github.com/LMCache/LMCache
- llm-d (Kubernetes distributed inference, cache-aware routing, tiered prefix cache, P/D): https://github.com/llm-d/llm-d
- NVIDIA Dynamo (orchestration over vLLM, SGLang, TensorRT-LLM; disaggregated serving, KV-aware routing): https://github.com/ai-dynamo/dynamo
- Zhong et al., DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving: https://arxiv.org/abs/2401.09670
- Patel et al., Splitwise: Efficient Generative LLM Inference Using Phase Splitting: https://arxiv.org/abs/2311.18677
- Qin et al., Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving: https://arxiv.org/abs/2407.00079
