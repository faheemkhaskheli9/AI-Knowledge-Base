---
title: Kubernetes for LLM serving (KServe, KubeRay, GPU autoscaling)
category: deployment
tags: [kubernetes, kserve, kuberay, ray-serve, keda, gpu-scheduling, autoscaling, cold-start, vllm, llm-d]
use_cases:
  - "deploy a self-hosted LLM on our Kubernetes cluster with GPU nodes"
  - "autoscale vLLM replicas on request queue depth instead of CPU"
  - "cut cold-start time when a new LLM pod has to download 100+ GB of weights"
  - "choose between KServe, KubeRay/Ray Serve and a plain Deployment for model serving"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/kserve/kserve
  - https://kserve.github.io/website/
  - https://docs.ray.io/en/latest/cluster/kubernetes/examples/rayserve-llm-example.html
  - https://docs.ray.io/en/latest/serve/llm/index.html
  - https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html
  - https://keda.sh/docs/latest/scalers/prometheus/
  - https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
  - https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
---

# Kubernetes for LLM serving (KServe, KubeRay, GPU autoscaling)

## Summary
Running LLM servers (vLLM, SGLang, TensorRT-LLM) on Kubernetes gives you rollouts, health checks, multi-tenant GPU pools and autoscaling, but LLM pods are unlike web pods. Each needs whole GPUs, takes minutes to start because weights are large, and saturates on KV-cache memory and queue length rather than CPU. KServe and KubeRay/Ray Serve are the main serving layers on top of Kubernetes. KEDA (or HPA on custom metrics) scales replicas on queue depth.

## Key concepts
- **GPU scheduling**: a device plugin (usually installed by the NVIDIA GPU Operator) advertises `nvidia.com/gpu` as an extended resource. Pods request it in `resources.limits`. GPUs are not overcommitted or shared by default. Sharing options are MIG partitions, time-slicing, or Dynamic Resource Allocation (DRA) on newer clusters. Use node labels/taints so only GPU workloads land on GPU nodes.
- **KServe** (CNCF incubating): `InferenceService` CRD with a Hugging Face runtime that uses vLLM for LLMs by default and exposes an OpenAI-compatible protocol, plus model caching, KV-cache offloading and llm-d integration. It runs in Knative (serverless, scale-to-zero, request-based autoscaling) or standard (raw Deployment) mode.
- **KubeRay + Ray Serve LLM**: a `RayService` CRD runs a Ray cluster. Ray Serve LLM (`ray.serve.llm:build_openai_app` with `llm_configs`) serves vLLM-backed models behind an OpenAI API and handles multi-node models, multiple models per app and Ray-native autoscaling.
- **Plain Deployment**: one `Deployment` running `vllm serve`, a `Service`, and an autoscaler. This is often enough for one or two models.
- **Scale on queue depth**: CPU% is meaningless for GPU servers. Scale on server metrics such as vLLM's `vllm:num_requests_waiting` (or running requests, KV-cache usage) scraped by Prometheus, through KEDA's Prometheus scaler or HPA custom metrics. HPA's rule is `desired = ceil(current x metric / target)`.
- **Cold start**: image pull (multi-GB CUDA images) + weight download + load to GPU + warmup / CUDA graph capture. This takes minutes for big models, so autoscaling must react early and scale-to-zero costs a slow first request.
- **LLM-aware routing**: round-robin ignores KV-cache state. Prefix- or load-aware routers (llm-d, the vLLM production-stack router, the Gateway API inference extension) send requests to the replica most likely to hit cache or least loaded.

## When to use / scenarios
- A company already on Kubernetes running private LLMs for several internal apps, wanting shared GPU pools, quotas and GitOps: KServe or plain Deployments + KEDA.
- Multi-node models (tensor + pipeline parallel across nodes), many models in one app, or teams already using Ray for training/data: KubeRay + Ray Serve LLM.
- Mixed classic ML + LLM serving (scikit-learn, XGBoost, ONNX next to LLMs) under one API: KServe predictive + generative runtimes; see [[onnx-and-triton-inference-server]].
- Not worth it for a single GPU box with one model (Docker Compose + [[inference-servers-vllm]]), or when a managed endpoint fits ([[managed-cloud-ai-platforms]], [[gpu-cloud-options]]).

## Setup & code
KServe `InferenceService` with the Hugging Face runtime (vLLM backend). Check `args` and the runtime image against your KServe release:
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: chat-llm
spec:
  predictor:
    minReplicas: 1                     # avoid scale-to-zero cold starts for user-facing traffic
    model:
      modelFormat:
        name: huggingface
      args: ["--model_name=chat-llm", "--model_id=<hf-org>/<model>"]
      env:
        - name: HF_TOKEN               # only for gated models
          valueFrom: {secretKeyRef: {name: hf-secret, key: HF_TOKEN}}
      resources:
        limits:   {nvidia.com/gpu: "1", memory: 48Gi}
        requests: {nvidia.com/gpu: "1", memory: 48Gi}
```
KEDA `ScaledObject` scaling a plain vLLM Deployment on queue depth (pattern from the vLLM production-stack KEDA guide):
```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
spec:
  scaleTargetRef: {name: vllm-deployment}
  minReplicaCount: 1
  maxReplicaCount: 4
  pollingInterval: 15
  cooldownPeriod: 360                  # slow scale-down: new pods are expensive to start
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus-operated.monitoring.svc:9090
        query: sum(vllm:num_requests_waiting{model_name="<model>"})
        threshold: "5"
```
Ray Serve LLM on KubeRay: set `import_path: ray.serve.llm:build_openai_app` and `args.llm_configs` in the `RayService` `serveConfigV2`; see the KubeRay LLM example in References for the full manifest.

Why scale on queue depth and early: a stdlib simulation of HPA-style scaling with a cold start. Traffic doubles at t=60 s, and new replicas only serve after the startup delay.
```python
import math

CAP = 8            # requests/s one replica can drain
COLD_START = 120   # s from scale-up decision to ready (image pull + weights + warmup)
TARGET = 5         # target waiting requests per replica

def simulate(cold_start, minutes=8):
    ready, pending, queue, worst = 2, [], 0.0, 0.0
    for t in range(minutes * 60):
        ready += sum(1 for s in pending if s <= t)            # pods finishing startup
        pending = [s for s in pending if s > t]
        rate = 10 if t < 60 else 22                            # requests/s arriving
        queue = max(0.0, queue + rate - ready * CAP)
        worst = max(worst, queue)
        if t % 15 == 0:                                        # KEDA polling interval
            per_pod = queue / ready
            desired = min(6, math.ceil(ready * per_pod / TARGET)) if per_pod > TARGET else ready
            for _ in range(desired - ready - len(pending)):
                pending.append(t + cold_start)
    return ready, worst

for cs in (0, 30, COLD_START):
    replicas, worst = simulate(cs)
    print(f"cold start {cs:>3}s -> final replicas {replicas}, worst queue {worst:,.0f} requests")
```
With a 120 s cold start the queue grows for two minutes before help arrives. Keep spare replicas (`minReplicaCount`), set a lower threshold, or scale on a leading signal (arrival rate) as well.

## Choosing / trade-offs
| Option | Good for | Cost |
|---|---|---|
| Deployment + Service + KEDA | 1-3 models, full control, simplest | you build routing, rollout, caching yourself |
| KServe | many models, standard CRD, scale-to-zero, canaries, mixed ML + LLM | Knative/Istio or gateway dependencies, more moving parts |
| KubeRay + Ray Serve LLM | multi-node models, multi-model apps, Ray-based teams | Ray cluster to operate and debug |
| Managed endpoints | no cluster ops | less control, per-hour or per-token pricing |

- **Scale-to-zero vs latency**: zero idle GPU cost, but the first request waits minutes. Use it for dev and batch, not user-facing chat.
- **Whole GPUs vs sharing**: MIG/time-slicing raises utilization for small models but adds noisy-neighbor latency; LLMs above a few B parameters usually get dedicated GPUs.
- **Bigger pods vs more pods**: tensor parallel across 2-8 GPUs serves bigger models and lowers latency; more single-GPU replicas give more throughput per GPU for small models.

## Gotchas
- Readiness probes must wait for the model to load (vLLM `/health`), or traffic hits pods that time out. Set a generous `startupProbe`.
- Model weights in the container image make images huge. Prefer a shared PVC, a node-local cache (KServe model cache) or pre-pulled images, and pin the model revision.
- `/dev/shm` defaults are tiny; tensor-parallel servers need a larger shared-memory `emptyDir` (medium: Memory).
- Scale-down kills in-flight streams. Use a long `terminationGracePeriodSeconds`, a preStop drain and a slow cooldown.
- A Prometheus query that returns nothing makes KEDA see 0 and scale in. Verify the metric name for your server version (the vLLM docs examples differ between `sum()` and per-pod queries).
- GPU nodes from cluster autoscalers take minutes too; pod cold start + node cold start add up.
- Spot/preemptible GPU nodes are cheap but evict; keep at least one on-demand replica for user-facing traffic.

## Related
- [[disaggregated-prefill-and-kv-cache-offloading]] - llm-d and Dynamo: cache-aware routing, P/D disaggregation and KV tiers.
- [[inference-servers-vllm]] - the server running in each pod.
- [[sglang-and-tensorrt-llm]] - alternative engines with the same deployment shape.
- [[onnx-and-triton-inference-server]] - Triton/KServe for non-LLM models.
- [[batch-inference-and-batch-apis]] - offline jobs as Kubernetes Jobs instead of always-on pods.
- [[gpu-cloud-options]] - GPU node pools and pricing models.
- [[llm-gateways-and-routing]] - the API layer in front of the cluster.
- [[llm-observability]] - the metrics autoscaling depends on.
- [[mlops-lifecycle]] - rollout and model versioning.
- [[model-weight-loading-and-cold-starts]] - shared model caches, streaming loaders and startup probes.
- [[ray-serve-bentoml-and-litserve]] - Ray Serve (on KubeRay), BentoML and LitServe compared.
- [[inference-server-and-gpu-metrics]] - the server and DCGM GPU metrics behind autoscaling and alerts.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/kubernetes-basics.md - Kubernetes objects and probes.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/capacity-planning-and-autoscaling.md - autoscaling theory and headroom.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/deployment-strategies.md - canary and blue-green rollouts of a new model.

## References
- KServe repository (features, deployment modes): https://github.com/kserve/kserve
- KServe docs: https://kserve.github.io/website/
- KubeRay, Serve an LLM with Ray Serve LLM on Kubernetes: https://docs.ray.io/en/latest/cluster/kubernetes/examples/rayserve-llm-example.html
- Ray Serve LLM docs: https://docs.ray.io/en/latest/serve/llm/index.html
- vLLM production stack, autoscaling with KEDA: https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html
- KEDA Prometheus scaler: https://keda.sh/docs/latest/scalers/prometheus/
- Kubernetes, Schedule GPUs: https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
- Kubernetes HPA algorithm: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/
