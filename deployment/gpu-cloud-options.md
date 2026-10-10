---
title: GPU cloud options (hyperscalers, GPU clouds, serverless)
category: deployment
tags: [gpu-cloud, aws, azure, gcp, serverless-inference, spot, runpod, modal, capacity]
use_cases:
  - "rent a GPU for a few hours to fine-tune a model"
  - "choose between AWS/Azure/GCP, a specialist GPU cloud and a hosted open-model API"
  - "deploy a model that scales to zero when idle"
  - "plan GPU capacity for a production LLM service"
status: draft
last_verified: 2026-10-03
sources:
  - https://aws.amazon.com/ec2/instance-types/
  - https://cloud.google.com/compute/docs/gpus
  - https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview
---

# GPU cloud options (hyperscalers, GPU clouds, serverless)

## Summary
You can get GPU compute from three kinds of providers: hyperscalers (AWS, Azure, Google Cloud), specialist GPU clouds (e.g. RunPod, Lambda, CoreWeave, Vast-style marketplaces), and serverless or hosted-inference platforms (e.g. Modal, Replicate, Together, Fireworks, cloud-vendor model catalogs). Offerings, GPU models and prices change often and are not listed here; check each provider's current catalog.

## Key concepts
- On-demand vs reserved/committed vs spot/preemptible: price falls and interruption risk rises in that direction (spot can be reclaimed).
- Instance = GPU type x count x CPU/RAM/disk/network; memory per GPU decides which models fit ([[gpu-cuda-setup]]).
- Serverless GPU: pay per second/request, cold starts on first request, scale to zero.
- Hosted inference API: no infrastructure, you pay per token for catalog open models.
- Quotas: hyperscalers often require a quota increase request for GPU instances.
- Data gravity and egress: moving large datasets and checkpoints costs time and money.

## When to use / scenarios
- Short fine-tuning job (QLoRA on a 7B-13B model): hourly rental from a GPU cloud or spot instance, with checkpoints saved off-instance ([[fine-tuning-and-peft]]).
- Enterprise with compliance and existing cloud contracts: hyperscaler in the required region, private networking.
- Spiky or low-volume inference: serverless GPU or hosted API instead of an idle GPU ([[cost-and-latency]]).
- Steady high-volume inference: reserved/dedicated GPUs running vLLM ([[inference-servers-vllm]]).
- Learning/experiments: notebook platforms or a local GPU.

## Setup & code
A generic rented-GPU workflow (any provider):
```bash
ssh <user>@<gpu-host>
nvidia-smi                                   # confirm GPU, driver
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <your-repo> && cd <repo>
uv sync --frozen && uv run python train.py   # write checkpoints to persistent storage
```
Habits that save money:
```bash
# run inside tmux/screen so a dropped SSH session does not kill the job
tmux new -s train
```
- Push checkpoints and logs to object storage or a persistent volume regularly; instances may be preempted or deleted.
- Shut down or delete instances when done; set budget alerts.
- Bake dependencies into a container image to cut start-up time.
- Use the provider's CLI/SDK or Terraform for reproducibility; flags differ per provider, so follow its docs.

Decision shortcut:
- Need it for hours, tolerant of interruptions: spot or GPU cloud.
- Need compliance, SLAs, managed networking: hyperscaler.
- Need to avoid ops and traffic is bursty: serverless/hosted API.
- Need it 24/7 at predictable load: reserved capacity or own hardware.

## Choosing / trade-offs
- Hyperscalers: broadest services and compliance certifications, quotas and higher complexity; GPU supply can be tight.
- Specialist GPU clouds: simpler and often cheaper per GPU-hour, fewer managed services and varying reliability/compliance.
- Serverless: zero idle cost, cold starts and per-platform packaging; at sustained load it can cost more than dedicated GPUs.
- Hosted APIs: fastest to ship, least control over model, versions and data handling.
- Own hardware: lowest long-run cost at constant use, upfront capital and maintenance.

## Gotchas
- A cheap per-hour GPU idle for a weekend can cost more than the job itself; automate shutdown.
- Spot reclaim mid-training without checkpoints loses the run.
- Disk is usually ephemeral by default; confirm what persists.
- Community/marketplace hosts may be unsuitable for sensitive data; check data-processing terms ([[ai-security-privacy-compliance]]).
- Region matters for latency, data residency and GPU availability.
- Pricing pages change; do not hard-code assumptions in design docs.

## Related
- [[gpu-cuda-setup]] - VRAM sizing.
- [[inference-servers-vllm]] - what to run on the GPU.
- [[cost-and-latency]] - break-even versus hosted APIs.
- [[fine-tuning-and-peft]] - training workloads.
- [[mlops-lifecycle]] - reproducible deployments.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/capacity-planning-and-autoscaling.md - how many GPUs and when to scale.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/reliability/kubernetes-basics.md - scheduling GPU workloads on a cluster.

## References
- AWS EC2 instance types: https://aws.amazon.com/ec2/instance-types/
- Google Cloud GPUs: https://cloud.google.com/compute/docs/gpus
- Azure VM sizes: https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/overview
- Specialist and serverless providers: see each vendor's own documentation (not verified here).
