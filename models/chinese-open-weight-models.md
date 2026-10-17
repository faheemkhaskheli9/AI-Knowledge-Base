---
title: Chinese open-weight models (Kimi, GLM, MiniMax and others)
category: models
tags: [kimi, moonshot, glm, zhipu, z-ai, minimax, open-weights, moe, licences, china, self-hosting]
use_cases:
  - "self-host a frontier-class open-weight model for an internal coding agent"
  - "compare Kimi, GLM and MiniMax licences before building a commercial product"
  - "choose an open MoE model with 1M-token context for long-document work"
  - "decide between a vendor API and self-hosting for a Chinese open model"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/MoonshotAI/Kimi-K3
  - https://github.com/MoonshotAI/Kimi-K3/blob/main/LICENSE
  - https://github.com/zai-org/GLM-5
  - https://github.com/MiniMax-AI/MiniMax-M3
  - https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/LICENSE
  - https://thenewstack.io/zai-glm-weights-license/
---

# Chinese open-weight models (Kimi, GLM, MiniMax and others)

## Summary
Several Chinese labs release frontier-scale open-weight mixture-of-experts
models, mostly aimed at agentic coding, tool use and long context: Moonshot AI
(Kimi), Zhipu / Z.ai (GLM) and MiniMax, alongside Alibaba's [[qwen]] and
[[deepseek]], which have their own files. Weights are downloadable and run on
vLLM/SGLang, but licences differ sharply, from MIT to "modified MIT" with
revenue clauses to non-commercial-without-permission. Read the licence of the
exact checkpoint before you build on it.

## Key concepts
Families as of 2026-10 (from the vendors' GitHub repos unless noted):

| Model | Size (total / active) | Context, modality | Licence |
|---|---|---|---|
| Kimi K3 (Moonshot) | 2.8T / ~104B; MoE, 896 experts, 16 routed | 1M tokens; text + image (video understanding mentioned) | Kimi K3 License (MIT-based, see below) |
| GLM-5.3 (Z.ai) | 744B / 40B; same base as GLM-5.2, new post-training | (GLM-5.2: 1M) | custom licence per news reports, not MIT; verify on the card |
| GLM-5.3-Flash (Z.ai) | 320B / 18B; new base, hybrid sparse + linear attention | multimodal per reports | MIT per news reports; verify on the card |
| GLM-5 / 5.1 / 5.2 | 744B / 40B | GLM-5.2: 1M | MIT per news reports |
| MiniMax M3 | ~428B / ~23B | 1M tokens; text, image, video | licence file on Hugging Face, not retrieved |
| MiniMax M2.7 | not stated in README | - | modified MIT: commercial use needs prior written permission |

Licence details that change decisions:
- **Kimi K3 License**: MIT-style grant plus (a) a separate agreement before
  commercial use if you run a model-as-a-service business with over US$20M
  revenue in any 12 months, and (b) "Kimi K3" shown prominently in the UI for
  products above 100M MAU or US$20M monthly revenue. Internal use is exempt.
- **MiniMax M2.7**: free for personal, research and non-profit use; commercial
  use needs written authorisation and a "Built with MiniMax M2.7" credit;
  includes a prohibited-use list (e.g. military use).
- **GLM-5.3**: reports say large companies (over US$10B revenue) hosting it
  need a Z.ai security review first; GLM-5.3-Flash stayed MIT. I could not
  load the Hugging Face card in this session.

Other shared traits:
- Thinking/effort controls: Kimi K3 thinking is always on with
  `reasoning_effort` low/high/max (default max), and multi-turn calls must pass
  back `reasoning_content`; GLM-5.3 accepts low/high/max; MiniMax M3 has
  `thinking` enabled/adaptive/disabled.
- Native low precision: Kimi K3 is trained with MXFP4 weights (quantization-
  aware); GLM ships FP8 and BF16 repos.
- Others exist (Baidu, Tencent, StepFun, Xiaomi and more); I did not verify
  their current releases here.

## When to use / scenarios
- Self-hosted agentic coding or tool-using assistants where API data egress is
  not allowed and a permissive or internal-use licence fits.
- Long-context work (1M-token windows) such as codebase or contract analysis,
  with the usual caveats from [[long-context]].
- Cost-sensitive API use: these vendors and third-party hosts price below most
  US frontier APIs (check current price pages).
- Chinese-language products.
- Not for: teams whose compliance rules forbid China-hosted APIs (self-host
  instead, or pick another vendor; see [[ai-security-privacy-compliance]]), or
  products that cannot meet a licence's revenue or attribution terms.

## Setup & code
Hosting options:
- **Vendor APIs**: Kimi at platform.kimi.ai (OpenAI- and Anthropic-compatible,
  model `kimi-k3`); Z.ai API platform for GLM-5.3 and GLM-5.3-Flash; MiniMax
  API platform (platform.minimax.io).
- **Third-party hosts and clouds**: GLM appears in Mistral's catalogue
  ([[mistral]]); routers such as OpenRouter and GPU clouds also host these
  models ([[llm-gateways-and-routing]], [[gpu-cloud-options]]). MiniMax and
  Moonshot publish "vendor verifier" tools to check third-party hosts serve the
  model correctly.
- **Self-host**: vLLM and SGLang for all three (plus TokenSpeed for Kimi K3;
  KTransformers and Unsloth for GLM and MiniMax M3; Ascend NPUs for GLM via
  vLLM-Ascend). Use the vendor's recipe for tool-call and reasoning parsers.

A runnable licence gate that encodes the conditions above (illustrative, not
legal advice):
```python
from dataclasses import dataclass

@dataclass
class Deployment:
    commercial: bool
    model_as_a_service: bool      # third parties call the model via your API
    annual_maas_revenue_usd: float
    monthly_active_users: int
    monthly_revenue_usd: float

def licence_flags(model: str, d: Deployment) -> list[str]:
    flags = []
    if model == "kimi-k3":
        if d.model_as_a_service and d.annual_maas_revenue_usd > 20e6:
            flags.append("separate agreement with Moonshot needed (MaaS > $20M/12 months)")
        if d.commercial and (d.monthly_active_users > 100e6 or d.monthly_revenue_usd > 20e6):
            flags.append("show 'Kimi K3' prominently in the UI")
    elif model == "minimax-m2.7":
        if d.commercial:
            flags.append("prior written authorisation from MiniMax + 'Built with MiniMax M2.7'")
    else:
        flags.append("unknown licence: read the model card")
    return flags

startup = Deployment(True, False, 0, 50_000, 40_000)
print("kimi-k3:", licence_flags("kimi-k3", startup) or ["no extra conditions found"])
print("minimax-m2.7:", licence_flags("minimax-m2.7", startup))
assert licence_flags("kimi-k3", startup) == []
```

Calling a self-hosted model through vLLM's OpenAI-compatible server (check the
GLM-5 repo's vLLM recipe for required flags first):
<!-- skip-check: needs a running vLLM server with a multi-GPU node -->
```python
from openai import OpenAI

# Server started with: vllm serve zai-org/GLM-5.3-Flash  (plus recipe flags)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="zai-org/GLM-5.3-Flash",
    messages=[{"role": "user", "content": "List three risks in this deployment plan: ..."}],
)
print(r.choices[0].message.content)
```

## Choosing / trade-offs
- **Licence first**: GLM-5.x (MIT, per reports) and GLM-5.3-Flash are the
  simplest for commercial self-hosting; Kimi K3 is fine below its thresholds;
  MiniMax M2.7 needs permission for any commercial use ([[model-licenses]]).
- **Hardware**: total parameters set memory needs, not active ones
  ([[mixture-of-experts]]). Kimi K3 (2.8T) and GLM-5.3 (744B) need multi-GPU
  nodes even quantized; GLM-5.3-Flash (320B/18B) and MiniMax M3 (~428B/~23B)
  are the more practical self-host targets.
- **API vs self-host**: vendor APIs are cheapest to start and get new
  versions first; self-hosting gives data control and fixed cost at volume
  ([[cost-and-latency]]).
- **Third-party hosts**: convenient, but quantization and parser choices vary;
  run your own evals (and the vendor verifiers) before trusting quality.

## Gotchas
- Licences change between point releases of the same family (GLM-5.2 vs 5.3,
  MiniMax M2 vs M2.7); re-check on every upgrade.
- Secondary sources disagree on release dates, sizes and licences; trust the
  model card and LICENSE file over news articles.
- Dropping `reasoning_content` between turns degrades Kimi K3 tool use.
- Vendor benchmark tables use their own agent scaffolds and max effort;
  compare on your tasks ([[reading-ai-benchmarks]]).
- Some prohibited-use lists (e.g. MiniMax) cover uses like military or
  disinformation; check them against your sector.
- Model behaviour on politically sensitive topics can reflect the training
  jurisdiction; test if it matters for your product.

## Related
- [[qwen]], [[deepseek]] - the other major Chinese open-weight families.
- [[open-weight-model-landscape]] - dated table across vendors and modalities.
- [[code-models]] - coding-specific comparison and benchmarks.
- [[model-licenses]] - licence types and commercial use.
- [[mixture-of-experts]] - why active parameters differ from memory needs.
- [[inference-servers-vllm]], [[sglang-and-tensorrt-llm]] - self-hosting.
- [[model-selection]] - open vs closed decisions.

## References
- Kimi K3 README and licence: https://github.com/MoonshotAI/Kimi-K3
- GLM-5 family README: https://github.com/zai-org/GLM-5
- MiniMax M3 README: https://github.com/MiniMax-AI/MiniMax-M3
- MiniMax M2.7 licence: https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/LICENSE
- GLM-5.3 licence coverage (news): https://thenewstack.io/zai-glm-weights-license/
