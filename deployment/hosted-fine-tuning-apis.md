---
title: Hosted fine-tuning and reinforcement fine-tuning APIs
category: deployment
tags: [fine-tuning, reinforcement-fine-tuning, rft, dpo, sft, openai, gemini, bedrock, together-ai, fireworks, managed-training]
use_cases:
  - "fine-tune a hosted model on my examples without running GPUs"
  - "decide between prompting, a hosted fine-tune and self-hosted LoRA"
  - "use reinforcement fine-tuning with a grader for a domain reasoning task"
  - "estimate the training and inference cost of a hosted fine-tune"
status: draft
last_verified: 2026-10-11
sources:
  - https://developers.openai.com/api/docs/guides/model-optimization
  - https://developers.openai.com/api/docs/deprecations
  - https://developers.openai.com/api/docs/pricing
  - https://developers.openai.com/api/docs/guides/direct-preference-optimization
  - https://developers.openai.com/api/docs/guides/reinforcement-fine-tuning
  - https://ai.google.dev/gemini-api/docs/model-tuning
  - https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning
  - https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/reinforcement-tuning
  - https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing
  - https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions
  - https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/custom-model-fine-tuning.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/reinforcement-fine-tuning.html
  - https://aws.amazon.com/bedrock/pricing/
  - https://docs.together.ai/docs/fine-tuning/overview
  - https://docs.fireworks.ai/fine-tuning/managed-finetuning-intro
---

# Hosted fine-tuning and reinforcement fine-tuning APIs

## Summary
Hosted fine-tuning lets you upload a JSONL dataset and get back a tuned model
(usually a LoRA adapter on the provider's base model) without managing GPUs.
Providers now offer three kinds: supervised (SFT, input/ideal-output pairs),
preference (DPO, preferred vs rejected answers) and reinforcement fine-tuning
(RFT, prompts plus a grader or reward function). As of 2026-10 the landscape
has shifted: OpenAI is winding down self-serve fine-tuning, the Gemini
Developer API has no tunable model, and the clouds (Google's Gemini Enterprise
Agent Platform, formerly Vertex AI, and Amazon Bedrock) plus open-weight hosts
(Together AI, Fireworks) are where new hosted tuning happens.

## Key concepts
- **SFT**: teaches format, tone, a narrow task, tool-call shape. OpenAI's
  hard minimum is 10 examples; how many you need beyond that is an eval
  question, not a constant.
- **Preference tuning (DPO)**: pairs of preferred and non-preferred responses
  to the same prompt. Good for style and emphasis that is easier to judge than
  to write ([[rlhf-and-preference-optimization]]).
- **Reinforcement fine-tuning (RFT)**: you supply prompts and a grader
  (string match, Python, model-graded). The platform samples several answers
  per prompt, scores them and applies policy-gradient updates. It needs a
  task with a checkable answer, the hosted version of RL with verifiable
  rewards ([[reasoning-models]], [[grpo-from-scratch-numpy]]).
- **What you get back**: an adapter or model ID that only that provider can
  serve. Inference is billed per token (sometimes at a premium over the base
  model) or needs dedicated capacity.
- **Tuned models die with their base**: OpenAI keeps fine-tuned models for
  inference only until the base model is deprecated. Check each cloud's
  retirement policy for tuned models, and keep the dataset so you can retrain
  on the successor.

## Provider matrix (verified 2026-10-11)
| Provider | Methods | Tunable base models | Cost model |
|---|---|---|---|
| OpenAI API | SFT, vision SFT, DPO | SFT/DPO: `gpt-4.1-2025-04-14`, `gpt-4.1-mini-2025-04-14`; vision: `gpt-4o-2024-08-06` | training per 1M tokens (e.g. $5 for gpt-4.1-mini); tuned inference per token |
| OpenAI API (retiring) | SFT/DPO, RFT | `gpt-4.1-nano-2025-04-14` and `o4-mini-2025-04-16`, the only RFT base, shut down 2026-10-23 (deprecations page) <!-- valid-until: 2026-10-23 --> | RFT $100 per training hour plus grader tokens |
| Gemini Developer API / AI Studio | none | no tunable model since Gemini 1.5 Flash-001 was deprecated (May 2025) | - |
| Gemini Enterprise Agent Platform (Vertex AI) | SFT, RL fine-tuning | SFT: Gemini 3.5 Flash, 3.1 Flash-Lite; RL: Gemini 3.5 Flash (pricing page); also Gemma, Llama, Qwen open models | per 1,000 training tokens (dataset tokens x epochs), e.g. $0.01 for Gemini 3.5 Flash; from Gemini 3, tuned-endpoint inference is 1.5x the base price |
| Agent Platform (retiring) | SFT, preference tuning | Gemini 2.5 Pro, 2.5 Flash and 2.5 Flash-Lite (the only preference-tuning bases) retire 2026-10-20 (model versions page) <!-- valid-until: 2026-10-20 --> | e.g. $0.005 per 1,000 training tokens for 2.5 Flash |
| Amazon Bedrock | SFT, RFT, distillation | SFT: Nova 2 Lite, Nova Lite/Micro/Pro (Active); RFT: Nova 2 Lite, gpt-oss-20B, Qwen3 32B | training per token or per hour (RFT, e.g. $80/hour for gpt-oss-20b), $1.95 per custom model per month storage, on-demand or Provisioned Throughput inference |
| Bedrock (legacy) | SFT | Llama 3.1 8B/70B and 3.3 70B are Legacy, so no new tuning jobs, with EOL 2027-03-30 (model cards) <!-- valid-until: 2027-03-30 -->; Llama 3.2 (EOL 2026-07-07) and Claude 3 Haiku (EOL 2026-09-10) still sit in the fine-tuning table but are past EOL | - |
| Together AI | SFT, DPO; LoRA or full | open-weight models listed in its docs | see its pricing page; serve on a dedicated endpoint or download the weights |
| Fireworks | SFT, DPO, RFT (managed RFT now deprecated in favour of its training API); LoRA only | open-weight models listed in its docs | trained LoRAs serve only on on-demand (dedicated) deployments |

OpenAI timeline: since 2026-05-07 organisations that never fine-tuned cannot
start; since 2026-07-02 orgs without fine-tuned inference in the past 60
days cannot create jobs; active customers can create jobs until 2027-01-06. <!-- valid-until: 2027-01-06 -->
Inference on existing fine-tunes continues until each base model is
deprecated.

## When to use / scenarios
- Prompting plus few-shot ([[prompt-engineering]]) and [[structured-output]]
  already work but the prompt is long and expensive at volume: SFT a smaller
  model to drop the examples.
- A classifier, extractor or router at high volume where a tuned small model
  matches a large one ([[knowledge-distillation-and-compression]]).
- Domain reasoning with a checkable answer (code that passes tests, a label
  with an expert key, a structured verdict): RFT.
- Tone and style that reviewers can rank but not easily write: DPO.
- Not for facts that change or need citations: use RAG ([[rag-basics]]).
- Not when you need the weights, on-prem serving or the newest frontier model
  (none of the providers above lists its newest flagship as tunable): self-host with LoRA
  ([[fine-tuning-and-peft]]).

## Setup & code
Build and sanity-check the three JSONL shapes OpenAI documents, and compare a
training bill. Runnable, no API calls:
```python
import json

sft = {"messages": [
    {"role": "system", "content": "Reply with the ticket category only."},
    {"role": "user", "content": "My card was charged twice"},
    {"role": "assistant", "content": "billing"}]}
dpo = {"input": {"messages": [{"role": "user", "content": "Summarise: the meeting moved to 3pm"}]},
       "preferred_output": [{"role": "assistant", "content": "Meeting now at 3pm."}],
       "non_preferred_output": [{"role": "assistant", "content": "There was a change."}]}
rft = {"messages": [{"role": "user", "content": "Is SOC 2 mentioned in the policy text?"}],
       "compliant": "yes"}  # extra fields are what the grader compares against

def check(line, kind):
    row = json.loads(line)
    if kind == "sft":
        assert row["messages"][-1]["role"] == "assistant", "SFT target must be last"
    elif kind == "dpo":  # one-turn; both outputs are the final assistant message
        assert row["preferred_output"][0]["role"] == "assistant"
        assert row["non_preferred_output"][0]["role"] == "assistant"
    elif kind == "rft":
        assert row["messages"][-1]["role"] == "user", "RFT rows end on the prompt"
        assert set(row) - {"messages"}, "RFT rows need reference fields for the grader"
    return row

for row, kind in [(sft, "sft"), (dpo, "dpo"), (rft, "rft")]:
    check(json.dumps(row), kind)

def training_cost(dataset_tokens, epochs, usd_per_million):
    return dataset_tokens * epochs / 1e6 * usd_per_million

tokens = 2_000_000  # e.g. 4,000 examples x 500 tokens
print("OpenAI gpt-4.1-mini SFT, 3 epochs: $", training_cost(tokens, 3, 5.00))
print("Gemini 3.1 Flash-Lite SFT ($0.003 / 1k), 3 epochs: $", training_cost(tokens, 3, 3.00))
print("Gemini 3.5 Flash SFT ($0.01 / 1k), 3 epochs: $", training_cost(tokens, 3, 10.00))
print("OpenAI o4-mini RFT, 8 training hours: $", 8 * 100.0, "+ grader tokens")  # <!-- valid-until: 2026-10-23 -->
```
Starting a Gemini tuning job on the Agent Platform with the Google Gen AI SDK
(from Google's codelab; needs a project, a Cloud Storage dataset and
credentials):
<!-- skip-check: needs Google Cloud credentials and a dataset in Cloud Storage -->
```python
import os
from google import genai
from google.genai import types

os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "true"  # route to the Agent Platform, not the Developer API
client = genai.Client()
job = client.tunings.tune(
    base_model="gemini-3.5-flash",
    training_dataset=types.TuningDataset(gcs_uri="gs://my-bucket/train.jsonl"),
    config=types.CreateTuningJobConfig(tuned_model_display_name="ticket-router", epoch_count=1),
)
print(job.name, job.state)
```

## Choosing / trade-offs
- **Hosted vs self-hosted**: hosted removes GPU ops and evaluation plumbing
  but locks the model to one provider, limits hyperparameters and base models,
  and ties the tune's life to the base model's retirement. Self-hosted LoRA on
  open weights ([[fine-tuning-and-peft]]) keeps the weights.
- **SFT vs DPO vs RFT**: write the ideal answer -> SFT; can only rank answers
  -> DPO; can score answers with code or a rubric -> RFT. RFT costs more
  (hourly billing, grader calls) and overfits to a leaky grader.
- **Cost**: training is cheap relative to inference at scale; check the
  inference premium (Gemini 3+ tuned endpoints at 1.5x, Bedrock storage fees,
  dedicated deployments on Fireworks/Together) before assuming savings.
- **Data residency and privacy**: training data is stored by the provider;
  the clouds give regional control ([[managed-cloud-ai-platforms]],
  [[ai-security-privacy-compliance]]).

## Gotchas
- Building on OpenAI fine-tuning now is a dead end for new organisations;
  plan a migration path for existing tunes before 2027-01-06. <!-- valid-until: 2027-01-06 -->
- A tune is pinned to a dated base snapshot; when that snapshot retires the
  tuned model stops serving. Track base-model retirement dates
  ([[openai-gpt]], [[google-gemini]]).
- Evaluate against the untuned base with the same prompt and a held-out set
  ([[llm-evaluation]]); many "wins" come from the prompt, not the weights.
- RFT reward hacking: the model learns the grader's blind spots. Test the
  grader on adversarial outputs before training.
- Fine-tuning teaches behaviour, not reliable new facts.

## Related
- [[fine-tuning-and-peft]] - self-hosted LoRA/QLoRA and when fine-tuning helps at all.
- [[rlhf-and-preference-optimization]] - the DPO and RLHF ideas behind preference tuning.
- [[reasoning-models]] - RL with verifiable rewards, which RFT productises.
- [[managed-cloud-ai-platforms]] - Bedrock and Vertex/Agent Platform around the tuning job.
- [[llm-evaluation]] - measuring whether the tune beat prompting.
- [[cost-and-latency]] - inference cost of the tuned model.

## References
- OpenAI model optimization (methods and tunable models): https://developers.openai.com/api/docs/guides/model-optimization
- OpenAI deprecations (self-serve fine-tuning timeline): https://developers.openai.com/api/docs/deprecations
- OpenAI pricing (fine-tuning table): https://developers.openai.com/api/docs/pricing
- Gemini API tuning status: https://ai.google.dev/gemini-api/docs/model-tuning
- Agent Platform tuning overview: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning
- Agent Platform RL fine-tuning: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/reinforcement-tuning
- Agent Platform pricing (Model Tuning table): https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing
- Agent Platform model versions and retirement dates: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions
- Bedrock model lifecycle (Legacy blocks new tuning jobs; dates on each model card): https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
- Bedrock customization methods: https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html
- Bedrock fine-tuning models: https://docs.aws.amazon.com/bedrock/latest/userguide/custom-model-fine-tuning.html
- Bedrock RFT models: https://docs.aws.amazon.com/bedrock/latest/userguide/reinforcement-fine-tuning.html
- Bedrock pricing: https://aws.amazon.com/bedrock/pricing/
- Together AI fine-tuning: https://docs.together.ai/docs/fine-tuning/overview
- Fireworks managed training: https://docs.fireworks.ai/fine-tuning/managed-finetuning-intro
