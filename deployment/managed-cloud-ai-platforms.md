---
title: Managed cloud AI platforms (Amazon Bedrock, Google Vertex AI, Azure AI Foundry)
category: deployment
tags: [bedrock, vertex-ai, azure-ai-foundry, claude, data-residency, private-networking, iam, quotas, fine-tuning, managed-llm]
use_cases:
  - "call Claude through AWS, Google Cloud or Azure instead of the vendor's own API"
  - "pick between Bedrock, Vertex AI and Azure AI Foundry for a company already on one cloud"
  - "keep LLM traffic inside a private network and in one region or geography for compliance"
  - "work out what changes in code when moving a Claude app from the Anthropic API to a cloud provider"
status: draft
last_verified: 2026-10-11
sources:
  - https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy
  - https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai
  - https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry
  - https://docs.aws.amazon.com/bedrock/latest/userguide/vpc-interface-endpoints.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html
  - https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
  - https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/data-residency
  - https://docs.cloud.google.com/vertex-ai/docs/quotas
  - https://learn.microsoft.com/en-us/azure/foundry/how-to/configure-private-link
  - https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/fine-tuning
---

# Managed cloud AI platforms (Amazon Bedrock, Google Vertex AI, Azure AI Foundry)

## Summary
The three big clouds each resell hosted foundation models behind their own IAM, networking, billing and quota systems: Amazon Bedrock (AWS), Vertex AI (Google Cloud) and Azure AI Foundry (Microsoft). Choose one when your data, identity and procurement already live in that cloud, or when you need private networking and regional processing guarantees. The same model, for example Claude, can also be called through the vendor's own API; the cloud route changes auth, model IDs, endpoint, feature coverage and sometimes pricing, not the model itself.

## Key concepts
- Same model, different front door. Anthropic documents Claude on all three clouds. The request is nearly the Messages API, with platform-specific auth, a platform-specific model identifier and a platform-specific endpoint.
- Model ID formats differ per platform. Bedrock uses its own IDs, often with an inference-profile prefix (`global.`, `us.`, `eu.`, `jp.`, `apac.`). Vertex uses short IDs such as `claude-sonnet-5-5` and puts the model in the URL path. Foundry uses your deployment name as the `model` value, which defaults to the model ID.
- Endpoint scope = data residency lever. Bedrock and Vertex both offer global (dynamic routing, best availability, no residency guarantee), geography-level (US/EU style) and, for Vertex, single-region choices. Anthropic's docs state regional or multi-region options carry a pricing premium over global; check the current price pages.
- Auth. Bedrock: AWS IAM (SigV4 via the default credential chain) or a Bedrock bearer token. Vertex: Google Application Default Credentials (`gcloud auth application-default login`, service accounts). Foundry: an Azure-issued API key or Microsoft Entra ID tokens with Azure RBAC.
- Private networking. Bedrock: interface VPC endpoints (AWS PrivateLink) for `bedrock`, `bedrock-runtime` and related services, with endpoint policies. Foundry: private endpoints (Private Link) on the Foundry resource, public network access can be disabled, plus DNS setup. Vertex: Google Cloud networking and security controls such as VPC Service Controls exist, but this topic did not verify the exact Claude coverage; read the Google pages before relying on it.
- Quotas are account/project scoped and not the same as the vendor API's tiers. Bedrock quotas are viewed and raised through Service Quotas, not granted automatically. Vertex quotas are per project and often per region, raised in the console. Foundry rate limits are managed through Azure, and its Claude page says Anthropic's usual rate-limit response headers are not returned.
- Feature parity lags. All three Claude pages list features not supported through that platform (for example the Message Batches API, Files API and some server-side tools on at least Bedrock, Vertex and Foundry). Check the platform page for the feature you need.
- Fine-tuning is per platform and per model, not per cloud. Bedrock documents supervised fine-tuning, reinforcement fine-tuning and distillation as customization methods. Foundry documents supervised fine-tuning for supported models. Do not assume a given Claude model can be tuned anywhere; look up the model in that platform's supported-models list.

## When to use / scenarios
- A bank already on AWS with PrivateLink-only egress: Bedrock, with `bedrock-runtime` behind an interface endpoint and IAM roles instead of API keys.
- A company standardized on Microsoft Entra ID and Azure Marketplace billing: Foundry, with Entra auth and a private endpoint.
- A data team on Google Cloud that wants EU-only processing: Vertex with an EU multi-region or specific regional endpoint, not the global one.
- Prototype or small team, no cloud mandate: the vendor's own API is simpler and usually gets new features first ([[api-sdk-setup]]). Anthropic's pages also describe offerings that are Anthropic-operated but billed through a cloud marketplace.
- Multi-cloud or failover across providers: put an abstraction in front ([[llm-gateways-and-routing]]).
- Not a fit: needing a feature the platform page lists as unsupported, or needing an open model you must host and tune yourself ([[inference-servers-vllm]], [[gpu-cloud-options]]).

## Setup & code
Install the SDKs (pin versions in your project; these are the packages Anthropic's pages name):
```bash
pip install -U "anthropic[bedrock]" "anthropic[vertex]" anthropic boto3 azure-identity
```

Bedrock, boto3 Converse API (a cross-model API on the `bedrock-runtime` endpoint). Use a model ID or inference-profile ID from the Bedrock catalog for your region:
<!-- skip-check: needs cloud credentials -->
```python
import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")  # default AWS credential chain
resp = client.converse(
    modelId="<inference-profile or model ID from the Bedrock catalog>",
    messages=[{"role": "user", "content": [{"text": "Hello"}]}],
    inferenceConfig={"maxTokens": 256},
)
print(resp["output"]["message"]["content"][0]["text"])
```
To keep traffic on a private path without private DNS, pass `endpoint_url` (your `vpce-...bedrock-runtime...vpce.amazonaws.com` address) to `boto3.client`.

Vertex AI, Anthropic SDK (`AnthropicVertex`; model is a short ID, region is `global`, `us`/`eu` or a specific region):
<!-- skip-check: needs cloud credentials -->
```python
from anthropic import AnthropicVertex

client = AnthropicVertex(project_id="MY_PROJECT_ID", region="global")  # uses Application Default Credentials
msg = client.messages.create(
    model="claude-sonnet-5-5",  # check the Vertex model table for current IDs
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello"}],
)
print(msg.content[0].text)
```

Azure AI Foundry, Anthropic SDK (`AnthropicFoundry`; `model` is your deployment name; Entra ID shown, an API key also works):
<!-- skip-check: needs cloud credentials -->
```python
from anthropic import AnthropicFoundry
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://ai.azure.com/.default")
client = AnthropicFoundry(resource="my-foundry-resource", azure_ad_token_provider=token_provider)
msg = client.messages.create(
    model="claude-sonnet-5-5",  # your deployment name (defaults to the model ID)
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello"}],
)
print(msg.content[0].text)
```

Vendor API for comparison: `anthropic.Anthropic().messages.create(model=..., max_tokens=..., messages=[...])` with `ANTHROPIC_API_KEY`. The Messages shape is the same across the three clouds when using Anthropic's SDK classes; boto3 Converse is the exception, with its own request and response shape.

A tiny routing table that picks platforms by required residency (runnable, stdlib only; the entries encode the scopes described on the platform pages above, verify them before relying on them):
```python
# What each platform can offer for Claude, per the platform docs (verified 2026-10-11).
SCOPES = {
    "bedrock": {"global", "us", "eu", "jp", "apac"},   # inference-profile prefixes
    "vertex": {"global", "us", "eu", "region"},        # multi-region us/eu, or a named region
    "foundry": {"global", "us"},                       # Global Standard, US Data Zone Standard
}

def candidates(required_scope: str) -> list[str]:
    return sorted(p for p, scopes in SCOPES.items() if required_scope in scopes)

print(candidates("eu"))      # ['bedrock', 'vertex']
print(candidates("us"))      # ['bedrock', 'foundry', 'vertex']
print(candidates("global"))  # ['bedrock', 'foundry', 'vertex']
assert candidates("jp") == ["bedrock"]
```
Expected output:
```text
['bedrock', 'vertex']
['bedrock', 'foundry', 'vertex']
['bedrock', 'foundry', 'vertex']
```

## Choosing / trade-offs
- Existing cloud footprint usually decides: IAM, networking, audit logging and billing integration beat small model-catalog differences.
- Global endpoint (cheapest, most available, no residency guarantee) vs geography or region endpoint (residency, price premium, sometimes fewer models). Google states that global endpoints do not provide residency guarantees.
- Cloud route vs vendor API: cloud gives enterprise networking and procurement; the vendor API tends to have the full feature set. Compare against the platform's "features not supported" list before committing.
- Catalog breadth: each cloud also hosts non-Claude models (Bedrock, Vertex Model Garden, the Foundry catalog), so one platform can serve a multi-vendor strategy ([[model-selection]]).
- API portability: using `AnthropicBedrock` / `AnthropicVertex` / `AnthropicFoundry` keeps one Messages-style code path; using boto3 Converse or a cloud-native SDK buys cross-model portability within that cloud but not across clouds.

## Gotchas
- Model IDs are not interchangeable: `claude-sonnet-5-5` works on Vertex but Bedrock needs its own (often prefixed) ID, and Foundry uses the deployment name. Hard-code none of them; read the platform table.
- Some newer Bedrock models cannot be invoked with the base model ID (on-demand throughput); you must pass an inference profile ID or ARN, otherwise you get an HTTP 400.
- On Bedrock, the Converse API and InvokeModel/Messages-style paths differ in which Claude features they expose (the legacy page notes PDF visual analysis via Converse requires citations). Check which endpoint your feature needs.
- Quota names, defaults and increase rules differ and change; read them from Service Quotas (AWS), the Quotas page (Google) or the Azure portal, not from memory or blog posts.
- A private endpoint is not enough on its own: Azure needs working private DNS, and on AWS you either enable private DNS on the endpoint or pass the VPC endpoint URL explicitly.
- Foundry has two hosting options for Claude (hosted on Azure vs on Anthropic) with different feature support and region scope; confirm which one a deployment uses.
- Lifecycle dates for models on partner platforms are set by the platform and can differ from the vendor API schedule.
- Model availability varies by region and by account access (Bedrock model access, Vertex Model Garden enablement, Foundry Marketplace terms) and the model catalog page is the source of truth.
- Prompts and logs still count as personal data wherever they run ([[ai-security-privacy-compliance]]); enabling invocation logging on any platform creates another copy.

## Related
- [[model-selection]] - choosing the model before choosing where to host it.
- [[ai-security-privacy-compliance]] - DPAs, retention and residency questions to answer first.
- [[llm-gateways-and-routing]] - one client interface over several providers, fallbacks and cost routing.
- [[cost-and-latency]] - regional versus global pricing and latency trade-offs.
- [[llm-observability]] - logging requests on a managed platform.
- [[api-sdk-setup]] - calling the vendor's own API and handling keys.
- [[gpu-cloud-options]] - the self-hosted alternative to managed endpoints.
- [[inference-servers-vllm]] - serving open models yourself instead.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/security/secrets-management.md - prefer IAM roles and managed identity over long-lived keys.

## References
- Claude on Amazon Bedrock (legacy Converse/InvokeModel page, model IDs, endpoints, features): https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy
- Claude on Google Cloud (Vertex): https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai
- Claude in Microsoft Foundry: https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry
- Bedrock interface VPC endpoints: https://docs.aws.amazon.com/bedrock/latest/userguide/vpc-interface-endpoints.html
- Bedrock quotas: https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html
- Bedrock model customization: https://docs.aws.amazon.com/bedrock/latest/userguide/custom-models.html
- Bedrock Converse API: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
- Google generative AI data residency: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/data-residency
- Vertex AI quotas: https://docs.cloud.google.com/vertex-ai/docs/quotas
- Foundry network isolation / private link: https://learn.microsoft.com/en-us/azure/foundry/how-to/configure-private-link
- Foundry supervised fine-tuning: https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/fine-tuning
