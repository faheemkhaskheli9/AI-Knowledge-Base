---
title: AI security, privacy and compliance basics
category: deployment
tags: [security, privacy, pii, gdpr, eu-ai-act, secrets, data-retention, compliance]
use_cases:
  - "send customer data to an LLM API without leaking PII"
  - "decide whether my AI feature falls under the EU AI Act and what to do about it"
  - "keep API keys and model credentials out of git and logs"
  - "set data retention and vendor terms for an AI feature in healthcare or finance"
status: stable
last_verified: 2026-10-03
sources:
  - https://artificialintelligenceact.eu/implementation-timeline/
  - https://gdpr.eu/
  - https://owasp.org/www-project-top-10-for-large-language-model-applications/
---

# AI security, privacy and compliance basics

## Summary
Shipping an AI feature creates data-flow, security and legal obligations that ordinary apps may not have: prompts carry personal data to third parties, models can be manipulated through inputs, and regulation (GDPR, EU AI Act, sector rules) applies to how you collect, process and retain data. This file is an engineering checklist, not legal advice; involve counsel for regulated use.

## Key concepts
- Data minimization: send the model only what the task needs; redact or pseudonymize identifiers first.
- PII/PHI: names, emails, IDs, health or financial data; special-category data has stricter rules.
- Vendor terms: whether prompts are retained or used for training, retention period, region, and a data processing agreement (DPA); enterprise/API tiers usually differ from consumer apps. Read the current terms.
- GDPR basics: lawful basis, purpose limitation, data subject rights (access, erasure), processors vs controllers, international transfer rules.
- EU AI Act: risk-tiered. Per the published timeline: prohibitions and AI literacy duties from 2 Feb 2025; general-purpose AI model rules from 2 Aug 2025; transparency/synthetic-content labeling from 2 Aug 2026; high-risk (Annex III) requirements from 2 Dec 2027 and Annex I from 2 Aug 2028. Dates have been subject to amendment proposals; verify the current schedule.
- OWASP Top 10 for LLM applications: prompt injection, insecure output handling, data leakage, excessive agency, etc.
- Secrets: API keys and tokens are credentials; leaked keys get abused fast.

## When to use / scenarios
- Support bot ingesting tickets with customer names and card fragments.
- Clinic or legal assistant: prefer self-hosted or a vendor with a signed DPA/BAA-equivalent, zero-retention options ([[ollama]], [[inference-servers-vllm]]).
- EU users or EU deployment of a hiring, credit or education AI (likely high-risk category): needs documentation, human oversight, logging.
- Agents with tool access (email, DB, payments): treat model output as untrusted input ([[prompt-injection]]).

## Setup & code
Secrets:
```bash
echo ".env" >> .gitignore
export ANTHROPIC_API_KEY=...        # or load from a secrets manager at runtime
```
Scan for leaks before pushing (example tools: gitleaks, trufflehog); rotate any key that was committed, since history rewrites do not un-leak it.

Minimal PII redaction before an LLM call (regex baseline; use a dedicated tool such as Microsoft Presidio for production):
```python
import re
PATTERNS = {"EMAIL": r"[\w.+-]+@[\w-]+\.[\w.]+", "PHONE": r"\+?\d[\d\s().-]{7,}\d",
            "CARD": r"\b(?:\d[ -]?){13,16}\b"}
def redact(text: str) -> str:
    for label, pat in PATTERNS.items():
        text = re.sub(pat, f"[{label}]", text)
    return text
assert redact("mail bob@x.com") == "mail [EMAIL]"
```
Engineering checklist:
1. Map data flows: what leaves your network, to which vendor, in which region.
2. Redact/pseudonymize; keep a reversible mapping server-side if you must restore values.
3. Set log and trace retention limits and access control ([[llm-observability]]).
4. Least privilege for agent tools; require human approval for destructive actions.
5. Validate and sanitize model output before rendering as HTML, SQL or shell.
6. Per-user authentication and rate limits ([[serving-with-fastapi]]).
7. Record model/vendor, purpose and risk classification in a short AI register.

## Choosing / trade-offs
- Hosted API (best models, vendor processes data under DPA) vs self-hosted (data stays, you own security and quality gaps).
- Redaction vs utility: over-redaction hurts answer quality; test on real samples.
- Detailed logging (debuggability) vs privacy (minimal retention): sample and expire.
- Compliance by design early is cheaper than retrofitting after launch.

## Gotchas
- Regexes miss names and free-text identifiers; use NER-based tools for real PII detection.
- Embeddings and vector stores also contain personal data; they need deletion paths for erasure requests ([[vector-databases]]).
- Prompt logs in observability tools are a second copy of customer data.
- "Not used for training" does not mean "not retained"; check retention separately.
- Fine-tuning on personal data can leak it via memorization.
- Model licenses may restrict use cases ([[model-licenses]]).
- Rules differ by jurisdiction and sector (HIPAA, PCI-DSS, local laws); this file covers only general points.

## Related
- [[prompt-injection]] - attack class for LLM apps.
- [[guardrails-and-safety]] - runtime controls.
- [[llm-observability]] - logging with retention.
- [[api-sdk-setup]] - key handling.
- [[gpu-cloud-options]] - where data is processed.
- [[healthcare]], [[finance]], [[legal]] - sector scenarios.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/security/threat-modeling.md - threat-modelling the whole system.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/security/secrets-management.md - API keys and credentials.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/security/owasp-top-10.md - web-app risks around the model.

## References
- EU AI Act timeline: https://artificialintelligenceact.eu/implementation-timeline/
- GDPR overview: https://gdpr.eu/
- OWASP LLM Top 10: https://owasp.org/www-project-top-10-for-large-language-model-applications/
