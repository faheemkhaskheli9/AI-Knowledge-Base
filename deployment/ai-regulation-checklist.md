---
title: AI regulation checklist (EU AI Act, ISO/IEC 42001)
category: deployment
tags: [eu-ai-act, iso-42001, compliance, governance, high-risk-ai, transparency, gpai, nist-ai-rmf, digital-omnibus]
use_cases:
  - "work out whether our AI feature is prohibited, high-risk, transparency-only or minimal-risk under the EU AI Act"
  - "list which EU AI Act obligations apply to us and from which date"
  - "label AI-generated content and chatbot interactions to meet EU transparency rules"
  - "decide whether to pursue ISO/IEC 42001 certification for an AI product company"
  - "build an AI system register and evidence pack an auditor or customer will ask for"
status: draft
last_verified: 2026-10-15
sources:
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
  - https://eur-lex.europa.eu/eli/reg/2026/1744/oj
  - https://eur-lex.europa.eu/eli/reg/2024/1689/2026-07-27/eng
  - https://ai-act-service-desk.ec.europa.eu/en/ai-act/timeline/timeline-implementation-eu-ai-act
  - https://www.consilium.europa.eu/en/press/press-releases/2026/06/29/artificial-intelligence-council-gives-final-green-light-to-simplify-and-streamline-rules/
  - https://www.iso.org/standard/42001
  - https://www.nist.gov/itl/ai-risk-management-framework
---

# AI regulation checklist (EU AI Act, ISO/IEC 42001)

## Summary
The EU AI Act (Regulation (EU) 2024/1689) regulates AI systems by risk tier: some practices are banned, high-risk systems carry heavy documentation and oversight duties, chatbots and generative systems carry transparency duties, and general-purpose AI (GPAI) model providers have their own obligations. It was amended in 2026 by the "Digital Omnibus on AI" (Regulation (EU) 2026/1744), which postponed the high-risk dates. ISO/IEC 42001:2023 is a certifiable AI management-system standard that gives you the processes to show compliance. This file is an engineering checklist, not legal advice.

## Key concepts
- **Roles**: *provider* (develops or has developed an AI system/model and places it on the EU market under its name), *deployer* (uses one under its authority), plus importer and distributor. Obligations depend on role; fine-tuning or rebranding a system can make you a provider.
- **Risk tiers**:
  - *Prohibited* (Art. 5): e.g. social scoring, manipulative techniques causing harm, untargeted facial-image scraping, emotion recognition at work or school (with exceptions). The omnibus adds a ban on AI that generates non-consensual intimate imagery or child sexual abuse material.
  - *High-risk* (Art. 6): Annex III use areas (e.g. employment, education, credit scoring, essential services, law enforcement, migration, critical infrastructure) and AI that is a safety component of products under Annex I law (e.g. machinery, medical devices).
  - *Transparency* (Art. 50): tell people they are talking to an AI; mark synthetic audio, image, video and text in a machine-readable way; disclose deepfakes.
  - *Minimal risk*: everything else; no specific duties beyond AI literacy.
- **High-risk provider duties** (Chapter III): risk management system, data governance, technical documentation, automatic logging, instructions for use, human oversight, accuracy/robustness/cybersecurity, quality management system, conformity assessment, CE marking, EU database registration, post-market monitoring, serious-incident reporting.
- **High-risk deployer duties**: use per instructions, assign competent human oversight, keep logs, monitor, inform affected workers; some deployers (public bodies, credit, insurance) must do a fundamental-rights impact assessment.
- **GPAI model providers**: technical documentation, information for downstream providers, copyright policy, public summary of training content; extra evaluation, incident and cybersecurity duties for models with systemic risk.
- **Penalties** (Art. 99 of 2024/1689): up to EUR 35M or 7% of worldwide turnover for prohibited practices; up to EUR 15M or 3% for most other breaches; up to EUR 7.5M or 1% for supplying incorrect information.
- **ISO/IEC 42001:2023**: requirements for an AI management system (AIMS): policy, roles, AI risk assessment and treatment, AI system impact assessment, controls (Annex A), internal audit, continual improvement. Same management-system structure as ISO/IEC 27001, so it bolts onto an existing ISMS. Certification is done by independent certification bodies.
- **NIST AI RMF 1.0**: voluntary US framework (Govern, Map, Measure, Manage); useful as a risk vocabulary, not a legal requirement.

### Key dates (EU AI Act as amended by Regulation (EU) 2026/1744)
| Date | What applies |
|---|---|
| 2025-02-02 | Prohibitions (Art. 5) and AI literacy (Art. 4) |
| 2025-08-02 | GPAI model obligations; governance and penalty rules |
| 2026-07-27 | Digital Omnibus (2026/1744) in force |
| 2026-08-02 | Transparency duties (Art. 50); Commission AI Office starts enforcing GPAI rules, including fines |
| 2026-12-02 | Omnibus prohibitions on non-consensual intimate imagery and CSAM apply; end of grace period for Art. 50(2) marking by generative systems already on the market before 2026-08-02 <!-- valid-until: 2026-12-02 --> |
| 2027-08-02 | GPAI models placed on the market before 2025-08-02 must comply; national AI regulatory sandboxes must be operational <!-- valid-until: 2027-08-02 --> |
| 2027-12-02 | High-risk obligations for Annex III systems (postponed from 2026-08-02) <!-- valid-until: 2027-12-02 --> |
| 2028-08-02 | High-risk obligations for AI in Annex I regulated products (postponed from 2027-08-02) <!-- valid-until: 2028-08-02 --> |

Sources: EUR-Lex record of Regulation (EU) 2026/1744 (in force, OJ L 24.7.2026), Council press releases of 2026-05-07 and 2026-06-29, and the Commission AI Act Service Desk timeline. Whether the omnibus changed the scope of the AI literacy duty (Art. 4) was not verified here; read the consolidated text.

## When to use / scenarios
- Any AI feature used by people in the EU, wherever the company is based (the Act applies to providers placing systems on the EU market and to outputs used in the EU).
- HR tech (CV screening, promotion decisions), edtech (exam scoring, admissions), fintech (creditworthiness), insurtech (life/health pricing): likely Annex III high-risk. Start the evidence pack now; 2027-12-02 is close for a full QMS.
- Customer-service chatbot or AI voice agent: transparency only; disclose that users are talking to an AI.
- Image, video or voice generation product: machine-readable marking of outputs and deepfake disclosure ([[image-generation-models]]).
- Fine-tuning and releasing an open-weight model: you may be a GPAI provider; open-source models have lighter duties unless they carry systemic risk ([[model-licenses]]).
- B2B sales to large enterprises: customers increasingly ask for ISO/IEC 42001 or an equivalent governance description.
- Not EU-facing and not regulated sector: NIST AI RMF or a light internal AI register may be enough; check local law (US state laws, sector regulators).

## Setup & code
A first-pass triage that records the answers and returns the tier plus the dated duties. It is a conversation starter for counsel, not a legal determination.

```python
from datetime import date

PROHIBITED = {"social_scoring", "manipulation_causing_harm", "untargeted_face_scraping",
              "emotion_recognition_work_school", "ncii_or_csam_generation"}
ANNEX_III = {"employment", "education", "credit_scoring", "essential_services", "law_enforcement",
             "migration_border", "critical_infrastructure", "biometrics", "justice_democracy",
             "life_health_insurance_pricing"}
DATES = {"transparency": date(2026, 8, 2), "high_risk_annex_iii": date(2027, 12, 2),
         "high_risk_annex_i": date(2028, 8, 2), "prohibitions": date(2025, 2, 2)}

def triage(practices=(), use_area=None, annex_i_product=False, interacts_with_people=False,
           generates_media=False, today=date(2026, 10, 15)):
    duties = []
    if PROHIBITED & set(practices):
        return "prohibited", ["Stop: practice is banned since " + str(DATES["prohibitions"])
                              + " (ncii/csam ban from 2026-12-02)"]
    tier = "minimal"
    if annex_i_product or use_area in ANNEX_III:
        tier = "high-risk"
        key = "high_risk_annex_i" if annex_i_product else "high_risk_annex_iii"
        when = DATES[key]
        status = "applies" if today >= when else f"applies from {when} ({(when - today).days} days)"
        duties += [f"Risk mgmt, data governance, tech docs, logging, human oversight, QMS, "
                   f"conformity assessment, EU database registration: {status}"]
    if interacts_with_people or generates_media:
        tier = tier if tier == "high-risk" else "transparency"
        if interacts_with_people:
            duties.append("Tell users they are interacting with an AI (Art. 50)")
        if generates_media:
            duties.append("Machine-readable marking of synthetic content; disclose deepfakes (Art. 50)")
    duties.append("AI literacy for staff operating the system (Art. 4)")
    return tier, duties

for case in [dict(use_area="employment", interacts_with_people=True),
             dict(interacts_with_people=True, generates_media=True),
             dict(practices=["social_scoring"])]:
    tier, duties = triage(**case)
    print(tier); [print("  -", d) for d in duties]
```

Practical checklist (do these in order):
1. **Inventory**: an AI system register: name, owner, purpose, users, vendor/model ID, data categories, EU exposure, your role (provider/deployer). A model registry supplies much of it ([[model-registry-and-versioning]]).
2. **Classify** each system with the triage above and record the reasoning; re-run when the purpose changes.
3. **Stop or redesign** anything near a prohibited practice.
4. **Transparency (now)**: AI-interaction notices in chat and voice UIs; content marking (C2PA-style metadata or watermarking) for generated media; deepfake labels.
5. **High-risk (by 2027-12-02 / 2028-08-02)**: risk-management file, dataset documentation and bias checks, technical documentation, event logging with retention, human-oversight design (who can override, how), accuracy and robustness tests ([[llm-evaluation]], [[adversarial-examples-and-robustness]]), cybersecurity ([[prompt-injection]]), QMS, conformity assessment, registration, post-market monitoring and incident reporting ([[llm-observability]]).
6. **GPAI**: if you train or substantially modify and distribute a general-purpose model: model documentation, downstream-provider information, copyright policy, training-content summary.
7. **Vendors**: get the upstream provider's documentation and instructions for use; check contractual allocation of AI Act duties and data terms ([[ai-security-privacy-compliance]]).
8. **Management system**: assign accountability, write an AI policy, train staff (AI literacy), schedule internal audits; map these to ISO/IEC 42001 clauses if you plan certification.

## Choosing / trade-offs
- **ISO/IEC 42001 certification vs internal governance only**: certification costs audit fees and process overhead but answers enterprise procurement questionnaires; it does not by itself give AI Act conformity.
- **Provider vs deployer**: building on a third-party model keeps you a deployer only if you do not put your name on it or substantially modify its intended purpose; otherwise you inherit provider duties.
- **Design out of high-risk**: a narrower purpose (e.g. AI drafts a summary, a human makes the hiring decision) can change the classification; Art. 6(3) allows some Annex III systems to be treated as not high-risk when they do not materially influence decisions, but you must document that assessment.
- **One global process vs per-jurisdiction**: aligning to the strictest regime (usually the EU) simplifies engineering at the cost of some over-compliance elsewhere.

## Gotchas
- The dates moved once (Digital Omnibus); secondary sources and older blog posts still show 2026-08-02 for high-risk. Check EUR-Lex for the consolidated text before planning.
- Transparency duties were not postponed; they apply from 2026-08-02.
- "We only call an API" does not make you exempt: deployers of high-risk systems have their own duties.
- Logs required for high-risk systems are also personal data under GDPR; set retention and access deliberately.
- Fine-tuning a GPAI model for a high-risk purpose can make you the provider of a high-risk system.
- ISO/IEC 42001 is a paid standard; free summaries do not list every Annex A control accurately.
- Sector rules (medical devices, financial services, employment law) apply on top of the AI Act.

## Related
- [[ai-security-privacy-compliance]] - GDPR, PII and security basics for AI features.
- [[model-registry-and-versioning]] - versioning and lineage evidence for audits.
- [[llm-observability]] - logging and post-market monitoring.
- [[llm-evaluation]] - accuracy and robustness evidence.
- [[guardrails-and-safety]] - runtime controls and human oversight.
- [[model-licenses]] - licence terms are separate from regulation.
- [[healthcare]], [[finance]] - sectors with high-risk use cases.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/security/privacy-and-pii-handling.md - privacy engineering around the AI system.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/security/threat-modeling.md - security risk assessment the AI Act expects.

## References
- Regulation (EU) 2024/1689 (AI Act): https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- Regulation (EU) 2026/1744 (Digital Omnibus on AI): https://eur-lex.europa.eu/eli/reg/2026/1744/oj
- Consolidated AI Act as of 2026-07-27: https://eur-lex.europa.eu/eli/reg/2024/1689/2026-07-27/eng
- Commission AI Act Service Desk timeline: https://ai-act-service-desk.ec.europa.eu/en/ai-act/timeline/timeline-implementation-eu-ai-act
- Council final approval (2026-06-29): https://www.consilium.europa.eu/en/press/press-releases/2026/06/29/artificial-intelligence-council-gives-final-green-light-to-simplify-and-streamline-rules/
- Council/Parliament agreement (2026-05-07): https://www.consilium.europa.eu/en/press/press-releases/2026/05/07/artificial-intelligence-council-and-parliament-agree-to-simplify-and-streamline-rules/
- ISO/IEC 42001:2023: https://www.iso.org/standard/42001
- NIST AI RMF: https://www.nist.gov/itl/ai-risk-management-framework
