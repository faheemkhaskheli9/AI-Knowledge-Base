---
title: Government and public sector AI
category: scenarios
tags: [government, public-sector, citizen-services, procurement, transparency, accessibility, ai-act, omb]
use_cases:
  - "build a citizen-facing assistant for government services"
  - "process forms, applications and casework documents in an agency"
  - "buy or procure an AI system for a public body"
  - "publish transparency records for algorithms used by government"
  - "search and summarise legislation, policy and consultation responses"
status: draft
last_verified: 2026-10-15
sources:
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
  - https://www.whitehouse.gov/wp-content/uploads/2025/02/M-25-21-Accelerating-Federal-Use-of-AI-through-Innovation-Governance-and-Public-Trust.pdf
  - https://www.gov.uk/government/collections/algorithmic-transparency-recording-standard-hub
  - https://www.nist.gov/itl/ai-risk-management-framework
  - https://www.w3.org/TR/WCAG22/
---

# Government and public sector AI

## Summary
Public bodies use AI for citizen help (answering questions, guiding
applications), back-office document work (forms, casework, correspondence,
FOI requests) and analysis (consultations, legislation, fraud and error).
Compared with the private sector, the bar is higher on fairness, due
process, transparency, accessibility, records retention and procurement
rules, because people often cannot choose another provider. Safe pattern:
assist staff and citizens with grounded, cited answers, and keep
eligibility and enforcement decisions with accountable officials.

## Key concepts
- Rights-affecting decisions: benefits, immigration, policing, tax and
  child-welfare decisions need human accountability, reasons the person
  can understand, and a route to appeal. Do not let a model deny a
  service on its own.
- EU AI Act: Annex III lists as high-risk, among others, AI used by or
  for public authorities to evaluate eligibility for essential public
  assistance benefits and services, and uses in law enforcement,
  migration and the administration of justice. Public-body deployers of
  high-risk systems must carry out a fundamental-rights impact assessment
  (Article 27). Social scoring is prohibited (Article 5). Application
  dates for high-risk duties were amended in 2026; check the current text.
- US federal: OMB memo M-25-21 (April 2025, replacing M-24-10) requires
  Chief AI Officers, agency AI strategies and minimum risk practices for
  "high-impact" AI; its companion M-25-22 covers AI acquisition. Agencies
  also publish AI use-case inventories. States and cities have their own
  rules.
- UK: the Algorithmic Transparency Recording Standard (ATRS) is mandatory
  for central government departments and some arm's-length bodies and
  recommended for the wider public sector.
- Risk framework: the NIST AI RMF (Govern, Map, Measure, Manage) is a
  common, vendor-neutral structure for risk work and procurement
  questions.
- Accessibility and language: public services must meet accessibility
  standards (e.g. WCAG; Section 508 in the US; the EU Web Accessibility
  Directive) and often serve many languages and low-literacy users.
- Records and FOI: prompts, outputs and logs may be public records,
  subject to retention schedules and freedom-of-information requests.
- Sovereignty and hosting: data residency, security accreditation of cloud
  services (e.g. FedRAMP in the US) and vendor lock-in shape model choice.

## When to use / scenarios
1. **Citizen service assistant.** Problem: people cannot find the right
   rule or form. Approach: RAG over official guidance only, citations to
   the page, plain-language answers, multilingual, hand-off to a human
   and no eligibility decisions. Read: [[rag-basics]], [[advanced-rag]],
   [[guardrails-and-safety]], [[customer-support]].
2. **Forms, applications and casework intake.** Approach: OCR/parse,
   schema extraction, validation against registers, caseworker review.
   Read: [[document-processing]], [[document-parsing]], [[ocr]],
   [[structured-output]].
3. **Correspondence and FOI drafting.** Approach: draft replies and
   redaction suggestions for officials to approve; log what was used.
   Read: [[prompt-engineering]], [[hallucination-and-grounding]],
   [[ai-security-privacy-compliance]].
4. **Consultation and petition analysis.** Approach: clustering and
   topic labelling of thousands of responses, with sampled human checks
   and quotes. Read: [[topic-modeling]], [[clustering]], [[embeddings]],
   [[nlp-classic-tasks]].
5. **Legislation and policy search.** Approach: hybrid search over
   statutes and guidance with versioning by date in force. Read:
   [[advanced-rag]], [[long-context]], [[legal]].
6. **Fraud and error detection in payments and grants.** Approach: risk
   scoring to prioritise reviews, never automatic sanctions; fairness
   audit by group. Read: [[anomaly-detection]],
   [[gradient-boosting-tabular]], [[model-interpretability]].
7. **Contact-centre voice and translation.** Read: [[speech-to-text]],
   [[text-to-speech]], [[voice-agents]].
8. **Internal staff copilots (policy Q&A, coding, analysis).** Read:
   [[software-engineering]], [[data-analytics]], [[text-to-sql]].
9. **Land, environment and infrastructure monitoring.** Approach:
   satellite and camera analytics for change detection. Read:
   [[agriculture]], [[segmentation]], [[video-analytics]].
10. **Education and health services run by the state.** Read:
    [[education]], [[healthcare]].

## Setup & code
Reference architecture (flagship: citizen service assistant):

```
citizen question (web, phone, many languages)
  -> retrieval over approved, dated guidance (versioned corpus)
  -> answer only if evidence is strong; cite page + date; else hand off
  -> no eligibility decisions; link to the official application or caseworker
  -> logs kept per records schedule; PII redacted from analytics
  -> transparency record published (ATRS / AI use-case inventory / FRIA)
  -> accessibility test (WCAG) and per-language evaluation before launch
```

Runnable sketch: answer only from retrieved official text, otherwise hand
off (TF-IDF retrieval with a confidence threshold; swap in your own search).

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = {
    "gov/parking-permit#apply": "Apply for a resident parking permit online "
        "with proof of address and vehicle registration.",
    "gov/waste#bulky": "Book a bulky waste collection online; a fee applies "
        "per item collected.",
    "gov/benefits#housing": "Housing support eligibility is decided by a "
        "caseworker after you submit an application with income evidence.",
}
ids, texts = list(docs), list(docs.values())
vec = TfidfVectorizer(stop_words="english").fit(texts)
D = vec.transform(texts)

def answer(question, threshold=0.2):
    sims = cosine_similarity(vec.transform([question]), D)[0]
    best = sims.argmax()
    if sims[best] < threshold:
        return "HANDOFF: connecting you to a staff member."
    return f"{texts[best]} [source: {ids[best]}]"

print(answer("How do I apply for a parking permit?"))
print(answer("Can you tell me if my visa will be approved?"))
```

## Choosing / trade-offs
- Build vs buy: buying is faster, but contracts must cover data use (no
  training on citizen data without agreement), audit and evaluation
  access, model change notice, exit and portability; open standards
  reduce lock-in.
- Hosted frontier model vs self-hosted open weights: hosted gives quality
  and speed; self-hosted gives residency and control at higher operating
  effort ([[inference-servers-vllm]], [[managed-cloud-ai-platforms]],
  [[model-licenses]]).
- Generative vs extractive answers: extractive (quote the official text)
  is safer for legal rights; generative plain-language rewording helps
  understanding but needs grounding checks.
- Chat vs better forms: often a clearer form or page fixes the problem
  more cheaply than an assistant; measure task completion, not chats.

## Gotchas
- Outdated guidance in the index gives confidently wrong answers; version
  documents by date in force and retire old ones.
- Answers that sound like decisions ("you are eligible") create legal
  expectations; phrase as guidance and link to the official process.
- Fraud models trained on past investigations inherit past targeting
  bias; audit by group and add random sampling.
- Prompt injection through uploaded documents or emails can manipulate
  casework tools ([[prompt-injection]]).
- Lower accuracy in minority languages and for non-standard speech or
  writing harms the people who most need help; evaluate per language
  ([[llm-evaluation]]).
- Logs with citizen PII need minimisation, access control and retention
  rules ([[llm-observability]]).

## Related
- [[legal]] - legislation, case law and legal-document patterns.
- [[insurance]] - social insurance and claims-style casework.
- [[agriculture]] - subsidy checks and land monitoring.
- [[logistics-and-supply-chain]] - public procurement and emergency supplies logistics.
- [[ai-regulation-checklist]] - EU AI Act and ISO/IEC 42001 checklist for public deployers.
- [[satellite-and-remote-sensing]] - Earth-observation monitoring for agencies.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/security/privacy-and-pii-handling.md

## References
- Regulation (EU) 2024/1689 (AI Act), Articles 5, 27 and Annex III: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- OMB M-25-21, Accelerating Federal Use of AI through Innovation, Governance, and Public Trust: https://www.whitehouse.gov/wp-content/uploads/2025/02/M-25-21-Accelerating-Federal-Use-of-AI-through-Innovation-Governance-and-Public-Trust.pdf
- UK Algorithmic Transparency Recording Standard hub: https://www.gov.uk/government/collections/algorithmic-transparency-recording-standard-hub
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework
- WCAG 2.2: https://www.w3.org/TR/WCAG22/
