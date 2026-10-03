---
title: Legal and contract AI
category: scenarios
tags: [legal, contracts, e-discovery, citations, privilege, compliance]
use_cases:
  - "review contracts and extract clauses and obligations"
  - "build legal research or case-law Q&A with verifiable citations"
  - "support e-discovery document review and classification"
  - "draft or summarise legal documents with lawyer review"
  - "keep client confidentiality when using LLMs"
status: draft
last_verified: 2026-10-03
sources:
  - https://www.americanbar.org/news/abanews/aba-news-archives/2024/07/aba-issues-first-ethics-guidance-ai-tools/
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
  - https://law.justia.com/cases/federal/district-courts/new-york/nysdce/1:2022cv01461/575368/54/
---

# Legal and contract AI

## Summary
Legal work is text-heavy, precedent-driven and unforgiving of invented
facts. LLMs are effective at first-pass review, clause extraction,
summarisation and drafting, provided every output is grounded in source text
and a qualified lawyer takes responsibility. This is not legal advice; the
file describes engineering patterns.

## Key concepts
- Fabricated citations are the signature failure: courts have sanctioned
  lawyers for filing LLM-invented cases (Mata v. Avianca, S.D.N.Y. 2023).
  Every citation must be retrieved and verified against a real source.
- Professional duties: confidentiality, competence and supervision apply to
  AI tools. The ABA issued Formal Opinion 512 (2024) on generative AI; check
  your jurisdiction's bar guidance.
- Privilege and confidentiality: sending client documents to a third-party
  model needs vendor terms (no training, retention limits) and client
  consent where required.
- Jurisdiction matters: the same clause reads differently by governing law;
  keep jurisdiction as metadata in retrieval.
- EU AI Act: some AI used in administration of justice is listed as
  high-risk; check Annex III of Regulation (EU) 2024/1689.

## When to use / scenarios
1. **Contract review against a playbook.** Problem: reviewers re-read NDAs and
   MSAs. Approach: extract clauses to a schema, compare to approved positions,
   flag deviations with quoted text. Read: [[structured-output]],
   [[document-parsing]], [[document-processing]].
2. **Obligation and date extraction.** Approach: schema with source span and
   page for every field; calendar integration. Read: [[structured-output]],
   [[tool-calling]].
3. **Legal research Q&A.** Approach: RAG over a trusted corpus (statutes,
   cases), citations are links to retrieved passages only. Read: [[rag-basics]],
   [[advanced-rag]], [[rerankers]], [[hallucination-and-grounding]].
4. **E-discovery review / technology-assisted review.** Approach: classifier
   with active learning plus LLM relevance reasons, sampling-based recall
   measurement. Read: [[nlp-classic-tasks]], [[embeddings]],
   [[llm-evaluation]].
5. **Due-diligence data room analysis.** Approach: batch extraction over
   hundreds of documents into a risk matrix. Read: [[document-processing]],
   [[cost-and-latency]].
6. **First-draft generation (letters, clauses).** Approach: template + retrieved
   precedent clauses, lawyer edits. Read: [[prompt-engineering]],
   [[long-context]].
7. **Document comparison / redlining summaries.** Approach: diff in code, LLM
   explains material changes. Read: [[long-context]], [[structured-output]].
8. **Client intake and triage chatbot.** Approach: collect facts, no advice,
   conflict-check handoff. Read: [[customer-support]], [[guardrails-and-safety]].
9. **Regulatory change monitoring.** Approach: crawl sources, classify
   relevance, summarise. Read: [[agents]], [[advanced-rag]].
10. **Transcription of hearings and depositions.** Read: [[speech-to-text]],
    [[speaker-diarization]].

## Setup & code
Reference architecture (flagship: playbook contract review):

```
contract PDF -> parse (keep page + char offsets) -> clause segmentation
  -> per clause: LLM classify + extract (schema incl. quote + page)
  -> compare with playbook (rules in code, LLM explains gap)
  -> verify: every quote is a substring of the source (reject otherwise)
  -> reviewer UI: highlights, accept/override -> report + audit log
```

```python
def verified(item, source_text):
    # reject any extraction whose quoted evidence is not in the document
    return item["quote"] in source_text

findings = [f for f in llm_findings if verified(f, doc_text)]
```

## Choosing / trade-offs
- General LLM vs legal-specialised vendor: vendors bring curated corpora and
  citators; general models need your own retrieval. Evaluate on your own
  matters ([[llm-evaluation]]).
- Long context vs RAG: a single contract fits in context; a case-law corpus
  needs retrieval ([[long-context]], [[vector-databases]]).
- Hosted vs local: client confidentiality may push to local models
  ([[ollama]], [[meta-llama]], [[qwen]]) or private deployments.

## Gotchas
- Invented or mis-quoted authorities; never let the model emit a citation
  that did not come from retrieval, and check currency (overruled law).
- Missing a clause is worse than a false flag: measure recall on a labelled
  set of past reviews.
- Prompt injection inside counterparty documents ([[prompt-injection]]).
- Logs hold privileged material; restrict access and retention
  ([[ai-security-privacy-compliance]]).
- Unauthorised practice of law if a consumer-facing tool gives individualised
  advice; check local rules.
- Automation bias: reviewers skim AI summaries; sample audits.

## Related
- [[finance]] - regulatory and compliance overlap.
- [[llm-observability]] - audit trails.
- [[model-licenses]] - terms when embedding models in a product.

## References
- ABA Formal Opinion 512 announcement: https://www.americanbar.org/news/abanews/aba-news-archives/2024/07/aba-issues-first-ethics-guidance-ai-tools/
- Regulation (EU) 2024/1689 (AI Act): https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- Mata v. Avianca, No. 22-cv-1461 (S.D.N.Y.): https://law.justia.com/cases/federal/district-courts/new-york/nysdce/1:2022cv01461/575368/54/ (verify the link before citing)
