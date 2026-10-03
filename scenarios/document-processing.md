---
title: Document processing and extraction
category: scenarios
tags: [idp, ocr, extraction, invoices, forms, pdf, structured-output]
use_cases:
  - "extract fields from invoices, receipts or forms into JSON"
  - "turn scanned PDFs into searchable, structured data"
  - "classify and route incoming documents"
  - "summarise or compare long contracts and reports"
  - "build a document Q&A system over a PDF archive"
status: draft
last_verified: 2026-10-03
sources:
  - https://docs.anthropic.com/en/docs/build-with-claude/pdf-support
  - https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs
---

# Document processing and extraction

## Summary
Intelligent document processing (IDP) turns unstructured files (PDF, scans,
images, email) into validated structured data or answers. The modern stack is:
parse/OCR to text or layout, an LLM or VLM to extract against a schema,
deterministic validation, and human review for low-confidence items.

## Key concepts
- Input quality dominates: native-text PDF, scan, phone photo and handwriting
  need different tools ([[document-parsing]], [[ocr]]).
- Schema-first: define the output schema and validators before prompting
  ([[structured-output]]). Validate totals, dates, checksums in code.
- Per-field confidence and a review queue beat chasing 100% automation.
- Accuracy needs differ: invoice totals and bank details need near-perfect
  accuracy plus checks; summaries tolerate more variance.
- Layout matters: tables, multi-column pages and stamps break naive text
  extraction.

## When to use / scenarios
1. **Invoice / receipt extraction.** Problem: manual data entry for AP.
   Approach: parse -> VLM or LLM with schema -> validate (line items sum to
   total, vendor lookup) -> review exceptions. Read: [[structured-output]],
   [[vision-language-models]], [[ocr]].
2. **Forms and KYC documents (IDs, utility bills).** Approach: OCR + VLM,
   cross-check fields, human review; privacy-heavy. Read: [[ocr]],
   [[finance]], [[ai-security-privacy-compliance]].
3. **Document classification and routing.** Approach: LLM few-shot or embedding
   classifier on first pages. Read: [[nlp-classic-tasks]], [[embeddings]].
4. **Contract/report Q&A.** Approach: layout-aware chunking, hybrid retrieval,
   citations to page. Read: [[rag-basics]], [[advanced-rag]], [[legal]].
5. **Long document summarisation.** Approach: direct if it fits the window,
   else map-reduce. Read: [[long-context]], [[prompt-caching-and-cost]].
6. **Table and financial statement extraction.** Approach: layout parser for
   tables, LLM only for semantics. Read: [[document-parsing]], [[finance]].
7. **Medical record intake.** Approach: as above with PHI controls. Read:
   [[healthcare]].
8. **Handwriting and historical archives.** Approach: VLM or specialised HTR,
   expect high error, sample-based QA. Read: [[ocr]], [[vision-language-models]].
9. **Email and attachment triage.** Approach: classify, extract, route; agent
   for follow-ups. Read: [[agents]], [[personal-assistants]].
10. **Bulk back-file digitisation.** Approach: batch API, cheap model first
    pass, escalate uncertain pages. Read: [[cost-and-latency]],
    [[small-language-models]].

## Setup & code
Reference architecture (flagship: invoices):

```
ingest (S3/email) -> type detect -> parse (text layer else OCR/VLM)
  -> page images + text -> LLM extract to schema
  -> validate (sum checks, date formats, vendor master, duplicates)
  -> confidence gate: auto-post | human review UI
  -> store source file + extraction + reviewer corrections
  -> corrections feed eval set / later fine-tune
```

```python
import anthropic, base64, json
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str; invoice_no: str; date: str; total: float; currency: str

client = anthropic.Anthropic()
pdf = base64.standard_b64encode(open("inv.pdf", "rb").read()).decode()
r = client.messages.create(
    model="claude-sonnet-5-5", max_tokens=800,
    messages=[{"role": "user", "content": [
        {"type": "document", "source": {"type": "base64",
         "media_type": "application/pdf", "data": pdf}},
        {"type": "text", "text": "Return JSON only: " +
         json.dumps(Invoice.model_json_schema())}]}])
inv = Invoice.model_validate_json(r.content[0].text)  # raises if malformed
assert inv.total >= 0
```
Check [[structured-output]] for provider-enforced schemas that replace the
"JSON only" instruction.

## Choosing / trade-offs
- Classic OCR + rules vs VLM: rules are cheap and deterministic for fixed
  templates; VLMs handle layout variety with no template work but cost more
  per page.
- Local vs API: sensitive documents may require local models
  ([[qwen]], [[ollama]]) or a provider with a signed data agreement.
- Direct PDF input vs separate parsing: direct is simpler; a parser gives
  more control, bounding boxes for review UIs, and cheaper reruns.

## Gotchas
- LLMs "repair" illegible digits plausibly. Always cross-validate numbers.
- Silent truncation of long pages or many-page PDFs; check page limits.
- Duplicates and re-submissions; keep file hashes.
- Prompt injection hidden in documents ([[prompt-injection]]).
- Retention of source documents: personal data rules (e.g. GDPR data
  minimisation) apply; see [[ai-security-privacy-compliance]].
- Measure per-field accuracy on real samples, not overall "looks right".

## Related
- [[data-labeling-and-synthetic-data]] - building the labelled set.
- [[llm-evaluation]] - field-level scoring.
- [[mlops-lifecycle]] - drift when vendors change templates.

## References
- Anthropic PDF support: https://docs.anthropic.com/en/docs/build-with-claude/pdf-support
- Anthropic structured outputs: https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs
