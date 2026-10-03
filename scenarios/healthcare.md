---
title: Healthcare and clinical AI
category: scenarios
tags: [healthcare, hipaa, phi, clinical-notes, medical-imaging, ambient-scribe]
use_cases:
  - "summarise clinical notes or draft documentation from a visit transcript"
  - "build a patient-facing FAQ or triage assistant safely"
  - "analyse medical images with a computer vision model"
  - "extract structured data from medical records or lab reports"
  - "use LLMs with PHI while staying within HIPAA or GDPR"
status: draft
last_verified: 2026-10-03
sources:
  - https://www.hhs.gov/hipaa/for-professionals/index.html
  - https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
  - https://gdpr-info.eu/art-9-gdpr/
---

# Healthcare and clinical AI

## Summary
Healthcare offers high-value language and imaging tasks but is dominated by
privacy law, patient-safety risk and device regulation. Safe pattern: start
with administrative and documentation tasks where a clinician reviews the
output, keep PHI inside a contractually covered boundary, and treat anything
that influences diagnosis or treatment as potentially regulated software.

## Key concepts
- PHI/special-category data: US HIPAA covers protected health information
  handled by covered entities and their business associates; a vendor that
  processes PHI generally needs a Business Associate Agreement (BAA). In the
  EU, health data is special-category data under GDPR Article 9, with
  stricter lawful-basis rules. Check counsel for your case.
- De-identification (HIPAA Safe Harbor / Expert Determination) is a legal
  standard, not "remove the name". LLM-based redaction is not proof.
- Device regulation: software intended for diagnosis or treatment decisions
  may be a medical device (FDA SaMD in the US, MDR in the EU). Check the
  FDA's AI-enabled device guidance and your regulator before clinical use.
- EU AI Act: AI used in medical devices and some health-related uses are
  classed high-risk with conformity duties; check Regulation (EU) 2024/1689
  and its phase-in dates.
- Human in the loop: clinician sign-off is the control that makes
  documentation tools tractable.

## When to use / scenarios
1. **Ambient scribe / visit note drafting.** Problem: documentation burden.
   Approach: diarized transcript -> structured note (SOAP) draft -> clinician
   edits and signs. Read: [[speech-to-text]], [[speaker-diarization]],
   [[structured-output]], [[hallucination-and-grounding]].
2. **Clinical note summarisation / chart review.** Approach: long-context
   summarisation with source quotes for each claim. Read: [[long-context]],
   [[prompt-engineering]].
3. **Records extraction (labs, discharge summaries, codes).** Approach: parse
   + schema extraction, validate against code sets. Read:
   [[document-processing]], [[document-parsing]], [[ocr]].
4. **Medical coding assistance (ICD/CPT suggestions).** Approach: retrieval
   over code books, suggestions reviewed by a coder. Read: [[rag-basics]],
   [[embedding-models]].
5. **Patient FAQ / appointment / insurance bot.** Approach: RAG over approved
   material, no individual medical advice, emergency escalation text.
   Read: [[customer-support]], [[guardrails-and-safety]].
6. **Medical imaging (X-ray, pathology, dermatology).** Approach: supervised
   CV with clinical validation; likely regulated. Read: [[image-classification]],
   [[segmentation]], [[object-detection]], [[experiment-tracking]].
7. **Clinical-trial matching and literature search.** Approach: RAG over
   protocols/papers with citations. Read: [[advanced-rag]], [[vector-databases]].
8. **Risk prediction on tabular EHR data (readmission).** Approach: gradient
   boosting, calibration, subgroup audits. Read: [[gradient-boosting-tabular]],
   [[classic-ml-scikit-learn]].
9. **Vitals / wearable anomaly detection.** Read: [[anomaly-detection]],
   [[time-series-forecasting]], [[edge-on-device]].
10. **Synthetic data for development.** Read:
    [[data-labeling-and-synthetic-data]] (synthetic does not automatically
    equal de-identified).

## Setup & code
Reference architecture (flagship: ambient note drafting):

```
Audio (consent captured) -> STT + diarization (in-boundary or BAA-covered)
 -> PHI-aware prompt -> LLM drafts structured note (no new facts rule)
 -> grounding check: each statement links to transcript span
 -> clinician review UI (accept/edit) -> EHR write-back
 -> audit log (who, what model, version) ; retention per policy
```

```python
# Minimal draft step; deployment must run under a BAA / in-boundary model.
NOTE_SCHEMA = {"subjective": "", "objective": "", "assessment": "", "plan": ""}
prompt = ("Draft a SOAP note ONLY from the transcript. If something is not "
          "stated, write 'not documented'. Quote the supporting span for "
          "each statement.\n\nTranscript:\n" + transcript)
# call your model of choice; require JSON matching NOTE_SCHEMA;
# reject output whose quoted spans are not substrings of transcript.
```

## Choosing / trade-offs
- Hosted vs local: some providers offer BAAs or zero-data-retention terms for
  eligible products; confirm in writing. Otherwise run open-weights locally
  ([[ollama]], [[inference-servers-vllm]], [[meta-llama]], [[qwen]]).
- Assist vs autonomous: keep a clinician in the loop for anything entering
  the record.
- Specialised medical models vs general frontier models: evaluate on your own
  notes ([[llm-evaluation]]); do not trust leaderboard scores.

## Gotchas
- Sending PHI to a consumer chatbot or an API without a BAA is a compliance
  breach in the US; EU transfers need a GDPR transfer mechanism.
- Hallucinated medication, dose or laterality is the classic harm; require
  quote-level grounding and verification.
- Dataset bias: imaging and risk models perform worse on under-represented
  groups; report subgroup metrics.
- Automation bias: clinicians accept plausible drafts; sample audits help.
- Logs and traces contain PHI; scrub observability data ([[llm-observability]]).
- Do not present outputs as medical advice to patients.
- Retention and breach-notification duties apply to transcripts and drafts.

## Related
- [[ai-security-privacy-compliance]] - data boundary, DPAs, retention.
- [[guardrails-and-safety]] - refusal and escalation rules.
- [[voice-agents]] - patient phone workflows.
- [[vision-language-models]] - report generation research (not for unsupervised use).

## References
- HHS HIPAA for professionals: https://www.hhs.gov/hipaa/for-professionals/index.html
- FDA AI-enabled medical devices: https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices
- Regulation (EU) 2024/1689 (AI Act): https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- GDPR Article 9: https://gdpr-info.eu/art-9-gdpr/
