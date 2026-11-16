---
title: PII redaction with placeholder round-trip for LLM calls
category: llm-apps
tags: [pii, phi, redaction, pseudonymization, de-identification, placeholders, hipaa, privacy, guards]
use_cases:
  - "send clinical notes, HR cases or support tickets to a hosted LLM without names or record numbers leaving our boundary"
  - "the reply must still name the right person or record, so masking with *** is not enough"
  - "stop a PII scrubber from mangling lab values, dosages, blood pressure or billing codes"
  - "keep the same placeholder for the same person across a multi-turn chat"
  - "review a redaction step for silent over- or under-redaction"
status: draft
last_verified: 2026-11-07
sources:
  - https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
  - https://microsoft.github.io/presidio/anonymizer/
---

# PII redaction with placeholder round-trip for LLM calls

## Summary
Replace each sensitive span with a stable numbered placeholder (`[PERSON_1]`, `[MRN_1]`) before text crosses the trust boundary, keep the placeholder-to-value map server-side, and decode the model's reply back through it. The same value always maps to the same placeholder, so the model can still reason about "the same patient", and the output stays usable. Domain guards stop the redactor from eating legitimate content that merely looks like an identifier.

## Key concepts
- **Placeholders, not masks.** Number per label and per normalized value (case/whitespace-insensitive), so repeats collapse onto one placeholder and the reply is restorable.
- **Placeholder shape.** `[LABEL_n]` survives HTML sanitizers and markdown renderers; `<LABEL_n>` is treated as a tag, and `<`/`>` are real operators in clinical text (`eGFR <60`).
- **Anchored rules first.** Bare digit runs are ambiguous. Anchor on field labels (`MRN:`, `Phone:`), titles (`Dr.`), credentials (`, MD`), address context or a checksum (Luhn, NPI). Keep the anchor outside the replaced span (`MRN: [MRN_1]`) so field meaning survives. Statistical NER (e.g. Presidio's analyzer) is an optional recall layer for names in prose.
- **Guards are subtractive.** A domain profile emits protected spans (measurements, dosages, codes); any candidate overlapping one is dropped. Users add guards for their own look-alikes without touching detection.
- **Known-value propagation.** Once an anchored rule establishes a value as sensitive, redact its other bare occurrences in the document and later turns. Skip short or purely numeric values (`92`, `1980`) that collide with measurements, and still pass propagated hits through guards.
- **Lenient decoding bounded by the map.** Models drop brackets, change case and swap `_` for a space. Accept `PERSON 1`, `person_1`, `[[PERSON_1]]`, but only for labels in the map, with a word boundary before the label (`PAGE 1` is not `AGE_1`) and none after the index (`AGE_12` is not `AGE_1`). Unknown placeholders stay verbatim.
- **The map is the sensitive data.** Store it server-side, write it atomically (temp file + `os.replace`), validate its schema on load.
- **Redaction is not de-identification.** HIPAA de-identification is Safe Harbor (18 listed identifier types removed, free text included, and no actual knowledge of re-identifiability) or Expert Determination. A reversible placeholder map is pseudonymization: the data stays PHI on your side.

## When to use / scenarios
- Regulated or confidential text needs an LLM feature and the vendor is outside the compliance boundary ([[healthcare]], [[legal]], HR, support).
- Outputs must name people or records: summaries, letters, follow-up emails.
- Corpora full of numerics that generic scrubbers break: `BP 120/80`, `500 mg`, CPT `99213`, ICD `E11.9`.
- Not enough on its own when the vendor contract or law forbids sending the record at all; then self-host ([[ai-security-privacy-compliance]]).

## Setup & code
Anchored rules, a measurement guard, a stable map and lenient restore, stdlib only:
```python
import re

class PlaceholderMap:
    def __init__(self):
        self.by_value, self.values = {}, {}
    def placeholder_for(self, label, value):
        key = (label, " ".join(value.lower().split()))
        if key not in self.by_value:
            n = sum(1 for l, _ in self.by_value if l == label) + 1
            self.by_value[key] = f"[{label}_{n}]"
            self.values[(label, n)] = value
        return self.by_value[key]
    def restore(self, text):
        labels = "|".join(sorted({l for l, _ in self.values}, key=len, reverse=True)) or "(?!)"
        pat = re.compile(rf"\[*\b({labels})[ _](\d+)(?!\d)\]*", re.I)
        def sub(m):
            v = self.values.get((m.group(1).upper(), int(m.group(2))))
            return v if v is not None else m.group(0)
        return pat.sub(sub, text)

RULES = [  # (label, pattern with a (?P<v>...) group; anchor stays outside it)
    ("PERSON", r"\b(?:Dr\.|Mr\.|Ms\.|Patient:)\s+(?P<v>[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"),
    ("MRN", r"\bMRN:?\s*(?P<v>\d{6,10})\b"),
    ("PHONE", r"\b(?:Phone|Tel):?\s*(?P<v>\+?\d[\d\s().-]{7,}\d)"),
]
GUARDS = [r"\b\d{2,3}/\d{2,3}\b(?!/\d)", r"\b\d+(?:\.\d+)?\s?(?:mg|mL|mmol/L|mL/min)\b"]

def redact(text, pmap):
    protected = [m.span() for g in GUARDS for m in re.finditer(g, text)]
    hits = []
    for label, pat in RULES:
        for m in re.finditer(pat, text):
            s, e = m.span("v")
            if not any(s < pe and ps < e for ps, pe in protected):
                hits.append((s, e, label, m.group("v")))
    for s, e, label, v in sorted(hits, reverse=True):   # replace right-to-left keeps offsets valid
        text = text[:s] + pmap.placeholder_for(label, v) + text[e:]
    for (label, _), v in list(pmap.values.items()):     # known-value propagation
        if len(v) > 3 and not v.isdigit():
            text = re.sub(rf"\b{re.escape(v)}\b", pmap.placeholder_for(label, v), text)
    return text

pmap = PlaceholderMap()
note = "Patient: John Doe, MRN: 4481920. BP 120/80, metformin 500 mg. Call John Doe at Phone: 555-201-7781."
safe = redact(note, pmap)
assert "John" not in safe and "4481920" not in safe and "555-201" not in safe
assert "120/80" in safe and "500 mg" in safe and safe.count("[PERSON_1]") == 2
reply = "Dear person 1, your record [MRN_1] is updated. PAGE 1 attached."   # model mangled the shape
assert pmap.restore(reply) == "Dear John Doe, your record 4481920 is updated. PAGE 1 attached."
```
For production recall, add an NER analyzer (Presidio's `AnalyzerEngine`) as another rule source behind the same guards and map; Presidio's anonymizer also ships reversible operators (encrypt/decrypt) when a placeholder map is not wanted.

## Choosing / trade-offs
- **Precision vs recall**: anchored regex is precise and explainable but misses names in free prose; NER finds them but over-fires on drug and place names. Layer them, with guards applied to both.
- **Over-redaction is the safer failure** but still costs the model context; spend effort on negative tests and stop-word lists (`Patient Summary`), not on loosening anchors.
- **Placeholder vs surrogate values** (fake but realistic names): surrogates read naturally but can be mistaken for real data downstream; placeholders are unambiguous.
- **Strict on explicit input**: an unknown profile, bad template or malformed map file should raise, never fall back to defaults the caller did not choose.

## Gotchas
- Bare `\d{5}` for ZIP eats CPT codes; a loose phone regex eats `500 mg` and `120/80`. Write the negative test first.
- A blood-pressure guard `\d{2,3}/\d{2,3}` swallows dates like `12/25/2020` unless it refuses a following `/digit`.
- Whitespace-tolerant decoders with `\s*` outside optional brackets eat the spaces around a bare `PERSON_1` and glue words together.
- Name tokens must never let `.` join words, or `Watson. John` becomes one name.
- HTML pre-cleaning with `<[^>]+>` deletes `<60 mL/min and K >`; strip only real tags (`</?[A-Za-z][^>]*>`).
- "Value already in the map" is not "already redacted": propagated matches still go through guards and the allowlist.
- Test fixtures must not hold real-looking identifiers; use checksum-valid unassigned numbers and example domains.
- The model can still infer identity from context (rare condition + town + job title); placeholders do not remove quasi-identifiers.

## Related
- [[ai-security-privacy-compliance]] - data boundary, DPAs, the regex baseline this extends.
- [[guardrails-and-safety]] - where redaction sits among input/output controls.
- [[agent-memory-lifecycle]] - isolation, audit and erasure for stored memories.
- [[healthcare]] - de-identification standards and clinical constraints.

## References
- HHS, Guidance on de-identification of PHI (Safe Harbor, Expert Determination): https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
- Microsoft Presidio anonymizer and deanonymizer: https://microsoft.github.io/presidio/anonymizer/
