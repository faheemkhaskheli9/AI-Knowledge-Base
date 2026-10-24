---
title: PII redaction and pseudonymization for LLM apps
category: deployment
tags: [pii, redaction, pseudonymization, anonymization, presidio, privacy, gdpr, de-identification, logging, data-minimization]
use_cases:
  - "remove names, emails and card numbers from support tickets before sending them to an LLM API"
  - "replace PII with placeholders and put the real values back into the model's answer"
  - "keep PII out of LLM traces, logs and eval datasets"
  - "measure how much PII our redaction step misses before we rely on it"
status: draft
last_verified: 2026-10-24
sources:
  - https://github.com/data-privacy-stack/presidio
  - https://raw.githubusercontent.com/microsoft/presidio/main/docs/getting_started/getting_started_text.md
  - https://raw.githubusercontent.com/microsoft/presidio/main/docs/anonymizer/index.md
  - https://pypi.org/project/presidio-analyzer/
  - https://eur-lex.europa.eu/eli/reg/2016/679/oj
  - https://csrc.nist.gov/pubs/sp/800/122/final
---

# PII redaction and pseudonymization for LLM apps

## Summary
Data minimization for LLM apps means the model, the provider and your logs see only the personal data the task needs. A redaction step detects personally identifiable information (PII) in text, replaces it with typed placeholders such as `<EMAIL_1>`, sends the placeholder text to the model, and, when the answer must contain the real values, swaps them back from a mapping that never leaves your server. Detection combines patterns with validation (emails, phone numbers, card numbers with a Luhn check, IDs) and named-entity recognition for names, places and organizations; Microsoft Presidio packages both. No detector finds everything, so measure recall on your own data and keep redaction as one layer next to contracts, retention settings and access control ([[ai-security-privacy-compliance]]).

## Key concepts
- **Detect, then transform**: detection returns spans with an entity type and a score; a separate step decides what to do with each type. Presidio splits these into `presidio-analyzer` (recognizers, spaCy or transformers NER) and `presidio-anonymizer` (operators).
- **Operators**: *redact* (delete), *mask* (`****1234`), *replace* with a type tag, *hash* (salted; a consistent salt keeps the same value mapping to the same hash across records), *encrypt* (reversible with a key), *keep*. Presidio's anonymizer ships these, plus custom functions; its deanonymizer reverses `encrypt` with `decrypt`.
- **Pseudonymization vs anonymization**: replacing values with tokens you can map back is pseudonymization; under GDPR pseudonymized data is still personal data because the mapping exists. Anonymized data cannot be linked back to a person. Most LLM pipelines pseudonymize.
- **Consistent placeholders**: the same value gets the same token within a conversation (`<PERSON_1>` everywhere Jane appears) so the model can still reason about who did what. Typed tokens keep more meaning than a generic `[REDACTED]`.
- **Restore (re-identification)**: replace tokens in the model's output with the original values only at the edge, for the user who is allowed to see them. The mapping is sensitive data itself: keep it in memory or an access-controlled store with a short TTL.
- **Validation cuts false positives**: a 16-digit number is a card only if it passes the Luhn checksum; an IBAN has its own check digits. Pattern plus checksum is far more precise than pattern alone.
- **Context words**: Presidio recognizers can raise a score when words like "phone" or "SSN" appear nearby; low-score spans can be kept or sent for review depending on risk.
- **Recall is the safety metric**: a missed PII span leaks; a false positive only costs answer quality. Evaluate on a labelled sample of your own traffic, per entity type.

## When to use / scenarios
- Support, HR or healthcare text sent to a hosted LLM where policy says the provider must not receive identifiers: pseudonymize before the call, restore in the reply ([[customer-support]], [[healthcare]]).
- Tracing and logging prompts and completions: redact before the span or log is written, not after export ([[llm-observability]], [[opentelemetry-genai-tracing]]).
- Building eval or fine-tuning datasets from production traffic: redact, then sample-review, before anything leaves the production boundary ([[llm-evaluation]], [[synthetic-data-for-llm-training]]).
- RAG over documents with personal data: redact at ingestion if answers never need the values; otherwise enforce access control at retrieval instead, because redacted chunks cannot answer "what is Jane's address" ([[rag-basics]]).
- Not when the task needs the identifier itself (draft a letter to a named customer, verify an address): use a provider and deployment the data agreement allows, with retention settings, instead of breaking the task ([[managed-cloud-ai-platforms]]).
- Not as the only control for regulated data (health records, special-category data): pair it with legal review and contractual terms ([[ai-regulation-checklist]]).

## Setup & code
Presidio (Python packages plus a spaCy English model; Docker images for the analyzer and anonymizer services also exist):
```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```
<!-- skip-check: needs presidio and a downloaded spaCy model -->
```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

text = "Hi, I'm Jane Doe, call me on 212-555-5555 or mail jane.doe@example.com"
analyzer = AnalyzerEngine()                                  # loads the spaCy model and built-in recognizers
results = analyzer.analyze(text=text, language="en",
                           entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD"])
for r in results:
    print(r.entity_type, text[r.start:r.end], round(r.score, 2))

anonymized = AnonymizerEngine().anonymize(
    text=text, analyzer_results=results,
    operators={"DEFAULT": OperatorConfig("replace"),         # no new_value -> "<ENTITY_TYPE>"
               "PHONE_NUMBER": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 8, "from_end": False})},
)
print(anonymized.text)
```
Runnable, stdlib only: a pseudonymizer with consistent typed placeholders, Luhn-validated card detection, restore of the model's output (including placeholders split across streamed chunks), and a recall/precision check against a hand-labelled sample. Names are detected here from a supplied list only, to show why NER is needed in practice.
```python
import re

PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?!\d)"),
    "CARD": re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
}

def luhn_ok(number):
    digits = [int(c) for c in number if c.isdigit()][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return len(digits) >= 13 and total % 10 == 0

class Pseudonymizer:
    """Replace PII with <TYPE_n> tokens; the same value always gets the same token."""
    def __init__(self, known_names=()):
        self.forward, self.reverse, self.counts = {}, {}, {}
        self.names = sorted(known_names, key=len, reverse=True)

    def _token(self, kind, value):
        key = (kind, value)
        if key not in self.forward:
            self.counts[kind] = self.counts.get(kind, 0) + 1
            tok = f"<{kind}_{self.counts[kind]}>"
            self.forward[key], self.reverse[tok] = tok, value
        return self.forward[key]

    def spans(self, text):
        found = []
        for kind in ("EMAIL", "CARD", "PHONE"):               # cards before phones: a card contains phone-like runs
            for m in PATTERNS[kind].finditer(text):
                if kind == "CARD" and not luhn_ok(m.group()):
                    continue
                found.append((m.start(), m.end(), kind))
        for name in self.names:                               # stand-in for NER (Presidio PERSON, spaCy, ...)
            for m in re.finditer(r"\b" + re.escape(name) + r"\b", text):
                found.append((m.start(), m.end(), "PERSON"))
        found.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        merged, end = [], -1
        for s in found:                                       # drop spans overlapping an earlier, longer one
            if s[0] >= end:
                merged.append(s); end = s[1]
        return merged

    def redact(self, text):
        out, last = [], 0
        for start, stop, kind in self.spans(text):
            out += [text[last:start], self._token(kind, text[start:stop])]
            last = stop
        return "".join(out + [text[last:]])

    def restore(self, text):
        return re.sub(r"<[A-Z]+_\d+>", lambda m: self.reverse.get(m.group(), m.group()), text)

p = Pseudonymizer(known_names=["Jane Doe", "Jane"])
ticket = ("Jane Doe (jane.doe@example.com, +1 212-555-0187) was charged twice on card "
          "4111 1111 1111 1111. Jane says order 4111 1111 1111 1112 is unrelated.")
safe = p.redact(ticket)
print(safe)
assert "jane.doe@example.com" not in safe and "4111 1111 1111 1111" not in safe
assert "4111 1111 1111 1112" in safe                     # card-shaped but fails Luhn: an order number

model_reply = "Refund issued to <PERSON_1>; confirmation sent to <EMAIL_1>."   # what the LLM wrote
print(p.restore(model_reply))

# Streaming: a token can be split across chunks ("<EMA" + "IL_1>"); hold back a possible partial token.
def restore_stream(chunks, pseudo):
    buf = ""
    for chunk in chunks:
        buf += chunk
        cut = buf.rfind("<")
        if cut != -1 and ">" not in buf[cut:] and len(buf) - cut <= 24:
            ready, buf = buf[:cut], buf[cut:]
        else:
            ready, buf = buf, ""
        if ready:
            yield pseudo.restore(ready)
    if buf:
        yield pseudo.restore(buf)

print("".join(restore_stream(["Refund to <PER", "SON_1>, mail <EMA", "IL_1> now."], p)))

# Measure on a labelled sample: recall per type is what tells you about leaks.
labelled = [
    ("Call Bob Smith on 415-555-0100", [("Bob Smith", "PERSON"), ("415-555-0100", "PHONE")]),
    ("bob@corp.io paid with 5500 0000 0000 0004", [("bob@corp.io", "EMAIL"), ("5500 0000 0000 0004", "CARD")]),
    ("Jane emailed jane@x.org", [("Jane", "PERSON"), ("jane@x.org", "EMAIL")]),
]
detector = Pseudonymizer(known_names=["Jane"])                # does not know "Bob Smith"
tp = fn = fp = 0
missed = []
for text, gold in labelled:
    pred = {(text[s:e], k) for s, e, k in detector.spans(text)}
    gold = set(gold)
    tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
    missed += sorted(gold - pred)
print(f"recall {tp / (tp + fn):.2f}, precision {tp / (tp + fp):.2f}, missed {missed}")
```

## Choosing / trade-offs
| Approach | Pick when | Watch out for |
|---|---|---|
| Regex + checksums | structured identifiers (emails, phones, cards, national IDs) | misses names, addresses and free-text identifiers |
| Presidio (patterns + spaCy/transformers NER) | general text in supported languages; want one open-source pipeline | NER quality drops on domain text, other languages and all-lowercase chat; tune recognizers and thresholds |
| Small fine-tuned NER model | high volume, domain-specific entity types | needs labelled data; evaluate per type |
| LLM-based detection | complex, contextual PII (indirect identifiers) | sends the raw text to a model, so it must be a model you are allowed to send it to (often local) |
| Managed cloud PII detection | already on that cloud; want a supported service | data goes to that service; per-call cost ([[managed-cloud-ai-platforms]]) |

- **Operator per type**: typed replacement for names and contacts the model must reason about; masking for numbers a user needs to recognize (last four digits); hashing with a fixed salt when records must be joined later without the raw value; deletion for data the task never needs.
- **Restore or not**: if the user never needs the real value in the answer, do not keep a reversible mapping at all; that removes a sensitive store.
- **Where it runs**: redact in the same trust zone as the source data (inside the service, before the gateway), so the raw text never crosses into logging or third-party SDKs.

## Gotchas
- Presidio's own README warns that automated detection gives no guarantee of finding all sensitive data; treat its output as reducing, not removing, risk.
- Over-redaction breaks tasks: dates, amounts, dosages and product codes look like PII to naive patterns. Scope rules per entity type and test answer quality before and after ([[llm-evaluation]]).
- The model may rewrite tokens (`<PERSON_1>'s`, `PERSON_1`, translated tags), so restore misses them; instruct it to copy tokens verbatim and check for unmatched tokens in the output.
- "Jane Doe" and later "Jane" get different tokens (`<PERSON_1>`, `<PERSON_2>`) unless you link mentions of the same person; the model may then treat them as two people.
- Tokens from one conversation must never be resolved with another conversation's mapping; scope the mapping per session or request.
- Indirect identifiers (job title plus town plus date) can identify a person with no single PII span; redaction does not make data anonymous.
- PII also arrives through tool results, retrieved documents, file names and images (screenshots, scanned IDs); redact every input path, not just the user message ([[document-parsing]]).
- Error messages and exception traces often include the raw prompt; scrub them before they reach error trackers.
- Streaming restore must buffer partial tokens across chunk boundaries (shown above), or users see `<EMA` and `IL_1>`.

## Related
- [[ai-security-privacy-compliance]] - the wider privacy, retention and compliance controls this step sits inside.
- [[guardrails-and-safety]] - PII checks as input and output guardrails.
- [[llm-observability]] - keeping PII out of prompt and completion logs.
- [[opentelemetry-genai-tracing]] - which span attributes carry message content.
- [[ai-regulation-checklist]] - legal obligations that decide how much redaction is required.
- [[agent-memory]] - personal data stored in long-term memory needs the same treatment.
- [[nlp-classic-tasks]] - training and evaluating NER models for custom entity types.
- [[permission-aware-rag-and-multi-tenancy]] - access control on retrieved documents, the complement to redaction.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/security/privacy-and-pii-handling.md - PII classification, retention and deletion in general systems.

## References
- Presidio repository (analyzer, anonymizer, image redactor; detection disclaimer): https://github.com/data-privacy-stack/presidio
- Presidio getting started, text de-identification: https://raw.githubusercontent.com/microsoft/presidio/main/docs/getting_started/getting_started_text.md
- Presidio anonymizer operators and deanonymizer: https://raw.githubusercontent.com/microsoft/presidio/main/docs/anonymizer/index.md
- presidio-analyzer on PyPI: https://pypi.org/project/presidio-analyzer/
- GDPR, Regulation (EU) 2016/679, Article 4 (definitions of personal data and pseudonymisation): https://eur-lex.europa.eu/eli/reg/2016/679/oj
- NIST SP 800-122, Guide to Protecting the Confidentiality of PII: https://csrc.nist.gov/pubs/sp/800/122/final
