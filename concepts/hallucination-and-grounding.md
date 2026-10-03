---
title: Hallucination and grounding
category: concepts
tags: [hallucination, grounding, citations, rag, verification, factuality]
use_cases:
  - "stop a support chatbot from inventing refund policies"
  - "make a legal or medical research assistant cite verifiable sources"
  - "add a verification step that flags unsupported claims in generated summaries"
  - "decide when retrieval, tools or fine-tuning is the right fix for wrong answers"
  - "measure hallucination rate of an LLM feature before launch in finance or healthcare"
status: draft
last_verified: 2026-10-03
sources:
  - https://arxiv.org/abs/2311.05232
  - https://arxiv.org/abs/2005.11401
  - https://docs.claude.com/en/docs/test-and-evaluate/strengthen-guardrails/reduce-hallucinations
  - https://docs.claude.com/en/docs/build-with-claude/citations
---

# Hallucination and grounding

## Summary
Hallucination is when an LLM produces fluent but false or unsupported content: invented facts, citations, APIs or numbers. It is a consequence of training to predict plausible text, not to know what it does not know. Grounding reduces it by tying answers to supplied evidence (retrieval, tools, documents) and verifying them.

## Key concepts
- **Types**: factual error (wrong claim), fabrication (invented source or entity), unfaithfulness (contradicts the provided context), reasoning error.
- **Causes**: gaps or noise in training data, pressure to answer, outdated knowledge, ambiguous prompts, and long or distracting context.
- **Grounding**: supply authoritative context ([[rag-basics]]), call tools/databases for facts ([[tool-calling]]), and instruct the model to answer only from that context.
- **Attribution/citations**: have the model point to the exact passage; some APIs return structured citations tied to source documents.
- **Verification**: a second pass (or deterministic check) tests each claim against sources; also self-consistency (disagreeing samples signal uncertainty).
- **Abstention**: explicitly allow "I don't know"; reward it in prompts and evals.
- **Faithfulness vs correctness**: a grounded answer can still be wrong if the retrieved source is wrong or irrelevant.

## When to use / scenarios
- Any user-facing factual assistant, especially regulated domains: healthcare guidance, legal research, finance, HR/policy bots.
- Real-world: an insurer's FAQ bot answers only from policy documents and cites clause numbers; an analyst tool quotes the filing sentence behind each figure.
- Heavier controls when errors are costly (money, health, law); light controls suffice for brainstorming.
- Not a fix: raising model size alone, or lowering temperature alone ([[decoding-and-sampling]]).

## Setup & code
Quote-first grounded answering with an abstention path (`pip install anthropic`; choose the model ID from the current models page):

```python
import anthropic

client = anthropic.Anthropic()
SYSTEM = (
    "Answer ONLY from the documents. First extract exact supporting quotes inside "
    "<quotes>. Then answer inside <answer>, citing quotes. If the documents do not "
    "contain the answer, reply exactly: INSUFFICIENT_CONTEXT."
)
docs = "<doc id='policy-3'>Refunds are available within 30 days of purchase.</doc>"
resp = client.messages.create(
    model="<current Claude model>", max_tokens=400, system=SYSTEM,
    messages=[{"role": "user", "content": f"{docs}\n\nQuestion: Can I get a refund after 60 days?"}],
)
text = resp.content[0].text
print(text)
# deterministic check: every quote must appear verbatim in the source docs
import re
for q in re.findall(r"<quotes>(.*?)</quotes>", text, re.S):
    for line in filter(None, (l.strip(" -\"'") for l in q.splitlines())):
        assert line in docs, f"unsupported quote: {line}"
```
For built-in source attribution see the Anthropic citations feature in the docs; for RAG pipelines see [[advanced-rag]].

## Choosing / trade-offs
- RAG (fresh, auditable, retrieval can fail) vs fine-tuning (style/format, poor for facts) vs tools (exact, needs integration).
- Strict "context only" prompting reduces fabrication but raises refusals; tune on your eval set.
- Verifier pass (LLM judge or NLI) adds cost and latency; apply it to high-risk answers only.
- Human review for high-stakes outputs remains the final control.

## Gotchas
- Fabricated citations look real; always check that quoted text and URLs exist.
- Retrieval returning irrelevant chunks makes models confidently answer from the wrong text.
- Models state numbers they computed "in their head" incorrectly; use code tools for arithmetic.
- Self-reported confidence is poorly calibrated.
- Asking a leading question ("why is X true?") invites agreement; phrase neutrally.
- A zero-hallucination guarantee does not exist; measure rate on your own data with [[llm-evaluation]] and monitor in production ([[llm-observability]]).

## Related
- [[rag-basics]] - primary grounding technique.
- [[advanced-rag]] - reranking, citations, verification patterns.
- [[tool-calling]] - exact lookups and calculations.
- [[llm-evaluation]] - measuring faithfulness.
- [[guardrails-and-safety]] - runtime checks and refusal policies.
- [[prompt-engineering]] - abstention and quote-first prompts.

## References
- Huang et al., Survey on hallucination in LLMs: https://arxiv.org/abs/2311.05232
- Lewis et al., RAG: https://arxiv.org/abs/2005.11401
- Claude, reduce hallucinations: https://docs.claude.com/en/docs/test-and-evaluate/strengthen-guardrails/reduce-hallucinations
- Claude citations: https://docs.claude.com/en/docs/build-with-claude/citations
