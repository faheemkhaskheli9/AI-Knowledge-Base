---
title: Customer support with AI
category: scenarios
tags: [support, chatbot, rag, ticket-triage, voice, helpdesk]
use_cases:
  - "build a support chatbot that answers from our help center"
  - "auto-triage and route incoming support tickets"
  - "draft agent replies and summarise long ticket threads"
  - "build a voice agent for a call center"
  - "measure whether an AI support bot is actually resolving issues"
status: draft
last_verified: 2026-10-03
sources:
  - https://docs.anthropic.com/en/docs/about-claude/use-case-guides/customer-support-chat
  - https://artificialintelligenceact.eu/article/50/
---

# Customer support with AI

## Summary
Support is the most common production LLM use case: a large corpus of
documentation, repetitive questions, clear success metrics (resolution rate,
handle time) and an obvious fallback (a human). The pattern is RAG over
approved content, tools for account lookups, and escalation rules.

## Key concepts
- Answers must be grounded in approved content; invented policy (refunds,
  warranties) creates real liability. See [[hallucination-and-grounding]].
- Two modes: customer-facing automation (high risk, needs guardrails) and
  agent assist (human reviews every message; much lower risk, faster to ship).
- Account actions (refund, cancel) go through tools with permission checks and
  confirmation, never free text to a backend ([[tool-calling]]).
- Latency matters in chat and critically in voice (sub-second turn-taking).
- Disclosure: in the EU, AI Act Article 50 transparency duties apply to
  systems that interact with people; check the current text and your
  jurisdiction's rules.

## When to use / scenarios
1. **FAQ / help-center bot.** Problem: 60% of tickets are repeat questions.
   Approach: RAG with citations, "I don't know" fallback to a human.
   Read: [[rag-basics]], [[advanced-rag]], [[vector-databases]],
   [[guardrails-and-safety]].
2. **Ticket triage and routing.** Problem: manual tagging, slow SLAs.
   Approach: LLM classifier into a fixed label set (category, priority,
   language, sentiment); move to embeddings + classifier at volume.
   Read: [[structured-output]], [[nlp-classic-tasks]], [[embeddings]].
3. **Agent assist (draft replies).** Problem: slow, inconsistent replies.
   Approach: retrieve similar solved tickets + docs, draft, human edits.
   Read: [[advanced-rag]], [[prompt-engineering]].
4. **Thread summarisation and handoff.** Problem: escalations lose context.
   Approach: summarise to a fixed template (issue, steps tried, sentiment).
   Read: [[structured-output]], [[long-context]].
5. **Order/account lookups inside chat.** Problem: bot cannot act.
   Approach: agent with read tools first, write tools behind confirmation.
   Read: [[agents]], [[tool-calling]], [[model-context-protocol]].
6. **Voice support line.** Problem: call volume, wait times.
   Approach: STT + LLM + TTS or realtime model, barge-in handling, warm
   transfer. Read: [[voice-agents]], [[speech-to-text]], [[text-to-speech]].
7. **Call QA and analytics.** Problem: only 2% of calls reviewed. Approach:
   transcribe, diarize, score against a rubric. Read: [[speaker-diarization]],
   [[llm-evaluation]].
8. **Multilingual support.** Problem: languages the team does not cover.
   Approach: multilingual model for reply and translation, native review for
   high-stakes flows. Read: [[model-selection]], [[qwen]], [[anthropic-claude]].
9. **Knowledge-base gap detection.** Problem: docs rot. Approach: cluster
   unanswered questions to find missing articles. Read: [[embeddings]],
   [[data-analytics]].
10. **Screenshot-based troubleshooting.** Approach: VLM reads error
    screenshots. Read: [[vision-language-models]].

## Setup & code
Reference architecture (flagship: help-center bot with escalation):

```
User -> input guard (PII redact, injection screen)
     -> intent router (small model): faq | account | complaint | human
     -> faq: retrieve (hybrid + rerank) -> LLM answer w/ citations
     -> account: tool-calling agent (read tools; write = confirm)
     -> confidence/policy check -> answer or handoff to human w/ summary
     -> log (trace, retrieved ids, outcome) -> eval + feedback loop
```

```python
import anthropic
client = anthropic.Anthropic()

def answer(question, chunks):
    ctx = "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    r = client.messages.create(
        model="claude-sonnet-5-5",  # check current model ids in [[anthropic-claude]]
        max_tokens=500,
        system=("Answer only from the context. Cite ids like [id]. "
                "If the context does not answer it, reply exactly: ESCALATE."),
        messages=[{"role": "user", "content": f"{ctx}\n\nQ: {question}"}],
    )
    t = r.content[0].text
    return None if t.strip() == "ESCALATE" else t   # None -> human handoff
```

## Choosing / trade-offs
- Automation vs assist: assist first; automate the top intents once the eval
  shows high accuracy per intent.
- Big model vs small: route simple intents to a small model
  ([[small-language-models]]), keep a strong model for complaints and
  multi-step cases ([[cost-and-latency]]).
- Hosted vs local: ticket text holds PII; if data cannot leave, see
  [[ollama]] and [[inference-servers-vllm]].
- Fine-tuning for tone is optional; do it after RAG quality is fixed
  ([[fine-tuning-and-peft]]).

## Gotchas
- Bot promises refunds or terms that do not exist. Courts and tribunals have
  held companies to chatbot statements in some cases (Moffatt v. Air Canada,
  2024); treat bot output as company speech.
- Prompt injection via ticket text or pasted emails ([[prompt-injection]]).
- Deflection rate is a vanity metric; measure resolution and re-contact rate.
- No visible human escape route frustrates users and can breach consumer
  rules in some regions; check local requirements.
- Stale docs produce confidently wrong answers; version and re-index content.
- Storing transcripts: retention, consent and call-recording laws vary; see
  [[ai-security-privacy-compliance]].

## Related
- [[llm-observability]] - tracing conversations and retrieval failures.
- [[llm-evaluation]] - golden sets of real tickets.
- [[agent-memory]] - customer context across sessions.
- [[ecommerce-retail]] - order-centric support.

## References
- Anthropic customer support guide: https://docs.anthropic.com/en/docs/about-claude/use-case-guides/customer-support-chat
- EU AI Act Article 50 (transparency obligations): https://artificialintelligenceact.eu/article/50/
- Moffatt v. Air Canada, 2024 BCCRT 149 (verify citation before relying on it)
