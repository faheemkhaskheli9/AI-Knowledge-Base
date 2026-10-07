---
title: Education and tutoring AI
category: scenarios
tags: [education, tutoring, grading, ferpa, children, assessment, accessibility]
use_cases:
  - "build an AI tutor that guides rather than gives answers"
  - "auto-grade or give feedback on student writing and code"
  - "generate quizzes and study material from course content"
  - "answer student questions over course documents"
  - "deploy AI for minors while respecting privacy law"
status: stable
last_verified: 2026-10-03
sources:
  - https://studentprivacy.ed.gov/ferpa
  - https://www.ftc.gov/legal-library/browse/rules/childrens-online-privacy-protection-rule-coppa
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
---

# Education and tutoring AI

## Summary
Education AI spans tutoring, feedback, content generation and course Q&A.
Value comes from personalisation and instructor time saved; risk comes from
student privacy, children's data rules, unreliable grading and academic
integrity. Design for learning outcomes, keep teachers in control.

## Key concepts
- Pedagogy first: a tutor that hands over answers harms learning. Prompt for
  hints, questions and worked-example fading ([[prompt-engineering]]).
- Student records: in the US, FERPA governs education records; COPPA covers
  online services directed to children under 13; the EU AI Act treats AI used
  to evaluate learning outcomes or determine access to education as
  high-risk. Check these and local rules before deploying.
- Grounding in the course: answers should come from the instructor's
  material ([[rag-basics]]), citing the week/slide.
- Grading is high-stakes; LLM scores are noisy and can be biased against
  non-native writing. Use as a first pass or rubric-based feedback, with
  human final marks.

## When to use / scenarios
1. **Course Q&A assistant.** Problem: repeat questions to TAs. Approach: RAG
   over syllabus, slides and readings with citations, refuse out-of-scope.
   Read: [[rag-basics]], [[advanced-rag]], [[document-parsing]].
2. **Socratic tutor.** Approach: system prompt for hints, step checking, a
   solver tool for verifying math. Read: [[prompt-engineering]],
   [[reasoning-models]], [[tool-calling]].
3. **Writing feedback.** Approach: rubric-based comments, never final grade
   alone. Read: [[structured-output]], [[llm-evaluation]].
4. **Programming assignments.** Approach: run tests + LLM hints. Read:
   [[coding-agents]], [[software-engineering]].
5. **Quiz / flashcard generation.** Approach: generate from source chunks,
   attach citations, instructor review. Read: [[structured-output]].
6. **Adaptive learning paths.** Approach: knowledge tracing and recommender
   ideas. Read: [[recommender-systems]], [[classic-ml-scikit-learn]].
7. **Language learning with speech.** Approach: STT for pronunciation
   feedback, TTS for dialogue. Read: [[speech-to-text]], [[text-to-speech]],
   [[voice-agents]].
8. **Lecture transcription and accessibility.** Approach: STT, summaries,
   captions. Read: [[speech-to-text]], [[speaker-diarization]].
9. **Early-warning dropout prediction.** Approach: tabular model with fairness
   audit and humane intervention. Read: [[gradient-boosting-tabular]].
10. **Plagiarism / AI-text detection.** Be cautious: detectors have
    unreliable false positives; do not use as sole evidence.
    Read: [[hallucination-and-grounding]] for related limits.
11. **Admin automation (emails, scheduling).** Read: [[personal-assistants]].

## Setup & code
Reference architecture (flagship: course tutor):

```
Instructor uploads materials -> parse + chunk by lecture/section -> index
Student question -> safety filter (age-appropriate, off-topic) -> retrieve
 -> tutor prompt: hint level N, cite source, never reveal graded answers
 -> (math/code) verify via tool -> reply -> log (pseudonymised)
 -> instructor dashboard: common confusions, flagged chats
```

```python
TUTOR = """You are a tutor for {course}. Use only the provided notes.
Do not give the final answer to graded problems. Offer a hint first, then ask
the student to try. If the notes do not cover it, say so.
Hint level: {level} (1=nudge, 3=worked step)."""
# retrieve notes -> fill template -> call model; raise level only if the student
# asks again after attempting.
```

## Choosing / trade-offs
- Hosted vs local: student data is sensitive; many institutions need a
  vendor agreement or on-prem ([[ollama]], [[small-language-models]]).
- Strong vs cheap model: use a strong model for tutoring quality, small for
  quiz formatting ([[model-selection]], [[cost-and-latency]]).
- Open access vs logged-in: logging helps teachers but needs consent notices.

## Gotchas
- Students prompt-inject the tutor to get answers ([[prompt-injection]]).
- Wrong explanations delivered confidently; verify maths and code with tools.
- Bias against dialects and non-native writers in automated scoring.
- Minors: parental consent, age gating and data minimisation requirements.
- Over-reliance: measure learning (pre/post tests), not engagement.
- Do not retain chats longer than necessary; see
  [[ai-security-privacy-compliance]].

## Related
- [[guardrails-and-safety]] - age-appropriate behaviour, refusal policies.
- [[llm-evaluation]] - evaluating tutor quality with teacher rubrics.
- [[agent-memory]] - tracking learner progress.

## References
- FERPA (US Dept of Education): https://studentprivacy.ed.gov/ferpa
- COPPA rule (FTC): https://www.ftc.gov/legal-library/browse/rules/childrens-online-privacy-protection-rule-coppa
- Regulation (EU) 2024/1689, Annex III (education): https://eur-lex.europa.eu/eli/reg/2024/1689/oj
