---
title: Marketing and content generation
category: scenarios
tags: [marketing, copywriting, seo, image-generation, brand, content]
use_cases:
  - "generate on-brand marketing copy at scale"
  - "produce product or campaign images with generative models"
  - "personalise emails and ads"
  - "summarise customer feedback and social sentiment"
  - "repurpose long content into multiple formats"
status: draft
last_verified: 2026-10-03
sources:
  - https://www.ftc.gov/business-guidance/advertising-marketing
  - https://artificialintelligenceact.eu/article/50/
  - https://c2pa.org/
---

# Marketing and content generation

## Summary
Marketing is the friendliest domain for generative AI: drafts are cheap to
review, variation is valuable and output is text and images. The risks are
brand voice drift, unsubstantiated claims, IP and licence issues, and
disclosure rules, so keep a human approval gate and ground claims in
approved facts.

## Key concepts
- Brand voice via examples: a style guide plus few-shot samples beats
  adjectives ([[prompt-engineering]]).
- Claims must trace to a source: product facts come from a spec database or
  approved messaging, not model memory ([[hallucination-and-grounding]]).
- Disclosure: advertising law (e.g. FTC guidance in the US) applies to AI
  content, including fake reviews and endorsements; the EU AI Act Article 50
  adds labelling duties for certain synthetic media and deepfakes. Check
  current rules for your market.
- IP: licences of image models and training-data provenance vary
  ([[model-licenses]], [[image-generation-models]]). Copyright protection of
  purely AI-generated output is uncertain and differs by country.
- Measurement: A/B test; do not judge copy by reading it.

## When to use / scenarios
1. **Ad, email and landing-page copy variants.** Problem: slow creative
   cycles. Approach: brief + brand guide + approved claims -> N variants ->
   human pick -> A/B test. Read: [[prompt-engineering]],
   [[structured-output]], [[model-selection]].
2. **SEO content and product descriptions at scale.** Approach: structured
   product data -> templated generation, spot-check; avoid thin
   mass-produced pages. Read: [[ecommerce-retail]], [[structured-output]].
3. **Campaign imagery and variations.** Approach: diffusion model with brand
   references, human art direction, provenance metadata. Read:
   [[image-generation-models]], [[diffusion-models]], [[model-licenses]].
4. **Repurposing (webinar -> blog -> posts).** Approach: transcribe,
   summarise, reformat per channel. Read: [[speech-to-text]],
   [[long-context]].
5. **Personalised outreach.** Approach: merge CRM fields into templates; LLM
   only for light tailoring; respect consent. Read: [[agents]],
   [[ai-security-privacy-compliance]].
6. **Social listening and review analysis.** Approach: classify sentiment and
   topics, cluster themes. Read: [[nlp-classic-tasks]], [[embeddings]],
   [[data-analytics]].
7. **Translation and localisation.** Approach: LLM draft + native review.
   Read: [[model-selection]], [[qwen]], [[anthropic-claude]].
8. **Audience segmentation and churn/propensity models.** Read:
   [[gradient-boosting-tabular]], [[recommender-systems]].
9. **Voiceover and video narration.** Read: [[text-to-speech]]; consent is
   required for cloning a real person's voice.
10. **Content QA and brand-compliance checks.** Approach: LLM-as-judge on a
    rubric, plus rule checks (banned claims). Read: [[llm-evaluation]],
    [[guardrails-and-safety]].

## Setup & code
Reference architecture (flagship: governed copy generation):

```
brief (audience, goal, channel) + brand guide + approved-claims DB
  -> generate N drafts (low temperature for facts, higher for hooks)
  -> automated checks: banned claims, length, reading level, claim->source
  -> human approval -> publish -> performance data -> winning copy
     added to few-shot pool
```

```python
import anthropic
c = anthropic.Anthropic()
brief = "Subject lines for spring sale, friendly, max 45 chars."
r = c.messages.create(
    model="claude-sonnet-5-5", max_tokens=300,   # verify model id in [[anthropic-claude]]
    system=BRAND_GUIDE + "\nUse only these facts:\n" + APPROVED_CLAIMS,
    messages=[{"role": "user", "content": brief + " Give 8 options as JSON list."}])
```

## Choosing / trade-offs
- Quality vs volume: a strong model for hero copy, small models for bulk
  variants ([[cost-and-latency]], [[small-language-models]]).
- Hosted image API vs local diffusion: API is simple and indemnity terms
  may exist; local gives control and no per-image fees
  ([[gpu-cloud-options]], [[huggingface-transformers]]).
- Fine-tune for voice only after prompting plus examples plateau
  ([[fine-tuning-and-peft]]).

## Gotchas
- Generic "AI-sounding" prose; edit, add real specifics and customer quotes.
- Unsupported superlatives and health or financial claims create regulatory
  exposure.
- Image models reproduce logos, styles or people; screen outputs and avoid
  real-person likeness without consent.
- Spam and consent law (CAN-SPAM, GDPR/ePrivacy) apply to AI-written outreach.
- Search engines may demote scaled low-value content; check their current
  guidance.
- Embedding customer data in prompts raises privacy duties.

## Related
- [[customer-support]] - shared tone and knowledge assets.
- [[personal-assistants]] - scheduling and posting automation.
- [[llm-observability]] - cost and quality tracking across campaigns.

## References
- FTC advertising and marketing guidance: https://www.ftc.gov/business-guidance/advertising-marketing
- EU AI Act Article 50: https://artificialintelligenceact.eu/article/50/
- C2PA content provenance: https://c2pa.org/
