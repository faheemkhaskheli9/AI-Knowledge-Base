---
title: Media, entertainment and gaming AI
category: scenarios
tags: [media, gaming, npc, content-generation, moderation, recommendation, provenance, c2pa]
use_cases:
  - "generate game assets, dialogue or marketing art with AI"
  - "build LLM-driven NPCs or companions in a game"
  - "moderate user-generated text, images, voice chat or video"
  - "recommend videos, articles, music or games to users"
  - "dub, subtitle or localise video and game content"
status: draft
last_verified: 2026-10-15
sources:
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
  - https://eur-lex.europa.eu/eli/reg/2022/2065/oj
  - https://c2pa.org/specifications/specifications/2.2/specs/C2PA_Specification.html
  - https://www.copyright.gov/ai/
---

# Media, entertainment and gaming AI

## Summary
Media and games use AI in four main ways: generating content (images,
audio, text, video, 3D) during production, powering interactive characters
at runtime, moderating user-generated content, and recommending what to
watch or play next. The hard parts are less the models than rights
(training data, voices, likeness, copyright of outputs), safety of
live-generated content, latency and cost per player, and transparency rules
for synthetic media and recommender systems.

## Key concepts
- Pre-generated vs live-generated content: content made by AI during
  production can be reviewed before release; content generated at runtime
  (NPC dialogue, user prompts) cannot, so it needs guardrails, filters and
  reporting. Storefronts such as Steam ask developers to disclose both
  kinds.
- Rights and consent: voice clones and likenesses need contracts and
  consent from the performer; training data licences matter
  ([[model-licenses]]). The US Copyright Office's reports say material
  generated purely by AI without sufficient human authorship is not
  protected; check the current guidance and your jurisdiction.
- Transparency: the EU AI Act (Article 50) requires providers to mark
  synthetic audio, image, video and text in a machine-readable way and
  deployers to disclose deepfakes, with lighter treatment for evidently
  artistic or fictional works; check the current text and guidelines.
  C2PA "Content Credentials" is the main provenance standard.
- Platforms: the EU Digital Services Act requires notice-and-action,
  statements of reasons for moderation decisions and recommender
  transparency; very large platforms must offer a feed option not based on
  profiling. Check which tier applies to you.
- Runtime budget: per-player inference cost and latency decide whether an
  NPC uses a frontier API, a small local model, or scripted dialogue with AI
  variations ([[cost-and-latency]], [[small-language-models]]).
- Minors: many games and platforms have child users; age-appropriate
  design and child-privacy laws apply.

## When to use / scenarios
1. **Concept art, textures and marketing visuals.** Approach: image
   generation with licensed or owned training data, artist in the loop,
   provenance metadata on outputs. Read: [[image-generation-models]],
   [[diffusion-models]], [[marketing-content]].
2. **NPC dialogue and companions.** Approach: LLM with character card,
   world-state tools and memory; content filter on input and output; hard
   limits on what the NPC can change in game state. Read: [[agents]],
   [[agent-memory]], [[tool-calling]], [[prompt-injection]],
   [[guardrails-and-safety]].
3. **Voice: dubbing, localisation, voiced NPCs.** Approach: STT ->
   translation -> TTS with consented voices; human QA for lip-sync and
   tone. Read: [[speech-to-text]], [[text-to-speech]], [[voice-agents]].
4. **Subtitles, transcripts and archive search.** Approach: ASR with
   diarization, embeddings over transcripts and frames. Read:
   [[speech-to-text]], [[speaker-diarization]], [[embeddings]],
   [[video-analytics]].
5. **Content moderation (chat, voice, images, UGC levels).** Approach:
   fast classifiers for clear cases, LLM or human review for the grey band,
   appeals and statements of reasons. Read: [[nlp-classic-tasks]],
   [[guardrails-and-safety]], [[image-classification]], [[llm-evaluation]].
6. **Recommendation (videos, articles, music, games, store).** Approach:
   candidate retrieval + ranking, exploration via bandits, diversity
   constraints, non-personalised fallback. Read: [[recommender-systems]],
   [[learning-to-rank]], [[multi-armed-bandits]],
   [[mmr-diversity-reranking-from-scratch]].
7. **Player analytics: churn, matchmaking, cheat and bot detection.**
   Approach: tabular models and anomaly detection on telemetry. Read:
   [[gradient-boosting-tabular]], [[anomaly-detection]],
   [[survival-analysis]], [[statistics-and-ab-testing]].
8. **Game-playing and testing agents.** Approach: RL or scripted bots for
   QA and balancing; LLM agents for exploratory testing. Read:
   [[reinforcement-learning]], [[computer-use-agents]].
9. **Newsroom tools (summaries, headlines, tagging, translation).**
   Approach: drafts reviewed by editors, citations to the source
   article. Read: [[hallucination-and-grounding]], [[topic-modeling]].
10. **Procedural levels and quests.** Approach: generator proposes,
    validators (solvability, balance) check; often classic PCG plus LLM
    flavour text. Read: [[structured-output]],
    [[black-box-and-evolutionary-optimization]].

## Setup & code
Reference architecture (flagship: LLM-driven NPC):

```
player input (text/voice) -> input filter (toxicity, injection, age rules)
  -> NPC prompt: character card + retrieved lore + short-term memory
  -> LLM with tools: get_quest_state(), give_item() (validated server-side)
  -> output filter + length cap -> TTS (consented voice) -> client
  -> fallback: scripted line on timeout or filter block
  -> logs for report review; per-player cost and latency budget
```

Runnable sketch: item-item collaborative filtering for "players who played
X also played" recommendations (synthetic interaction matrix).

```python
import numpy as np

rng = np.random.default_rng(3)
n_users, n_items = 500, 40
genre = rng.integers(0, 4, n_items)            # hidden item genre
taste = rng.integers(0, 4, n_users)            # hidden user preference
p = np.where(genre[None, :] == taste[:, None], 0.30, 0.03)
R = (rng.random((n_users, n_items)) < p).astype(float)  # played or not

norm = np.linalg.norm(R, axis=0) + 1e-9
S = (R.T @ R) / np.outer(norm, norm)            # cosine item similarity
np.fill_diagonal(S, 0)

def recommend(user, k=5):
    scores = R[user] @ S
    scores[R[user] > 0] = -np.inf               # do not re-recommend
    return np.argsort(scores)[::-1][:k]

u = 0
recs = recommend(u)
print("user taste", taste[u], "-> recommended genres", genre[recs])
```

## Choosing / trade-offs
- Hosted frontier model vs on-device small model for NPCs: hosted gives
  better dialogue but adds per-player cost, latency and an online
  dependency; local models are cheaper at scale and work offline but are
  weaker and use the player's hardware ([[edge-on-device]], [[ollama]]).
- Fully generated vs AI-assisted content: assisted (artist edits, writer
  curates) keeps quality, style consistency and authorship clearer.
- Automated vs human moderation: automation scales and is fast; humans
  handle context, appeals and edge cases. Most systems use both.
- Engagement vs wellbeing in recommendations: optimising watch time alone
  can amplify harmful or low-quality content; add quality and diversity
  objectives and measure long-term retention.

## Gotchas
- Players will jailbreak NPCs; anything the model can do through tools,
  a player can make it do. Validate tool calls on the server.
- Live-generated content can produce illegal or off-brand output; filter,
  log and provide in-game reporting.
- Voice cloning without clear consent and contract terms creates legal and
  reputational risk.
- Style drift across generated assets breaks visual consistency; use
  reference images, fine-tuned styles and art direction review.
- Cold start in recommendations: new items and users need content-based
  or popularity fallbacks.
- Moderation classifiers have uneven error rates across dialects and
  languages; evaluate per language and community.
- Watermarks and metadata are stripped by re-encoding and screenshots;
  provenance helps but does not prove absence of AI.

## Related
- [[marketing-content]] - campaign copy and creative production.
- [[customer-support]] - player support and community management.
- [[multi-agent-systems]] - simulations with many interacting agents.
- [[ai-security-privacy-compliance]] - player data, minors and retention.
- [[video-generation-models]] - text-to-video and image-to-video for production.
- [[voice-cloning-and-ethics]] - consent and watermarking for cloned voices.
- [[ai-regulation-checklist]] - Article 50 transparency and other AI Act duties.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/case-studies/design-video-streaming.md

## References
- Regulation (EU) 2024/1689 (AI Act), Article 50: https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- Regulation (EU) 2022/2065 (Digital Services Act): https://eur-lex.europa.eu/eli/reg/2022/2065/oj
- C2PA Technical Specification (Content Credentials): https://c2pa.org/specifications/specifications/2.2/specs/C2PA_Specification.html
- US Copyright Office, Copyright and Artificial Intelligence: https://www.copyright.gov/ai/
