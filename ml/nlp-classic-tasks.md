---
title: Classic NLP tasks (classification, NER, sentiment)
category: ml
tags: [nlp, text-classification, ner, sentiment, transformers, setfit, spacy, zero-shot]
use_cases:
  - "classify support tickets or emails into categories"
  - "analyse sentiment of product reviews or tweets at scale"
  - "extract names, organisations, dates and amounts from documents (NER)"
  - "decide whether to use a small fine-tuned model or an LLM for text labelling"
  - "build a cheap, fast text classifier with only a few hundred labels"
status: draft
last_verified: 2026-10-03
sources:
  - https://huggingface.co/docs/transformers/main/en/tasks/sequence_classification
  - https://huggingface.co/docs/transformers/main/en/tasks/token_classification
  - https://huggingface.co/docs/setfit/index
  - https://spacy.io/usage
---

# Classic NLP tasks (classification, NER, sentiment)

## Summary
Text classification, sentiment analysis and named-entity recognition (NER) are well-defined supervised tasks. Options range from TF-IDF + linear model, to fine-tuned encoder transformers (BERT-family), to prompting an LLM. The right choice depends on label count, volume, latency/cost and language; small fine-tuned encoders are often cheaper and more consistent at scale, LLMs win when there is no labelled data.

## Key concepts
- Sequence classification: one label per text (topic, intent, sentiment). Token classification: one label per token (NER, with BIO tags).
- Encoder models (BERT, RoBERTa, DeBERTa, multilingual XLM-R) fine-tuned end-to-end; typically 100M-400M parameters, run on CPU or small GPU.
- Embeddings + classifier: freeze a sentence-embedding model (see [[embedding-models]]) and train logistic regression on top; very strong with little data.
- SetFit: few-shot fine-tuning of sentence transformers with contrastive learning; good with 8-64 examples per class.
- Zero-shot / LLM labelling: prompt an LLM with label definitions and a JSON schema (see [[structured-output]]); no training data needed.
- Metrics: per-class precision/recall/F1, macro-F1 for imbalance, entity-level F1 (seqeval) for NER.
- Distillation pattern: label with an LLM, human-check a sample, train a small model on those labels (see [[data-labeling-and-synthetic-data]]).

## When to use / scenarios
- Customer support: ticket routing, intent detection, urgency; high volume favours a small model.
- Social/brand monitoring and reviews: sentiment, aspect-based sentiment.
- Legal/finance/healthcare documents: entity extraction (parties, amounts, drugs); regulated data may need local models.
- Content moderation, spam, language ID.
- Use an LLM instead when: labels change often, few or no examples, long/complex context, or need reasoning plus extraction in one call.
- Use rules/regex for rigidly formatted entities (IBANs, emails, dates).

## Setup & code
```bash
pip install transformers torch scikit-learn
```
```python
from transformers import pipeline

# Zero-shot-ready sentiment pipeline (downloads a default model; pin one for production)
clf = pipeline("text-classification",
               model="distilbert-base-uncased-finetuned-sst-2-english")
print(clf(["Great battery life", "Arrived broken and support ignored me"]))

# Token classification (NER)
ner = pipeline("token-classification", model="dslim/bert-base-NER",
               aggregation_strategy="simple")
print(ner("Maria Lopez joined Acme Corp in Berlin."))
```
Fine-tuning: see the Hugging Face sequence/token classification task guides (use `Trainer`, `AutoModelForSequenceClassification`). spaCy (`pip install spacy`, then download a model) is a good production NER/pipeline toolkit.

## Choosing / trade-offs
- No labels, low volume, changing taxonomy: LLM with structured output.
- 50+ labels per class, high volume or latency/cost limits: fine-tuned encoder (or embeddings + logistic regression).
- Few labels (under ~100 total): SetFit or LLM-labelled bootstrap.
- Non-English / low-resource languages: multilingual encoders (XLM-R) or a strong multilingual LLM; evaluate on your own data.
- Privacy/offline: local encoder models avoid sending text to an API.
- Cost: an encoder serves millions of texts cheaply on CPU; LLM per-call cost scales linearly.

## Gotchas
- Default pipeline models are tuned for one domain (e.g. movie reviews) and degrade on yours; evaluate before trusting.
- Label noise caps accuracy; check inter-annotator agreement before blaming the model.
- NER subword tokens: align labels to word pieces and aggregate entities properly.
- Truncation at 512 tokens silently drops text; chunk long documents.
- Class imbalance: report macro-F1, not accuracy.
- Sarcasm, negation and domain slang break sentiment; keep a hard-example test set.
- LLM labellers drift across versions and prompts; pin the model and keep a golden set.

## Related
- [[embedding-models]] - embeddings as features.
- [[structured-output]] - LLM labelling with schemas.
- [[fine-tuning-and-peft]] - adapting larger models.
- [[huggingface-transformers]] - library setup.
- [[data-labeling-and-synthetic-data]] - creating training labels.
- [[llm-evaluation]] - evaluating classifiers and labellers.

## References
- https://huggingface.co/docs/transformers/main/en/tasks/sequence_classification
- https://huggingface.co/docs/transformers/main/en/tasks/token_classification
- https://huggingface.co/docs/setfit/index
- https://spacy.io/usage
