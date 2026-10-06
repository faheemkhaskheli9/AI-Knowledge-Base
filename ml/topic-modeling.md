---
title: Topic modeling (LDA, NMF, BERTopic)
category: ml
tags: [topic-modeling, lda, latent-dirichlet-allocation, nmf, bertopic, text-mining, unsupervised, bag-of-words, tf-idf, scikit-learn]
use_cases:
  - "find the recurring themes in thousands of support tickets, reviews or survey answers without labels"
  - "tag each document with a mix of topics for search, routing or dashboards"
  - "track how discussion themes change over time in news, complaints or social posts"
  - "get a first map of an unknown text collection before building a classifier"
status: draft
last_verified: 2026-10-04
sources:
  - https://scikit-learn.org/stable/modules/decomposition.html#latentdirichletallocation
  - https://scikit-learn.org/stable/modules/decomposition.html#nmf
  - https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
  - https://maartengr.github.io/BERTopic/
  - https://www.jmlr.org/papers/v3/blei03a.html
---

# Topic modeling (LDA, NMF, BERTopic)

## Summary
Topic modeling discovers themes in a collection of documents without labels. Each topic is a weighted list of words ("refund, invoice, charge, card"), and each document gets a mix of topics. Classic methods work on word counts: LDA is a probabilistic model where documents are mixtures of topics and topics are distributions over words; NMF factorises the TF-IDF matrix into non-negative document-topic and topic-word matrices. Modern pipelines such as BERTopic embed documents with a sentence-embedding model, cluster the embeddings, and describe each cluster with its distinctive words. Use it for exploration, dashboards and as a cheap first pass before supervised classification.

## Key concepts
- **Bag of words.** Classic topic models ignore word order; the input is a document-term matrix (`CountVectorizer` for LDA, `TfidfVectorizer` for NMF). Preprocessing decides quality: stop words, `max_df` to drop words in most documents, `min_df` to drop rare ones, optional bigrams.
- **LDA (Latent Dirichlet Allocation).** Generative model: for each document draw a topic mixture, for each word draw a topic then a word. Priors `doc_topic_prior` (alpha) and `topic_word_prior` (eta) control how many topics per document and words per topic. Output: topic-word weights (`components_`) and document-topic mixtures (`transform`, rows sum to 1).
- **NMF.** X ≈ W·H with all entries ≥ 0; W is document-topic, H is topic-word. Deterministic given an init, fast, often crisper topics on short texts.
- **Embedding-based (BERTopic and similar).** Embed documents ([[embeddings]]), reduce dimensions (UMAP), cluster (HDBSCAN), then label clusters with class-based TF-IDF. Captures meaning beyond shared words, handles synonyms, gives one main topic per document plus an outlier topic (-1).
- **Number of topics (k).** A choice, not a fact. Pick by coherence scores, by inspecting top words, or by downstream usefulness; HDBSCAN-based methods choose it from cluster density instead.
- **Coherence.** Measures how often a topic's top words co-occur (e.g. C_v, NPMI via gensim). Better than perplexity, which tends to favour models humans find less interpretable.
- **Topics are not labels.** A topic is a word list; a human (or an LLM) names it. Topics can merge two real themes or split one.

## When to use / scenarios
- Customer support: cluster ticket text to find the top complaint drivers and route new tickets ([[customer-support]]).
- Product reviews and NPS free text: themes per product and per rating band ([[ecommerce-retail]], [[data-analytics]]).
- News, policy or research corpora: map a collection and track topic share over time.
- Bootstrap a labelled dataset: discover classes, then label a sample and train a classifier ([[nlp-classic-tasks]]).
- NOT the right tool when categories are already known: train a classifier (or zero-shot with an LLM) instead.
- NOT for a few dozen documents; read them, or ask an LLM to summarise themes directly.
- For short, noisy texts (tweets, chat lines), word-count LDA struggles; prefer NMF or embedding-based clustering ([[clustering]]).

## Setup & code
```bash
pip install scikit-learn numpy
```
A toy corpus of support tickets drawn from three hidden themes; LDA on counts and NMF on TF-IDF both recover them:
```python
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Toy corpus: support tickets drawn from three hidden themes.
rng = np.random.default_rng(0)
themes = {
    "billing": "invoice refund charge payment card billing overcharged receipt",
    "delivery": "delivery courier late parcel tracking shipped address package",
    "login": "password login account locked reset email verification app",
}
filler = "please help thanks hello issue today again customer".split()
docs = []
for _ in range(300):
    words = rng.choice(themes[rng.choice(list(themes))].split(), 6).tolist()
    docs.append(" ".join(words + rng.choice(filler, 3).tolist()))


def show(model, vocab, k=5):
    for i, comp in enumerate(model.components_):
        print(f"  topic {i}: {' '.join(vocab[comp.argsort()[::-1][:k]])}")


# LDA wants raw counts; drop words in >50% of docs (the filler) and very rare ones.
cv = CountVectorizer(stop_words="english", max_df=0.5, min_df=2)
counts = cv.fit_transform(docs)
lda = LatentDirichletAllocation(n_components=3, random_state=0).fit(counts)
print("LDA:")
show(lda, cv.get_feature_names_out())
doc_topics = lda.transform(counts)          # rows sum to 1: topic mixture per doc
print("doc 0 mixture:", doc_topics[0].round(2), "|", docs[0])

# NMF works on TF-IDF and is often crisper on short texts.
tf = TfidfVectorizer(stop_words="english", max_df=0.5, min_df=2)
nmf = NMF(n_components=3, init="nndsvda", random_state=0).fit(tf.fit_transform(docs))
print("NMF:")
show(nmf, tf.get_feature_names_out())
```
Output with scikit-learn 1.9.0:
```text
LDA:
  topic 0: app account verification reset login
  topic 1: parcel package address shipped late
  topic 2: card receipt overcharged charge invoice
doc 0 mixture: [0.91 0.04 0.04] | email reset account account password password please help again
NMF:
  topic 0: receipt card overcharged charge invoice
  topic 1: account app login verification reset
  topic 2: address parcel package shipped late
```
Embedding-based alternative (downloads a sentence-transformers model; needs a few thousand real documents to cluster well):
```bash
pip install bertopic
```
```python
from bertopic import BERTopic

topic_model = BERTopic()                       # embed -> UMAP -> HDBSCAN -> c-TF-IDF
topics, probs = topic_model.fit_transform(docs)  # docs: list[str], ideally 1000+
print(topic_model.get_topic_info().head())      # topic id, size, top words; -1 = outliers
```

## Choosing / trade-offs
- **NMF** as the fast, deterministic baseline, especially on short texts and when you will fix k by inspection.
- **LDA** when you want true per-document topic mixtures and a probabilistic model (priors, held-out likelihood); slower, more sensitive to preprocessing and seeds.
- **BERTopic / embedding clustering** when meaning matters more than shared vocabulary, texts are short or multilingual, and a GPU or an embedding API is available. Costs an embedding pass and has more knobs (UMAP and HDBSCAN parameters).
- **LLM-based** labelling or summarising of clusters: use an LLM to name topics from their top words and sample documents; cheap and readable, but keep the clustering itself deterministic for repeatability.
- **k**: try a range (e.g. 5-50), compare coherence and inspect; too few merges themes, too many splits them into near-duplicates.

## Gotchas
- Garbage preprocessing gives garbage topics: boilerplate (signatures, "please help", templates) dominates unless removed by `max_df` or custom stop words.
- LDA results change with the random seed; fix it and check stability across seeds before reporting themes.
- Feeding TF-IDF to LDA is a common mistake; LDA expects counts.
- Topic numbering is arbitrary and changes between runs; never hard-code "topic 3 = billing" across retrains.
- BERTopic assigns many documents to the outlier topic -1 by default; use its outlier-reduction options or lower HDBSCAN's `min_cluster_size` rather than ignoring them.
- Topic share over time is confounded by volume and by new vocabulary; fit once on a reference period or use a dynamic topic model.
- PII in free text ends up in topic word lists and dashboards; redact names, emails and IDs first.

## Related
- [[nlp-classic-tasks]] - TF-IDF features and text classification once classes are known.
- [[embeddings]] - document embeddings behind embedding-based topic models.
- [[clustering]] - k-means, HDBSCAN and how to judge clusters.
- [[dimensionality-reduction]] - NMF, SVD and UMAP.
- [[word-embeddings-word2vec]] - word vectors and the bag-of-words baseline.
- [[customer-support]] - ticket theme discovery in practice.
- [[nmf-from-scratch]] - NMF built by hand with multiplicative updates for parts-based topics.

## References
- scikit-learn, LatentDirichletAllocation: https://scikit-learn.org/stable/modules/decomposition.html#latentdirichletallocation
- scikit-learn, NMF: https://scikit-learn.org/stable/modules/decomposition.html#nmf
- scikit-learn example, Topic extraction with NMF and LDA: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
- BERTopic documentation: https://maartengr.github.io/BERTopic/
- Blei, Ng & Jordan, Latent Dirichlet Allocation (JMLR 2003): https://www.jmlr.org/papers/v3/blei03a.html
