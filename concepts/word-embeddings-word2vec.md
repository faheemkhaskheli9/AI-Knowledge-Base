---
title: Classic word embeddings (word2vec, GloVe, fastText)
category: concepts
tags: [word-embeddings, word2vec, glove, fasttext, skip-gram, cbow, negative-sampling, gensim, nlp, distributional-semantics]
use_cases:
  - "train word vectors on a domain corpus (medical notes, legal text, product titles)"
  - "find similar words or expand search queries with synonyms from your own data"
  - "get cheap fixed-size features for a small text classifier without a GPU"
  - "handle misspellings and unseen words with subword (fastText) vectors"
  - "understand where modern embeddings and the transformer embedding layer came from"
status: draft
last_verified: 2026-10-04
sources:
  - https://arxiv.org/abs/1301.3781
  - https://arxiv.org/abs/1310.4546
  - https://nlp.stanford.edu/projects/glove/
  - https://arxiv.org/abs/1607.04606
  - https://radimrehurek.com/gensim/models/word2vec.html
---

# Classic word embeddings (word2vec, GloVe, fastText)

## Summary
Static word embeddings map each word in a vocabulary to one dense vector (typically 100-300 dims) learned from co-occurrence: words used in similar contexts get nearby vectors ("you shall know a word by the company it keeps"). word2vec learns them by predicting context words, GloVe by factorising a global co-occurrence matrix, fastText by adding character n-grams so unseen and misspelled words still get a vector. They train in minutes on a CPU and are still a strong cheap baseline, but each word has exactly one vector regardless of context; for sentence/document meaning, use contextual embedding models ([[embeddings]]).

## Key concepts
- **Distributional hypothesis.** Meaning is approximated by the contexts a word appears in; the training objective never sees definitions.
- **word2vec (Mikolov et al. 2013).** A shallow network with one hidden (embedding) layer.
  - **Skip-gram:** predict surrounding words from the centre word. Better for rare words and small corpora.
  - **CBOW:** predict the centre word from the averaged context. Faster, slightly better for frequent words.
  - **Negative sampling:** instead of a full softmax over the vocabulary, classify the true (word, context) pair against k random "negative" words (k=5-20). Frequent-word **subsampling** drops very common tokens ("the") during training.
  - **Window size:** small (2-5) gives syntactic/functional similarity; large (10+) gives topical similarity.
- **GloVe (Stanford 2014).** Fits vectors so that dot products match log co-occurrence counts over the whole corpus. Distributed mainly as pretrained vectors (Wikipedia+Gigaword, Common Crawl, Twitter).
- **fastText (Facebook 2016).** A word vector is the sum of its character n-gram vectors (3-6 chars), so `"runninng"` still lands near `"running"`. Best for morphologically rich languages and noisy text; pretrained vectors exist for 157 languages.
- **Vector arithmetic.** `king - man + woman ≈ queen` works on large corpora; treat it as a demo, not a reliable property.
- **Document vectors.** Average (or TF-IDF-weighted average) of word vectors; or Doc2Vec. Simple averages lose word order.
- **Embedding layer.** `nn.Embedding` in a neural net is the same lookup table, learned end-to-end; pretrained word2vec/GloVe vectors can initialise it.

## When to use / scenarios
- Domain vocabulary that general models handle poorly (drug names, part numbers, internal jargon): train word2vec/fastText on your own corpus and use nearest neighbours for synonym lists and query expansion in search.
- E-commerce: "item2vec" - treat each user session as a sentence of product IDs and train word2vec; nearest neighbours become "related products" (see [[recommender-systems]]).
- Small text classifiers on CPU / edge where a transformer is too heavy: averaged fastText vectors + logistic regression, or the `fasttext` supervised classifier itself.
- Teaching and analysis: bias audits of vector spaces, linguistics studies of semantic change.
- NOT for semantic search, RAG or sentence similarity: one vector per word ignores context ("bank" of a river vs money) and averaging loses meaning; use sentence embedding models ([[embeddings]], [[embedding-models]]).
- NOT when you will fine-tune a transformer anyway; it learns its own (contextual) embeddings.

## Setup & code
```bash
pip install gensim==4.3.3
```
Train skip-gram on a toy corpus, query neighbours, and get a vector for an unseen misspelling with fastText:
```python
from gensim.models import FastText, Word2Vec

# Each "sentence" is a list of tokens. Real use: tokenise thousands to millions of lines.
corpus = [
    "the patient was given aspirin for chest pain".split(),
    "the patient received ibuprofen for back pain".split(),
    "aspirin and ibuprofen are common painkillers".split(),
    "the doctor prescribed aspirin to the patient".split(),
    "the nurse gave ibuprofen to the patient".split(),
    "chest pain and back pain were reported".split(),
] * 200  # repeat so the toy model has enough updates

w2v = Word2Vec(corpus, vector_size=50, window=3, min_count=1, sg=1,  # sg=1: skip-gram
               negative=5, epochs=10, seed=0, workers=1)
print(w2v.wv.most_similar("aspirin", topn=3))
print("aspirin~ibuprofen", round(w2v.wv.similarity("aspirin", "ibuprofen"), 3))
print("aspirin~nurse    ", round(w2v.wv.similarity("aspirin", "nurse"), 3))

# word2vec has no vector for an unseen word ...
print("asprin" in w2v.wv.key_to_index)  # False

# ... fastText builds one from character n-grams.
ft = FastText(corpus, vector_size=50, window=3, min_count=1, min_n=3, max_n=5,
              epochs=10, seed=0, workers=1)
print("fastText asprin~aspirin", round(ft.wv.similarity("asprin", "aspirin"), 3))

# Save only the vectors (KeyedVectors) for serving; much smaller than the full model.
w2v.wv.save("vectors.kv")
```
On a corpus this small all similarities come out high (aspirin~ibuprofen ≈0.98, aspirin~nurse ≈0.94); only the ranking is meaningful. Tested on Python 3.12; on 3.14 gensim had no prebuilt wheel and failed to build from source, so use a 3.12 environment (`uv run --python 3.12 --with gensim==4.3.3 ...`). Pretrained vectors: `import gensim.downloader as api; wv = api.load("glove-wiki-gigaword-100")` (downloads ~130 MB).

## Choosing / trade-offs
- **word2vec vs GloVe vs fastText.** Similar quality on analogy/similarity benchmarks when tuned. Pick fastText for typos, rare words and inflected languages; word2vec for non-text "tokens" (product IDs, graph walks as in node2vec); GloVe when you only need off-the-shelf English vectors.
- **Train your own vs pretrained.** Pretrained general vectors miss domain senses; your own corpus needs roughly millions of tokens for stable vectors. A common middle path: start from pretrained fastText and continue training on domain text.
- **Static vs contextual.** Static vectors: tiny, CPU, microsecond lookups, no context. Contextual (BERT-style, sentence-transformers): much better quality for anything sentence-level, but need more compute. For new projects default to contextual unless cost or footprint rules it out.
- **Dimensions.** 100-300 is the usual range; more dims need more data to fill.

## Gotchas
- Preprocessing must match between training and lookup (lowercasing, tokenisation, phrase joining like `new_york`); a mismatch silently returns "word not in vocabulary" or a poor vector.
- `min_count` (default 5) drops rare words; on small corpora important domain terms vanish without warning. Check `len(model.wv)`.
- Results vary between runs; for reproducibility set `seed` *and* `workers=1` (multi-threaded training is nondeterministic).
- Cosine similarity, not Euclidean distance; vector norms grow with word frequency.
- Antonyms ("hot"/"cold") come out *similar* because they share contexts; similarity is relatedness, not synonymy.
- Embeddings encode corpus biases (gender, ethnicity) and pass them to downstream models; audit before using them in decisions about people.
- Averaging word vectors for long documents washes out meaning; TF-IDF features often beat it on classification ([[nlp-classic-tasks]]).

## Related
- [[embeddings]] - contextual sentence/document embeddings, the modern replacement for most uses.
- [[embedding-models]] - which embedding model/API to pick today.
- [[nlp-classic-tasks]] - TF-IDF and classic NLP pipelines that these vectors plug into.
- [[tokenization]] - subword tokenisation, the descendant of fastText's n-gram idea.
- [[recommender-systems]] - item2vec and session-based embeddings.
- [[transformers-and-attention]] - the embedding layer inside a transformer.

## References
- Mikolov et al., Efficient Estimation of Word Representations in Vector Space: https://arxiv.org/abs/1301.3781
- Mikolov et al., Distributed Representations of Words and Phrases (negative sampling): https://arxiv.org/abs/1310.4546
- GloVe project page and pretrained vectors: https://nlp.stanford.edu/projects/glove/
- Bojanowski et al., Enriching Word Vectors with Subword Information (fastText): https://arxiv.org/abs/1607.04606
- gensim Word2Vec docs: https://radimrehurek.com/gensim/models/word2vec.html
