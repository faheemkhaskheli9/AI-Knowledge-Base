---
title: BM25 from scratch (keyword retrieval, IDF, term saturation, length normalization)
category: llm-apps
tags: [bm25, okapi-bm25, information-retrieval, keyword-search, sparse-retrieval, tf-idf, hybrid-search, rag, search, from-scratch]
use_cases:
  - "implement Okapi BM25 ranking in plain Python without a search engine"
  - "add a keyword retriever next to vector search for hybrid RAG"
  - "understand what the k1 and b parameters do and when to tune them"
  - "explain BM25 vs TF-IDF vs dense embeddings in a search or RAG interview"
status: draft
last_verified: 2026-10-04
sources:
  - https://doi.org/10.1561/1500000019
  - https://lucene.apache.org/core/9_0_0/core/org/apache/lucene/search/similarities/BM25Similarity.html
  - https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html
---

# BM25 from scratch (keyword retrieval, IDF, term saturation, length normalization)

## Summary
BM25 is the default ranking function of Lucene, Elasticsearch and OpenSearch, and the keyword half of most hybrid RAG retrievers. It scores a document by adding, for each query term, the term's rarity (IDF) times a term-frequency factor. That factor saturates, so the tenth repeat of a word adds little, and it is normalized by document length, so long documents do not win just by containing more words. Below, a 40-line implementation ranks eight help-centre snippets for seven queries. One snippet is a long "account settings" page that repeats its keywords. Counting query-term occurrences ranks that page first for "account settings". BM25 ranks the short, relevant snippet first, and MRR rises from 0.857 to 0.929. Turning length normalization off (`b = 0`) brings the stuffed page back to first place.

## Key concepts
- **Score.** `score(q, d) = Σ_t∈q IDF(t) · tf(t,d)·(k1 + 1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))`.
- **IDF.** `log(1 + (N − n_t + 0.5)/(n_t + 0.5))`, where `n_t` is the number of documents containing `t`. Rare terms (error codes, product names) carry most of the score. The `1 +` (Lucene's variant) keeps IDF positive for terms that appear in more than half the documents.
- **Saturation (`k1`).** The tf factor rises towards `k1 + 1` and never exceeds it. `k1 = 0` ignores term frequency entirely, and a large `k1` is close to raw tf. Typical values are 1.2-2.0.
- **Length normalization (`b`).** `b = 1` fully scales term frequency by `|d|/avgdl`, and `b = 0` ignores length. The usual default is 0.75.
- **Bag of words.** Order, synonyms and meaning are ignored. "Car" does not match "automobile". Matching depends entirely on the tokenizer: lowercasing, stop words, stemming.

## When to use / scenarios
- Learning: the baseline every retrieval method is compared against, and the clearest way to see IDF and length normalization at work.
- Interviews: "how does Elasticsearch rank", "BM25 vs TF-IDF", "why hybrid search", "what do k1 and b do".
- Practice: the sparse half of hybrid RAG (fuse with vector search via reciprocal rank fusion, see [[advanced-rag]]); exact identifiers that embeddings blur (error codes, SKUs, function names, legal citations); a strong zero-training baseline before you pay for embeddings; first-stage retrieval before a cross-encoder reranker.
- Not for: paraphrase or cross-lingual queries, where nothing overlaps literally (use dense embeddings, [[rag-basics]]); or ranking with many learned signals such as clicks and freshness (use [[learning-to-rank]] on top).

## Setup & code
Standard library only.

```python
import math
import re
from collections import Counter

docs = [
    "Reset your password from the account settings page.",
    "To reset a forgotten password click 'Forgot password' on the login screen.",
    "Error code E1042 means the payment card was declined by the bank.",
    "Our refund policy allows returns within 30 days of delivery.",
    "Two-factor authentication adds a one-time code to the login process.",
    "Shipping usually takes 3 to 5 business days within the country.",
    "The account page lets you change your email, password, address, phone number, "
    "notification settings, language, time zone, privacy options, linked devices, "
    "billing details, saved cards, order history, newsletter preferences and more. "
    "Account account account settings settings.",
    "If the login screen shows error E2001 your session expired; log in again.",
]
STOP = {"the", "a", "to", "your", "of", "on", "by", "my", "i", "is", "and", "from", "in", "how", "do", "what", "does", "was", "means"}


def tokenize(s):
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP]


class BM25:
    def __init__(self, corpus, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in corpus]
        self.len = [sum(d.values()) for d in self.docs]
        self.avgdl = sum(self.len) / len(self.len)
        N = len(corpus)
        df = Counter(t for d in self.docs for t in d)
        # Lucene-style IDF: always positive
        self.idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def score(self, query):
        q = tokenize(query)
        out = []
        for d, dl in zip(self.docs, self.len):
            s = 0.0
            for t in q:
                tf = d.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


def tf_overlap(query):                      # naive baseline: count query-term occurrences
    q = tokenize(query)
    return [sum(Counter(tokenize(d))[t] for t in q) for d in docs]


queries = [  # (query, index of the relevant doc)
    ("I forgot my password", 1),
    ("change password settings", 0),
    ("what does error E1042 mean", 2),
    ("account settings", 0),
    ("login code", 4),
    ("refund within 30 days", 3),
    ("session expired error on login", 7),
]


def rank_of(scores, rel):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return order.index(rel) + 1


def mrr(scorer):
    return sum(1 / rank_of(scorer(q), rel) for q, rel in queries) / len(queries)


bm = BM25(docs)
print(f"{len(docs)} docs, avg length {bm.avgdl:.1f} tokens (doc 6 has {bm.len[6]})")
print(f"idf: password {bm.idf['password']:.2f}, e1042 {bm.idf['e1042']:.2f}, login {bm.idf['login']:.2f}")
print("\nquery                               relevant  rank(tf)  rank(bm25)")
for q, rel in queries:
    print(f"{q:<36}{rel:>8}{rank_of(tf_overlap(q), rel):>10}{rank_of(bm.score(q), rel):>12}")
print(f"MRR  term-count {mrr(tf_overlap):.3f}   BM25 {mrr(bm.score):.3f}")

print("\nterm-frequency saturation, one term idf=1, doc of average length:")
for k1 in [0.5, 1.2, 2.0]:
    print(f"  k1={k1}: " + "  ".join(f"tf={tf}:{tf * (k1 + 1) / (tf + k1):.2f}" for tf in [1, 2, 5, 20]))

print("\nsweep on 'account settings' (relevant doc 0, the keyword-stuffed doc is 6):")
for k1, b in [(1.2, 0.0), (1.2, 0.75), (1.2, 1.0), (3.0, 0.75)]:
    s = BM25(docs, k1, b).score("account settings")
    print(f"  k1={k1} b={b:<4} doc0 {s[0]:.2f}  doc6 {s[6]:.2f}  -> rank of doc0: {rank_of(s, 0)}   MRR {mrr(BM25(docs, k1, b).score):.3f}")
```

Output (Python 3.14):
```
8 docs, avg length 11.2 tokens (doc 6 has 33)
idf: password 0.94, e1042 1.79, login 0.94

query                               relevant  rank(tf)  rank(bm25)
I forgot my password                       1         1           1
change password settings                   0         2           2
what does error E1042 mean                 2         1           1
account settings                           0         2           1
login code                                 4         1           1
refund within 30 days                      3         1           1
session expired error on login             7         1           1
MRR  term-count 0.857   BM25 0.929

term-frequency saturation, one term idf=1, doc of average length:
  k1=0.5: tf=1:1.00  tf=2:1.20  tf=5:1.36  tf=20:1.46
  k1=1.2: tf=1:1.00  tf=2:1.38  tf=5:1.77  tf=20:2.08
  k1=2.0: tf=1:1.00  tf=2:1.50  tf=5:2.14  tf=20:2.73

sweep on 'account settings' (relevant doc 0, the keyword-stuffed doc is 6):
  k1=1.2 b=0.0  doc0 2.56  doc6 4.18  -> rank of doc0: 2   MRR 0.857
  k1=1.2 b=0.75 doc0 3.32  doc6 3.05  -> rank of doc0: 1   MRR 0.929
  k1=1.2 b=1.0  doc0 3.68  doc6 2.80  -> rank of doc0: 1   MRR 1.000
  k1=3.0 b=0.75 doc0 3.73  doc6 3.29  -> rank of doc0: 1   MRR 1.000
```

MRR is mean reciprocal rank: 1.0 means the relevant snippet is always first. The error code "e1042" appears in one document, so its IDF (1.79) is almost twice that of "password" or "login" (0.94), and any document containing it jumps to the top. That is why keyword search beats embeddings on identifiers. The term-count baseline puts the 33-token settings page first for "account settings", because it repeats both words. BM25 caps the benefit of repeats (at k1 = 1.2, twenty occurrences score only 2.08x a single one) and divides by length, so the short snippet wins. The sweep isolates `b`. At `b = 0` the stuffed page wins again, and at `b = 1` "change password settings" also gets fixed. On a corpus this small, tuning on seven queries is overfitting. On a real corpus, keep the defaults unless a labelled query set says otherwise.

## Choosing / trade-offs
- **BM25 vs TF-IDF cosine.** Both weight by IDF. BM25 adds term-frequency saturation and a tunable length normalization, and usually ranks better with no tuning. TF-IDF vectors remain useful as features for classifiers.
- **BM25 vs dense embeddings.** BM25 needs no model or GPU, is explainable, and is excellent on rare exact terms. Embeddings handle synonyms and paraphrases. Hybrid with reciprocal rank fusion usually beats either alone ([[advanced-rag]], [[vector-databases]]).
- **Learned sparse (SPLADE) and query expansion.** These keep the inverted index but add learned or expanded terms to fix vocabulary mismatch, at the cost of a model at index time.
- **Library vs scratch.** For production use an inverted index (Elasticsearch/OpenSearch, Tantivy, Postgres full-text, the `bm25s` or `rank-bm25` Python packages for small corpora). This scratch version scans every document per query, which is O(N).
- **Variants.** BM25+ adds a floor so long documents do not score near zero, and BM25F weights fields (title vs body). Field boosting in Elasticsearch is the practical equivalent.

## Gotchas
- The tokenizer is half the model. Different lowercasing, stemming or stop words at index time and at query time silently break matching. Splitting "E1042" into "e" and "1042", or dropping digits, kills exact-code search.
- Stop-word lists can delete meaning ("to be or not to be", "vitamin A"). Prefer keeping them and letting IDF down-weight them.
- IDF depends on the corpus. Scores are not comparable across indexes or after large re-indexing, so never threshold raw BM25 scores across collections.
- The classic Robertson IDF `log((N − n + 0.5)/(n + 0.5))` goes negative for terms in more than half the documents. Use the Lucene `1 +` form or clamp at zero.
- Chunk length interacts with `b`. In RAG, very uneven chunk sizes make length normalization dominate. Chunk to roughly even sizes ([[document-parsing]]).
- Fusing raw BM25 scores with cosine similarities by adding them is meaningless, because their scales differ. Fuse ranks (RRF) or normalize scores per query first.

## Related
- [[advanced-rag]] - hybrid BM25 + vector retrieval with RRF and reranking.
- [[rag-basics]] - the dense-retrieval baseline this complements.
- [[vector-databases]] - stores that offer built-in hybrid search.
- [[learning-to-rank]] - learned rankers that use BM25 as one feature.
- [[locality-sensitive-hashing-from-scratch]] - approximate nearest-neighbour search for the dense side.
- [[tokenization]] - why text splitting decides what can match.

## References
- Robertson and Zaragoza (2009), "The Probabilistic Relevance Framework: BM25 and Beyond", Foundations and Trends in IR: https://doi.org/10.1561/1500000019
- Apache Lucene, BM25Similarity (IDF formula and defaults): https://lucene.apache.org/core/9_0_0/core/org/apache/lucene/search/similarities/BM25Similarity.html
- Elasticsearch reference, similarity module (BM25 `k1` and `b` defaults): https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html
