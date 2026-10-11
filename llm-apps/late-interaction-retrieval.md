---
title: Late-interaction and multi-vector retrieval (ColBERT, ColPali)
category: llm-apps
tags: [colbert, colpali, colqwen, late-interaction, multi-vector, maxsim, plaid, visual-document-retrieval, rag]
use_cases:
  - "retrieve PDF pages with charts and tables without an OCR-then-chunk pipeline"
  - "get reranker-like precision from first-stage retrieval over a text corpus"
  - "decide between dense+rerank and ColBERT-style retrieval for a RAG system"
  - "estimate index storage for a multi-vector retriever before choosing a vector DB"
status: draft
last_verified: 2026-10-11
sources:
  - https://arxiv.org/abs/2004.12832
  - https://arxiv.org/abs/2112.01488
  - https://arxiv.org/abs/2205.09707
  - https://arxiv.org/abs/2407.01449
  - https://github.com/illuin-tech/colpali
  - https://qdrant.tech/documentation/manage-data/vectors/
  - https://docs.weaviate.io/weaviate/tutorials/multi-vector-embeddings
  - https://docs.vespa.ai/en/rag/embedding.html
  - https://huggingface.co/blog/multi-vector-encoder
---

# Late-interaction and multi-vector retrieval (ColBERT, ColPali)

## Summary
Dense retrieval squeezes a document into one vector; a cross-encoder reranker
reads query and document together but is too slow for a whole corpus.
Late interaction sits between them: the document is encoded offline into one
vector per token (ColBERT) or per image patch (ColPali, ColQwen), the query into
one vector per token, and the score is MaxSim: for every query vector take its
best-matching document vector, then sum. You get much of a reranker's
token-level precision at index time, paid for with many more stored vectors.
ColPali applies this to page images, so a PDF page with tables and charts can be
retrieved without OCR, layout parsing or chunking.

## Key concepts
- **MaxSim score**: `score(Q, D) = sum_i max_j  q_i . d_j` over normalised
  vectors (Khattab & Zaharia, 2020). Qdrant implements the same sum-of-max
  definition as its `MAX_SIM` multivector comparator.
- **Late vs early interaction**: a bi-encoder interacts nothing (one dot
  product), a cross-encoder interacts everything inside the transformer, late
  interaction interacts only at the final vectors, so document vectors can be
  precomputed.
- **ColBERT** (2020): BERT token embeddings projected to 128 dims. **ColBERTv2**
  (2021) adds residual compression: a 4-byte centroid ID plus a 1- or 2-bit
  quantised residual, 20 or 36 bytes per 128-dim vector instead of 256 bytes at
  16-bit precision (PLAID paper, section 3.1). **PLAID** is the search engine
  that prunes candidates by centroid before full MaxSim.
- **ColPali** (2024): PaliGemma-3B patch outputs projected to D=128, giving a
  multi-vector per page image; the paper reports 257.5 KB per page. Successors
  in the `colpali` repo's model table include ColQwen2/2.5 (Apache 2.0,
  dynamic resolution, trained with 768 image patches per page) and ColSmol
  (256M/500M) for small hardware.
- **Token pooling**: cluster similar vectors (blank background patches) and
  average them. The ColPali paper reports a pool factor of 3 cuts vectors by
  66.7% while keeping 97.8% of retrieval performance, with text-dense pages
  degrading most.
- **Library status (2026-10)**: the `colpali-engine` package is deprecated;
  its README points new projects to Sentence Transformers v6
  `MultiVectorEncoder`, which loads the `vidore/*` checkpoints.

## When to use / scenarios
- Visually rich documents (slide decks, financial reports, scanned forms,
  manuals with diagrams) where OCR-then-chunk loses tables and figures:
  ColPali/ColQwen page retrieval, then send the page image to a VLM
  ([[vision-language-models]]).
- Out-of-domain or keyword-heavy text (part numbers, legal terms) where a
  single dense vector blurs details: ColBERT as first stage, or as a reranker
  over BM25/dense candidates.
- Use **dense + reranker** ([[embedding-models]], [[rerankers]]) instead when
  the corpus is large and mostly plain text, storage cost matters, or your
  vector DB has no multivector support. It is the cheaper default; measure
  late interaction against it on your own queries before switching.
- Use **OCR/parsing** ([[document-parsing]], [[ocr]]) when you need the text
  itself (field extraction, citations by character offset, search over exact
  strings). Page-image retrieval returns pages, not text.

## Setup & code
MaxSim and the storage arithmetic in NumPy, with random vectors standing in
for model output:
```python
import numpy as np

rng = np.random.default_rng(0)

def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def maxsim(q, d):
    """q: (Lq, D) query vectors, d: (Ld, D) doc vectors, both L2-normalised."""
    return (q @ d.T).max(axis=1).sum()

D = 128
docs = [norm(rng.normal(size=(n, D))) for n in (40, 60, 80)]
# a query whose 5 tokens are noisy copies of tokens in doc 1
query = norm(docs[1][[3, 10, 20, 30, 50]] + 0.3 * rng.normal(size=(5, D)))
scores = [maxsim(query, d) for d in docs]
print("MaxSim scores:", np.round(scores, 3))
assert int(np.argmax(scores)) == 1

# single-vector baseline: mean-pool each side, one dot product
pooled = [float(norm(query.mean(0)) @ norm(d.mean(0))) for d in docs]
print("mean-pooled cosine:", np.round(pooled, 3))

# index size per page/passage: vectors * dims * bytes
def size_kib(n_vectors, dim=128, bytes_per_value=2):
    return n_vectors * dim * bytes_per_value / 1024

print("ColPali page, 1030 x 128 x bf16:", size_kib(1030), "KiB")  # 257.5, as in the paper
print("same page, pool factor 3:", round(size_kib(1030 / 3), 1), "KiB")
print("one 1024-dim float32 dense vector:", size_kib(1, 1024, 4), "KiB")
print("ColBERTv2 2-bit residual, 100 tokens:", 100 * 36 / 1024, "KiB")
```
Real models via Sentence Transformers v6 (per the `colpali` README; downloads
a 2-3B checkpoint and wants a GPU):
<!-- skip-check: downloads a multi-GB model and needs a GPU -->
```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("vidore/colqwen2.5-v0.2")
q = model.encode_query(["Which year has the highest total outlay?"])
d = model.encode_document(["page_1.png", "page_2.png"])  # paths, URLs or PIL images
print(model.similarity(q, d))
```
Storage: Qdrant stores a multivector per point with
`multivector_config=MultiVectorConfig(comparator=MAX_SIM)`; its tutorials keep
HNSW for a dense vector and disable it (`m=0`) on the multivector, using
MaxSim only to rerank the dense candidates. Qdrant caps a multivector at
`number_of_sub_vectors * vector_size < 1,048,576`.

## Choosing / trade-offs
| | Dense + reranker | Late interaction |
|---|---|---|
| Index size | 1 vector per chunk | 1 vector per token/patch (tens to ~1000x more) |
| Query cost | ANN + reranker call on top-k | ANN/centroid pruning + MaxSim on candidates |
| Precision | reranker recovers it, on top-k only | token-level matching at first stage |
| Visual pages | needs OCR/parsing first | ColPali embeds the page image |
| Tooling | every vector DB | needs multivector support (Qdrant, Vespa, Weaviate) or PLAID |

- Two-stage hybrid is common: cheap dense (or pooled) vector for ANN recall,
  MaxSim over the stored multivectors to rerank the top few hundred.
- Compression (ColBERTv2 residuals, token pooling, binary quantisation) is the
  main cost lever; each costs some quality, most on text-dense pages.
- Licences differ: ColPali checkpoints inherit the Gemma licence, ColQwen2/2.5
  are Apache 2.0 (model table in the `colpali` repo).

## Gotchas
- Storage explodes quietly: a 100k-page archive at ColPali's 257.5 KB/page is
  about 25 GB of raw vectors before any index overhead.
- Do not normalise MaxSim by document length; it sums over query tokens, so
  longer queries give larger scores. Compare scores only within one query.
- Padding and special tokens: masked-out query/document positions must be
  dropped before MaxSim, or pad vectors win the max.
- Page-level retrieval gives the VLM a whole page; for answer citations you
  still need the text, so many systems keep OCR for display and ColPali for
  retrieval.
- `colpali-engine` is deprecated; older tutorials use `ColQwen2.from_pretrained`
  and `processor.score_multi_vector`, which map to `MultiVectorEncoder` and
  `model.similarity` in Sentence Transformers v6.

## Related
- [[advanced-rag]] - where late interaction fits among hybrid search and reranking.
- [[rerankers]] - the cross-encoder alternative it is usually compared against.
- [[embedding-models]] - single-vector dense retrieval.
- [[vector-databases]] - multivector support differs by engine.
- [[document-parsing]] - the OCR/parse-then-chunk route ColPali replaces.
- [[vision-language-models]] - reads the retrieved page image.
- [[document-processing]] - scenarios over PDF archives.

## References
- Khattab & Zaharia, ColBERT (2020): https://arxiv.org/abs/2004.12832
- Santhanam et al., ColBERTv2 (2021): https://arxiv.org/abs/2112.01488
- Santhanam et al., PLAID (2022): https://arxiv.org/abs/2205.09707
- Faysse et al., ColPali (2024): https://arxiv.org/abs/2407.01449
- ColPali repo, model table and deprecation note: https://github.com/illuin-tech/colpali
- Qdrant multivectors: https://qdrant.tech/documentation/manage-data/vectors/
- Weaviate multi-vector embeddings: https://docs.weaviate.io/weaviate/tutorials/multi-vector-embeddings
- Vespa ColBERT embedder: https://docs.vespa.ai/en/rag/embedding.html
- Sentence Transformers `MultiVectorEncoder`: https://huggingface.co/blog/multi-vector-encoder
