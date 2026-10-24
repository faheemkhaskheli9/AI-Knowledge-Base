---
title: Serving embedding and reranker models (TEI, vLLM pooling, Sentence Transformers)
category: deployment
tags: [embeddings, rerankers, text-embeddings-inference, tei, vllm, sentence-transformers, cross-encoder, batching, caching, self-hosting]
use_cases:
  - "self-host an open embedding model behind an HTTP API for our RAG pipeline"
  - "serve a cross-encoder reranker with low latency next to the vector database"
  - "re-embed a 50M-chunk corpus quickly without overloading the query-time service"
  - "cut embedding API cost by caching vectors and batching requests"
status: draft
last_verified: 2026-10-24
sources:
  - https://huggingface.co/docs/text-embeddings-inference/index
  - https://huggingface.co/docs/text-embeddings-inference/quick_tour
  - https://docs.vllm.ai/en/latest/models/pooling_models.html
  - https://sbert.net/docs/sentence_transformer/usage/efficiency.html
  - https://sbert.net/docs/cross_encoder/usage/usage.html
---

# Serving embedding and reranker models (TEI, vLLM pooling, Sentence Transformers)

## Summary
Embedding and reranker models are encoders: one forward pass per input, no token-by-token generation, so serving them is a batching and padding problem, not a KV-cache problem. Self-host one with Hugging Face Text Embeddings Inference (TEI), vLLM's pooling-model support, or Sentence Transformers inside your own service; or call a hosted API. The work that pays off is the same either way: batch by token budget, cache vectors by content hash, keep query-time and bulk re-embedding traffic apart, and pin the exact model revision because vectors from different models (or revisions) are not comparable.

## Key concepts
- **Two workloads, one model**: query-time embedding (one short query, latency-bound, tens of milliseconds matter) and indexing (millions of chunks, throughput-bound). Run them on separate replicas or queues so a backfill never starves search.
- **Dynamic batching by tokens**: servers group concurrent requests into one forward pass up to a token budget (TEI exposes `--max-batch-tokens` and `--max-client-batch-size`). Cost per batch grows with the longest sequence in it, so sorting inputs by length before batching cuts padding waste.
- **Rerankers are cross-encoders**: they score each (query, document) pair jointly, so cost is `candidates x (query + doc length)` per search. Rerank the top 20-100 retrieved chunks, not the whole index ([[rerankers]]).
- **Endpoints**: TEI serves `/embed` (`{"inputs": [...]}`), `/rerank` (`{"query", "texts"}` returning index and score per text) and an OpenAI-compatible `/v1/embeddings`. vLLM serves pooling models through `/v1/embeddings` and score/rerank endpoints; the flag that selects pooling mode has changed between releases, so check its pooling-models page.
- **Normalization and similarity**: most retrieval models expect cosine similarity; L2-normalize once at write time and use dot product in the index ([[vector-databases]]).
- **Query/document prefixes**: many open models need an instruction or `query:`/`passage:` prefix (see the model card). The server does not add it for you; a missing prefix silently lowers recall ([[embedding-models]]).
- **Model identity is part of the data**: store `model_id@revision`, dimension and normalization with every vector. Changing any of them means a full re-embed into a new index, then a cut-over.

## When to use / scenarios
- RAG over private documents that may not leave your network: self-host an open embedding model and reranker next to the vector database ([[rag-basics]], [[ai-security-privacy-compliance]]).
- High-volume semantic search or deduplication where per-token API cost dominates ([[cost-and-latency]]).
- Multilingual or domain models (code, legal) not offered by your API provider.
- Corpus backfills after a model change: run as an offline batch job on rented GPUs ([[batch-inference-and-batch-apis]], [[gpu-cloud-options]]).
- Not worth it for small corpora or low query volume: a hosted embedding API is simpler and the bill is small ([[embedding-models]]).
- Not for generation: an LLM server tuned for decoding is the wrong shape; use [[inference-servers-vllm]] for that.

## Setup & code
TEI in Docker (pick the image tag for your GPU or a `cpu-` tag from the TEI docs; `<embedding-model>` is a Hugging Face repo ID):
```bash
docker run --gpus all -p 8080:80 -v $PWD/hf-cache:/data \
  ghcr.io/huggingface/text-embeddings-inference:<tag> --model-id <embedding-model>
curl -s localhost:8080/embed -H 'Content-Type: application/json' \
  -d '{"inputs": ["query: what is goodput?"]}'
# a reranker model in a second container exposes /rerank:
curl -s localhost:8081/rerank -H 'Content-Type: application/json' \
  -d '{"query": "refund policy", "texts": ["Refunds take 5 days.", "We ship worldwide."]}'
```
In-process with Sentence Transformers (good for batch jobs and small services):
<!-- skip-check: downloads model weights from the Hugging Face Hub -->
```python
from sentence_transformers import SentenceTransformer, CrossEncoder

emb = SentenceTransformer("<embedding-model>", revision="<commit-sha>")   # pin the revision
vecs = emb.encode(["passage: Refunds take 5 days.", "passage: We ship worldwide."],
                  batch_size=64, normalize_embeddings=True)
rr = CrossEncoder("<reranker-model>")
print(vecs.shape, rr.predict([("refund policy", "Refunds take 5 days."),
                              ("refund policy", "We ship worldwide.")]))
```
Runnable with NumPy only: the client-side plumbing that matters whichever server you use. A stand-in encoder (deterministic random vectors whose cost is batch size x longest input) shows the effect of length-sorted token-budget batching, a content-hash cache keyed by model revision, normalization, and int8 storage.
```python
import hashlib
import numpy as np

MODEL = "example-embedder@rev-abc123"        # model id + revision is part of the cache key
DIM = 384
rng = np.random.default_rng(0)
words = "refund policy shipping invoice account login error billing export data".split()
docs = [" ".join(rng.choice(words, size=int(rng.integers(5, 400)))) for _ in range(2000)]
docs += docs[:300]                            # duplicates, as in real corpora

def n_tokens(text):                           # stand-in for the model's tokenizer
    return len(text.split())

def fake_encode(batch):
    """Stand-in encoder: returns vectors and the padded token cost of the batch."""
    cost = len(batch) * max(n_tokens(t) for t in batch)
    vecs = np.stack([np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16))
                     .standard_normal(DIM) for t in batch])
    return vecs, cost

def batches(texts, max_tokens=8192, sort=True):
    order = sorted(range(len(texts)), key=lambda i: n_tokens(texts[i])) if sort else range(len(texts))
    cur, longest = [], 0
    for i in order:
        longest_new = max(longest, n_tokens(texts[i]))
        if cur and longest_new * (len(cur) + 1) > max_tokens:   # padded size would exceed budget
            yield cur; cur, longest_new = [], n_tokens(texts[i])
        cur.append(i); longest = longest_new
    if cur:
        yield cur

cache = {}
def key(text):
    return hashlib.sha256(f"{MODEL}\x00{text}".encode()).hexdigest()

def embed_all(texts, sort=True):
    todo = list(dict.fromkeys(t for t in texts if key(t) not in cache))   # dedupe + cache
    cost = 0
    for idx in batches(todo, sort=sort):
        vecs, c = fake_encode([todo[i] for i in idx]); cost += c
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)              # normalize once
        for i, v in zip(idx, vecs):
            cache[key(todo[i])] = v
    return np.stack([cache[key(t)] for t in texts]), cost, len(todo)

real = sum(n_tokens(t) for t in dict.fromkeys(docs))
_, cost_unsorted, _ = embed_all(docs, sort=False); cache.clear()
X, cost_sorted, encoded = embed_all(docs, sort=True)
print(f"real tokens {real}, padded cost unsorted {cost_unsorted}, sorted {cost_sorted}")
print(f"encoded {encoded} unique of {len(docs)} inputs; second call encodes {embed_all(docs)[2]}")

scale = 127 / np.abs(X).max()                 # int8 storage: 4x smaller than float32
Xq = np.round(X * scale).astype(np.int8)
q = X[0]
top_f = np.argsort(-(X @ q))[:10]
top_q = np.argsort(-(Xq.astype(np.float32) @ q))[:10]
print("top-10 overlap float32 vs int8:", len(set(top_f) & set(top_q)) / 10)
```

## Choosing / trade-offs
| Option | Pick when | Watch out for |
|---|---|---|
| Hosted embedding/rerank API | small to medium volume, no GPU ops, data may leave the network | per-token cost at scale, rate limits, model retirement forces a re-embed |
| TEI | dedicated embedding/reranker service on GPU or CPU, many supported encoder architectures | one model per container; check the model architecture is supported |
| vLLM pooling models | you already run vLLM, or the embedder is an LLM-sized decoder (for example Qwen3-Embedding) | pooling flags and endpoints change between releases; read the docs for your version |
| Sentence Transformers in-process | batch jobs, prototypes, CPU-only small models | you write batching, concurrency and health checks yourself |

- **GPU vs CPU**: small encoders serve query traffic acceptably on CPU; large LLM-based embedders and bulk backfills need a GPU. Measure latency at your concurrency ([[llm-load-testing-and-benchmarking]]).
- **Bigger model vs reranker**: a small embedder plus a reranker on the top candidates often beats a larger embedder alone at lower index cost. Check on your eval set ([[advanced-rag]]).
- **Dimension and precision**: truncated (Matryoshka) dimensions and int8/binary vectors cut storage and search cost at some recall loss; measure recall@k before adopting.

## Gotchas
- Mixing vectors from two models, two revisions or with and without normalization in one index gives plausible but wrong rankings with no error.
- Truncation: inputs beyond the model's max length are cut (often silently). Chunk to fit the model's limit ([[chunking-strategies]]).
- A backfill sharing replicas with search pushes query latency up; give bulk jobs their own replicas or a low-priority queue.
- Caching by text alone breaks when you change the model; include the model ID and revision in the key.
- Rerank latency scales with candidate count and document length; cap both and set a timeout that falls back to the first-stage order.
- Unpinned Hub models can change under you; pin a commit SHA and record it with the index ([[huggingface-transformers]]).

## Related
- [[embedding-models]] - which embedding model to serve.
- [[rerankers]] - which reranker to serve and when reranking pays off.
- [[vector-databases]] - where the vectors go; similarity metric and quantization.
- [[rag-basics]], [[advanced-rag]] - the pipelines that call these services.
- [[semantic-caching]] - a different cache: reusing LLM answers keyed by query embedding.
- [[batch-inference-and-batch-apis]] - running corpus backfills as offline jobs.
- [[llm-load-testing-and-benchmarking]] - measuring latency and throughput of the service.
- [[docker-for-ml]] - GPU containers and model-cache volumes.
- [[ray-serve-bentoml-and-litserve]] - general model-serving frameworks with server-side batching for encoders.
- [[embedding-model-migration-and-reindexing]] - the zero-downtime procedure for a full re-embed.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/caching-strategies.md - cache keys, invalidation and hit rates.

## References
- Text Embeddings Inference docs: https://huggingface.co/docs/text-embeddings-inference/index
- TEI quick tour (Docker, `/embed`, `/rerank`): https://huggingface.co/docs/text-embeddings-inference/quick_tour
- vLLM pooling models: https://docs.vllm.ai/en/latest/models/pooling_models.html
- Sentence Transformers efficiency (ONNX, OpenVINO, quantization): https://sbert.net/docs/sentence_transformer/usage/efficiency.html
- Sentence Transformers CrossEncoder usage: https://sbert.net/docs/cross_encoder/usage/usage.html
