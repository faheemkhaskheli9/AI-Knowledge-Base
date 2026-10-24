---
title: Fine-tuning embedding models for retrieval (Sentence Transformers)
category: llm-apps
tags: [embeddings, fine-tuning, sentence-transformers, contrastive-learning, hard-negatives, matryoshka, retrieval, rag, mnrl]
use_cases:
  - "our RAG retrieval misses domain jargon; fine-tune the embedding model on our documents"
  - "build query-passage training pairs from our own corpus when we have no labelled data"
  - "measure whether a fine-tuned embedder actually beats the base model on recall@k"
  - "shrink embedding dimensions for a cheaper index without losing much recall"
status: draft
last_verified: 2026-10-24
sources:
  - https://sbert.net/docs/sentence_transformer/training_overview.html
  - https://sbert.net/docs/sentence_transformer/loss_overview.html
  - https://huggingface.co/blog/train-sentence-transformers
  - https://arxiv.org/abs/1705.00652
  - https://arxiv.org/abs/2004.04906
  - https://arxiv.org/abs/2205.13147
  - https://arxiv.org/abs/2101.06983
---

# Fine-tuning embedding models for retrieval (Sentence Transformers)

## Summary
A general-purpose embedding model can place domain queries ("pt c/o SOB", internal product codes, legal citations) far from the passages that answer them. Fine-tuning the embedder on (query, relevant passage) pairs from your own domain, with a contrastive loss that uses the other passages in the batch as negatives, often lifts retrieval recall more cheaply than switching to a bigger model. The workflow is: build pairs (logged queries or LLM-generated questions per chunk), mine hard negatives, train with Sentence Transformers' `SentenceTransformerTrainer` and `MultipleNegativesRankingLoss`, and accept the new model only if recall@k and MRR improve on a held-out set split by document. Then re-embed the whole corpus.

## Key concepts
- **Training data is pairs**: `(anchor=query, positive=passage)`, optionally a third `negative` column. Sources, best first: real user queries with clicked or cited passages; human-labelled pairs; LLM-generated questions per chunk, filtered ([[data-labeling-and-synthetic-data]]). A few thousand good pairs is a common starting point; measure rather than assume.
- **MultipleNegativesRankingLoss (MNRL, in-batch negatives)**: for a batch of B pairs, each query must score its own passage above the other B-1 passages (softmax cross-entropy over similarities, the InfoNCE objective; [[contrastive-learning-infonce-from-scratch-numpy]]). Bigger batches mean more negatives; `CachedMultipleNegativesRankingLoss` (GradCache) reaches large effective batches in fixed GPU memory.
- **Duplicates in a batch are false negatives**: two queries with the same positive passage in one batch push that passage away from itself. Use the `NO_DUPLICATES` batch sampler.
- **Hard negatives**: passages that the base model ranks high but are not relevant (BM25 or the base embedder's near misses, as in DPR). They teach finer distinctions; `sentence_transformers.util.mine_hard_negatives` mines them. Unlabelled relevant passages mined as "negatives" hurt, so skip the very top hits or filter with a reranker.
- **Matryoshka training**: `MatryoshkaLoss` wraps the base loss so that prefixes of the vector (for example the first 256 of 768 dims) also work as embeddings; truncate at query time for a smaller, cheaper index ([[vector-databases]]).
- **Evaluation**: `InformationRetrievalEvaluator(queries, corpus, relevant_docs)` reports recall@k, MRR@k, NDCG@k and MAP over a held-out query set searched against the full corpus. Split by document (or by time), never by random query, or near-duplicate questions leak.
- **Prefixes and prompts**: if the base model expects `query:` / `passage:` prefixes or an instruction, use the same ones in training data and at inference ([[embedding-models]]).

## When to use / scenarios
- RAG on specialised text (clinical notes, contracts, codebases, part catalogues) where retrieval evals show relevant chunks outside the top-k ([[rag-basics]], [[advanced-rag]]).
- Multilingual or code-mixed queries against one-language documents, when you can generate translated query pairs.
- High query volume where a fine-tuned small embedder replaces a large hosted one ([[cost-and-latency]]).
- Not first: try hybrid BM25 + vector search, better chunking and a reranker; they need no training and often close the gap ([[chunking-strategies]], [[rerankers]]).
- Not when you cannot re-embed the corpus (no compute, vectors owned by a vendor); a query-side linear adapter (below) keeps corpus vectors unchanged.
- Not possible for closed API embedders that offer no tuning; switch to an open model ([[embedding-models]]).

## Setup & code
```bash
pip install "sentence-transformers[train]" datasets
```
Training with Sentence Transformers v3+ (`<base-embedder>` is a Hugging Face model ID; column order matters: anchor, positive, then optional negative):
<!-- skip-check: downloads model weights; needs a GPU for realistic sizes -->
```python
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import mine_hard_negatives

model = SentenceTransformer("<base-embedder>")
train = Dataset.from_dict({
    "anchor":   ["query: what is the notice period for termination?", "query: max dose of drug X"],
    "positive": ["passage: Either party may terminate with 30 days written notice...",
                 "passage: The maximum daily dose of drug X is..."],
})
train = mine_hard_negatives(train, model, num_negatives=1)   # adds a negative column

# held-out queries, the full corpus, and relevant corpus IDs per query (split by document)
queries = {"q1": "query: how long is the notice period?"}
corpus = {"d1": "passage: Either party may terminate with 30 days written notice...",
          "d2": "passage: Invoices are payable within 45 days."}
relevant = {"q1": {"d1"}}
evaluator = InformationRetrievalEvaluator(queries, corpus, relevant, name="dev")
print("base:", evaluator(model))

loss = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model), matryoshka_dims=[768, 256, 64])
args = SentenceTransformerTrainingArguments(
    output_dir="out/embedder", num_train_epochs=1, per_device_train_batch_size=64,
    learning_rate=2e-5, warmup_ratio=0.1, batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=100,
)
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train,
                                     loss=loss, evaluator=evaluator)
trainer.train()
print("tuned:", evaluator(model))
model.save_pretrained("out/embedder/final")
```
Set `matryoshka_dims` from the base model's output dimension (largest first). Generating synthetic queries: prompt an LLM with each chunk to write one or two questions it answers, then drop questions that the chunk does not answer or that copy its wording ([[structured-output]]).

Runnable with NumPy only: why in-batch-negative training helps, on synthetic data. A "base model" embeds documents well but domain queries land off target (a fixed distortion plus noise). Training a query-side linear map with the MNRL gradient (no duplicate positives per batch) raises recall@1 and MRR on queries for held-out documents, and the corpus vectors never change, so no re-index is needed for this variant.
```python
import numpy as np

rng = np.random.default_rng(0)
DIM, N_DOCS = 64, 600

def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

# Base model: documents embed fine, but domain queries land off target.
docs = unit(rng.standard_normal((N_DOCS, DIM)))
R = np.linalg.qr(rng.standard_normal((DIM, DIM)))[0]          # fixed domain distortion
SHIFT = 0.55 * np.eye(DIM) + 0.45 * R

def make_queries(doc_ids, n_per_doc=3):
    q_ids = np.repeat(doc_ids, n_per_doc)
    q = unit(docs[q_ids] @ SHIFT.T + 1.4 / np.sqrt(DIM) * rng.standard_normal((len(q_ids), DIM)))
    return q, q_ids

# Split by DOCUMENT, not by query, so eval documents were never seen in training.
perm = rng.permutation(N_DOCS)
train_docs, eval_docs = perm[:450], perm[450:]
q_tr, y_tr = make_queries(train_docs)
q_ev, y_ev = make_queries(eval_docs, n_per_doc=2)

def evaluate(W, q, y):
    scores = unit(q @ W.T) @ docs.T                             # search the whole corpus
    rank = (scores > scores[np.arange(len(y)), y][:, None]).sum(1)   # 0 = top hit
    return {"recall@1": round(float(np.mean(rank < 1)), 3),
            "recall@10": round(float(np.mean(rank < 10)), 3),
            "mrr@10": round(float(np.mean(np.where(rank < 10, 1.0 / (rank + 1), 0.0))), 3)}

def mnrl_step(W, qb, db, scale=20.0, lr=0.05):
    """Multiple-negatives ranking loss: each query's positive vs the other positives in the batch."""
    logits = scale * (qb @ W.T) @ db.T
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits); p /= p.sum(1, keepdims=True)
    loss = -np.mean(np.log(p[np.arange(len(qb)), np.arange(len(qb))]))
    g_logits = (p - np.eye(len(qb))) / len(qb)
    g_W = (scale * g_logits @ db).T @ qb
    return W - lr * g_W, loss

W = np.eye(DIM)                                                 # start from the base model
base = evaluate(W, q_ev, y_ev)
print("base      ", base)
BATCH = 64
for epoch in range(30):
    order = rng.permutation(len(q_tr))
    for s in range(0, len(order) - BATCH + 1, BATCH):
        idx = order[s:s + BATCH]
        _, first = np.unique(y_tr[idx], return_index=True)     # no duplicate positives in a batch
        idx = idx[first]
        W, loss = mnrl_step(W, q_tr[idx], docs[y_tr[idx]])
    if epoch % 10 == 9:
        print(f"epoch {epoch + 1:2d}  ", evaluate(W, q_ev, y_ev), f"train loss {loss:.3f}")
assert evaluate(W, q_ev, y_ev)["recall@1"] > base["recall@1"]
```

## Choosing / trade-offs
| Option | Pick when | Cost |
|---|---|---|
| No tuning: hybrid search + reranker | first try, small eval gap | no training; reranker latency per query |
| Full fine-tune of an open embedder (MNRL) | clear domain gap, thousands of pairs, you control the index | GPU hours plus a full corpus re-embed and index rebuild |
| Query-side adapter (linear map on query vectors) | corpus vectors fixed or expensive to recompute, including vendor embeddings | small gain ceiling; only reshapes the query side |
| Fine-tune a reranker instead (cross-encoder) | first-stage recall@100 is fine, precision@5 is not | per-query reranker cost; see [[rerankers]] (Sentence Transformers v4+ has a `CrossEncoderTrainer`) |

- **Model size**: tuning a small base often matches a larger untuned one on in-domain queries and is cheaper to serve ([[embedding-and-reranker-serving]]); confirm on your eval set.
- **Synthetic vs real queries**: synthetic questions are cheap and plentiful but phrased like the passage; mix in real queries as soon as logs exist, and evaluate on real ones.
- **Hard negatives**: more and harder negatives sharpen ranking but raise the false-negative risk and training cost; start with one per pair.
- **Matryoshka dims**: smaller dims cut index memory and search cost roughly in proportion; measure the recall drop at each size before picking one.

## Gotchas
- Vectors from the fine-tuned model are not comparable with the old index. Re-embed everything into a new index and cut over; store `model_id@revision` with every vector ([[embedding-and-reranker-serving]]).
- Evaluating on queries generated from the same chunks used for training inflates scores; hold out whole documents, and keep a real-query set.
- Catastrophic forgetting: a narrow fine-tune can hurt queries outside the training domain; keep a general retrieval check (a slice of a public benchmark or older query logs) in the eval.
- Mined negatives that are actually relevant (duplicate passages, overlapping chunks) teach the model to push away correct answers; dedupe the corpus and skip the top few mined hits.
- Prefix mismatch: training without the `query:`/instruction prefix the model expects, then serving with it (or the reverse), silently costs recall.
- Learning rate too high or too many epochs on a small set collapses the embedding space; watch the evaluator, not the training loss.
- Licence: the fine-tune inherits the base model's licence ([[model-licenses]]).

## Related
- [[embedding-models]] - choosing the base embedder to fine-tune.
- [[embeddings]] - what embeddings are and how similarity works.
- [[contrastive-learning-infonce-from-scratch-numpy]] - the InfoNCE loss behind MNRL, from scratch.
- [[rerankers]] - the alternative or complement: a cross-encoder on top-k.
- [[embedding-and-reranker-serving]] - serving the tuned model, re-embedding and cache keys.
- [[advanced-rag]], [[rag-basics]] - where retrieval quality is measured and used.
- [[llm-evaluation]] - building the held-out eval set.
- [[data-labeling-and-synthetic-data]] - generating and checking query-passage pairs.
- [[metric-learning-and-few-shot]] - the same idea for non-text embeddings.
- [[late-interaction-retrieval]] - multi-vector retrieval as another fix for missed matches.
- [[embedding-model-migration-and-reindexing]] - re-embedding the corpus and cutting over to the tuned model.

## References
- Sentence Transformers training overview (trainer, datasets, evaluators): https://sbert.net/docs/sentence_transformer/training_overview.html
- Sentence Transformers loss overview (MNRL, cached MNRL, Matryoshka): https://sbert.net/docs/sentence_transformer/loss_overview.html
- Hugging Face blog, Training and finetuning embedding models with Sentence Transformers v3: https://huggingface.co/blog/train-sentence-transformers
- Henderson et al., Efficient Natural Language Response Suggestion for Smart Reply (in-batch negatives ranking loss, 2017): https://arxiv.org/abs/1705.00652
- Karpukhin et al., Dense Passage Retrieval (hard negatives from BM25, 2020): https://arxiv.org/abs/2004.04906
- Kusupati et al., Matryoshka Representation Learning (2022): https://arxiv.org/abs/2205.13147
- Gao et al., Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup (GradCache, 2021): https://arxiv.org/abs/2101.06983
