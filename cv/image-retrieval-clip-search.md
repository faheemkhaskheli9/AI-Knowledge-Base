---
title: Image retrieval and CLIP search (text-to-image, image-to-image, dedup)
category: cv
tags: [image-retrieval, clip, siglip, open-clip, visual-search, reverse-image-search, near-duplicate, embeddings, faiss, vector-search]
use_cases:
  - "let users search a photo library with text like 'red dress on a beach'"
  - "find visually similar products from a customer's photo"
  - "detect duplicate and near-duplicate images in a large dataset"
  - "build a reverse image search over my company's image assets"
  - "zero-shot tag images with a list of labels without training"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/mlfoundations/open_clip
  - https://github.com/openai/CLIP
  - https://arxiv.org/abs/2103.00020
  - https://huggingface.co/docs/transformers/model_doc/siglip2
  - https://arxiv.org/abs/2502.14786
  - https://github.com/facebookresearch/faiss
---

# Image retrieval and CLIP search (text-to-image, image-to-image, dedup)

## Summary
CLIP-style models embed images and text into one vector space, so a text query or a query image can be matched against an index of image embeddings by cosine similarity. This one mechanism gives text-to-image search, "more like this" image search, zero-shot tagging and near-duplicate detection, with no training. Production systems pair an open model (OpenCLIP, SigLIP 2) or a multimodal embedding API with a vector index, metadata filters and optional reranking.

## Key concepts
- Dual encoder: an image encoder and a text encoder trained contrastively on image-caption pairs (CLIP: softmax contrastive loss; SigLIP: sigmoid loss). Similarity = dot product of L2-normalised embeddings.
- Query types: text -> image (search), image -> image (visual similarity, reverse search), image -> text (zero-shot tags: compare against prompts like "a photo of a {label}").
- Index: store one vector per image (or per crop/region for object-level search) with metadata (id, path, tags, date). Exact search with numpy up to roughly hundreds of thousands of vectors; ANN (HNSW, IVF-PQ) in FAISS or a vector DB beyond that ([[vector-databases]]).
- Dedup: exact duplicates by file hash; near-duplicates (resized, recompressed, cropped) by embedding similarity above a threshold, or perceptual hashes (pHash) for cheap first pass; group pairs with union-find.
- Same model for index and query: embeddings from different models or versions are not comparable; re-embed everything when you switch.
- Object-level search: crop with a detector ([[open-vocabulary-detection]]) or masks ([[segment-anything-sam2]]) and embed crops, so small items in busy photos are findable.

## When to use / scenarios
- E-commerce: "shop the look" / visual search from a customer photo; similar-product recommendations ([[ecommerce-retail]], [[recommender-systems]]).
- Media and DAM: search archives by description; find reuse of licensed images.
- ML data hygiene: remove duplicates and near-duplicates across train/test splits to prevent leakage ([[data-leakage-and-validation-splits]]); cluster and sample diverse data for labelling.
- Trust and safety: match uploads against a known-bad image set (use the vendor/industry hash-matching services where required; embeddings are a complement).
- Not for: exact identity (a specific person, a specific serial-numbered part) - use face recognition with consent or a fine-tuned metric-learning model ([[metric-learning-and-few-shot]]); reading text in images ([[ocr]]); questions about image content ([[vision-language-models]]); search over document pages ([[late-interaction-retrieval]]).

## Setup & code
Runnable core with numpy: normalise embeddings, brute-force top-k for text-to-image and image-to-image, recall@k, and near-duplicate grouping with union-find. Synthetic vectors stand in for model output.
```python
import numpy as np

rng = np.random.default_rng(0)
D, n_concepts, per = 128, 20, 25
concepts = rng.normal(size=(n_concepts, D))                     # shared semantic directions

def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

labels = np.repeat(np.arange(n_concepts), per)
img = l2n(concepts[labels] + rng.normal(scale=0.9, size=(len(labels), D)))   # "image embeddings"
dup_src = rng.choice(len(img), 30, replace=False)                            # plant near-duplicates
img = np.vstack([img, l2n(img[dup_src] + rng.normal(scale=0.01, size=(30, D)))])
labels = np.r_[labels, labels[dup_src]]
txt = l2n(concepts + rng.normal(scale=0.3, size=concepts.shape))             # "text embeddings" per concept

def top_k(queries, index, k):
    sims = queries @ index.T                       # cosine similarity (inputs are normalised)
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    order = np.take_along_axis(sims, idx, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(idx, order, 1)

# Text -> image search and recall@10 (any of the top 10 has the queried concept).
hits = top_k(txt, img, 10)
recall10 = np.mean([(labels[h] == q).any() for q, h in enumerate(hits)])
precision10 = np.mean(labels[hits] == np.arange(n_concepts)[:, None])
print(f"text->image recall@10={recall10:.2f} precision@10={precision10:.2f}")

# Image -> image: nearest neighbours of one image, excluding itself.
q = 7
nn = [i for i in top_k(img[q:q + 1], img, 6)[0] if i != q][:5]
print("similar to", q, "->", nn, "labels", labels[nn].tolist())

# Near-duplicate groups: pairs above a high threshold, merged with union-find.
parent = list(range(len(img)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
sims = img @ img.T
for i, j in zip(*np.where(np.triu(sims, 1) > 0.95)):   # threshold: calibrate on known duplicate pairs
    parent[find(i)] = find(j)
groups = {}
for i in range(len(img)):
    groups.setdefault(find(i), []).append(i)
dups = [g for g in groups.values() if len(g) > 1]
print(len(dups), "duplicate groups; keep one per group, drop", sum(len(g) - 1 for g in dups))
assert recall10 == 1.0 and len(dups) == 30
```

Real embeddings with OpenCLIP (`pip install open_clip_torch torch pillow`), from the OpenCLIP README; the result feeds the functions above:
<!-- skip-check: downloads model weights, needs torch -->
```python
import numpy as np, torch, open_clip
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

paths = ["a.jpg", "b.jpg", "c.jpg"]
with torch.no_grad():
    ims = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])
    img = model.encode_image(ims); img /= img.norm(dim=-1, keepdim=True)
    q = model.encode_text(tokenizer(["a red dress on a beach"])); q /= q.norm(dim=-1, keepdim=True)
scores = (q @ img.T).squeeze(0).numpy()
print(sorted(zip(scores, paths), reverse=True))
np.save("image_index.npy", img.numpy().astype(np.float32))   # or add to FAISS / a vector DB
```
SigLIP 2 via Transformers (`google/siglip2-base-patch16-224` and larger checkpoints) exposes `get_image_features` / `get_text_features`; its docs note text must be padded with `padding="max_length", max_length=64` as in training. At scale, put vectors in FAISS (`faiss.IndexFlatIP` for exact inner product, HNSW/IVF for ANN) or a vector DB with metadata filtering ([[vector-databases]]).

## Choosing / trade-offs
| Choice | Options | What pushes each way |
|---|---|---|
| Model | OpenCLIP checkpoints, SigLIP 2, multimodal embedding APIs ([[embedding-models]]) | open: privacy, no per-call fee, fine-tunable; API: no GPU, multilingual and video support in some |
| Model size | small ViT-B vs large ViT-L/H/g | larger = better retrieval, slower embedding and bigger vectors |
| Index | numpy exact, FAISS, vector DB | exact under roughly 1e5-1e6 vectors; ANN + filters beyond ([[hnsw-from-scratch]], [[product-quantization-from-scratch]]) |
| Granularity | whole image vs crops/regions | crops find small objects but multiply index size |
| Dedup method | file hash, perceptual hash, embedding threshold, [[locality-sensitive-hashing-from-scratch]] | hash for exact; pHash for cheap near-dups; embeddings for crops/edits, at more compute |
| Ranking | embedding only vs rerank with a VLM or metadata | rerank top-k when precision on the first page matters |

- Licences: OpenAI CLIP code is MIT and OpenCLIP code is MIT-style; trained weights (and their training data) carry their own terms - read each model card ([[model-licenses]]).
- Domain shift: general CLIP models underperform on medical, satellite, fashion detail or industrial parts; fine-tune with contrastive pairs or use a domain model when recall on your eval set is too low.

## Gotchas
- Forgetting to L2-normalise makes dot products favour large-norm vectors; normalise both index and queries.
- Similarity scores are not probabilities and differ by model; calibrate dedup and "no result" thresholds on labelled pairs from your data.
- Text queries work best phrased like captions ("a photo of ..."); very long or negated queries ("not red") are weak.
- CLIP-family models are biased toward what is common on the web and can encode social bias; audit results for people-related queries.
- Duplicates across train/test splits inflate metrics; dedupe before splitting.
- Changing the model, preprocessing (resize/crop) or precision changes vectors; version the embedding pipeline and re-embed on change.
- Center-crop preprocessing drops content at image edges; for wide product shots, pad instead of crop or embed multiple crops.
- Privacy: embeddings of personal photos are personal data; apply the same retention and access controls as the images ([[ai-security-privacy-compliance]]).

## Related
- [[embeddings]] - what embedding spaces are and how similarity works.
- [[embedding-models]] - multimodal embedding APIs and open models.
- [[vector-databases]] - storing and filtering vectors at scale.
- [[hnsw-from-scratch]] - the ANN index most vector stores use.
- [[contrastive-learning-infonce-from-scratch-numpy]] - the CLIP training objective.
- [[open-vocabulary-detection]] - detect regions to embed for object-level search.
- [[segment-anything-sam2]] - masks for clean object crops.
- [[recommender-systems]] - visual similarity as a recommendation signal.
- [[image-classification]] - when a fixed label set and training data exist.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/system-design/search-systems.md - search architecture around the index.

## References
- Radford et al., CLIP: https://arxiv.org/abs/2103.00020 (code: https://github.com/openai/CLIP)
- OpenCLIP: https://github.com/mlfoundations/open_clip
- Tschannen et al., SigLIP 2: https://arxiv.org/abs/2502.14786; Transformers doc: https://huggingface.co/docs/transformers/model_doc/siglip2
- FAISS: https://github.com/facebookresearch/faiss
