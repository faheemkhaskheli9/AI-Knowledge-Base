---
title: Multimodal RAG (page images, tables, charts)
category: llm-apps
tags: [multimodal-rag, visual-rag, colpali, colqwen, multimodal-embeddings, page-images, tables, charts, vlm, rag]
use_cases:
  - "answer questions over PDF reports whose key numbers live in charts and tables"
  - "build search over slide decks and scanned manuals without an OCR-then-chunk pipeline"
  - "decide between parsing to text, page-image embeddings and ColPali-style retrieval for a document archive"
  - "combine a text index and a page-image index in one RAG system"
status: draft
last_verified: 2026-10-15
sources:
  - https://arxiv.org/abs/2407.01449
  - https://github.com/illuin-tech/colpali
  - https://docs.voyageai.com/docs/multimodal-embeddings
  - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html
  - https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2
  - https://arxiv.org/abs/2103.00020
  - https://platform.claude.com/docs/en/build-with-claude/vision
---

# Multimodal RAG (page images, tables, charts)

## Summary
Text-only RAG fails on documents where the answer is in a chart, a table
layout, a diagram or a scanned page, because the parse step loses it before
retrieval ever runs. Multimodal RAG keeps the visual signal: either convert
visuals into good text (parse, extract tables, caption figures), or embed the
page image itself (single-vector multimodal embeddings, or ColPali-style
multi-vector late interaction), then send the retrieved page images to a
vision-language model to answer. Pick by what your documents look like and
what the answer must cite, and measure on your own questions.

## Key concepts
- **Three retrieval routes**:
  1. *Parse to text*: OCR/layout parsing, table extraction to Markdown/HTML,
     a VLM writes a caption or data summary for each figure; then ordinary
     text RAG ([[document-parsing]], [[ocr]], [[rag-basics]]).
  2. *Single-vector page embedding*: one vector per page screenshot (or per
     interleaved text+image item) from a multimodal embedding model.
     Voyage's multimodal models and Cohere Embed v4 accept interleaved
     text and images; Gemini Embedding 2 maps text, images, video, audio and PDFs into one space
     ([[embedding-models]]).
  3. *Multi-vector late interaction*: ColPali/ColQwen encode a page image into
     one vector per patch and score with MaxSim; strongest on visually rich
     pages, heaviest storage ([[late-interaction-retrieval]]).
- **CLIP-style vs document-style embedders**: CLIP/SigLIP dual encoders
  (Radford et al., 2021) match photos to short captions; they are weak on
  text-dense pages. Voyage's docs contrast their single-backbone model with
  CLIP for exactly this reason. Use CLIP for photo/product search, a
  document-trained embedder or ColPali for pages.
- **Unit of retrieval is the page (or region)**, not a 500-token chunk. Store
  page number, document ID, and a rendered image path with every vector, so
  the generator can be shown the exact page and the answer can cite it.
- **Generation reads pixels**: pass the top-k page images (not their OCR text)
  to a VLM ([[vision-language-models]]). Each image costs input tokens, so k
  is usually small (1-5 pages); crop to the region when you know it.
- **Hybrid indexes**: many systems keep a text index (exact IDs, numbers,
  full-text search) and an image index (charts, layout), then fuse ranks with
  RRF ([[reciprocal-rank-fusion-from-scratch]]).
- **ViDoRe** is the visual-document retrieval benchmark introduced with
  ColPali (Faysse et al., 2024); use it to shortlist retrievers, then test on
  your own pages.

## When to use / scenarios
- Finance and consulting: annual reports, earnings decks where revenue
  splits are bar charts ([[finance]]).
- Manufacturing and field service: manuals with exploded diagrams and wiring
  tables ([[manufacturing-iot]]).
- Insurance and public sector: scanned forms, stamped letters, handwritten
  annotations ([[document-processing]]).
- E-commerce: product search by photo plus text uses CLIP-style embeddings
  rather than page retrieval ([[ecommerce-retail]]).
- Not needed when documents are born-digital prose (HTML, Markdown, clean
  PDFs without figures): text RAG is cheaper and cites by character offset.
- Not a substitute for extraction: if you need fields into a database, use
  structured extraction ([[structured-output]], [[document-parsing]]).

## Setup & code
Toy hybrid retrieval in NumPy: a text index built from parsed text misses the
fact that only exists in a chart, the page-image index finds it, and RRF over
both returns the right page. Random vectors stand in for real embeddings.

```python
import numpy as np

rng = np.random.default_rng(0)
D, N_PAGES = 64, 40

def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

topics = unit(rng.normal(size=(N_PAGES, D)))           # what each page is "about"
# Parsed-text vectors: good for most pages, but page 17's key fact lives in a chart
# the parser dropped, so its text vector is mostly noise.
text_vecs = unit(topics + 0.3 * rng.normal(size=topics.shape))
text_vecs[17] = unit(rng.normal(size=D))
# Page-image vectors (multimodal embedder): see the chart, noisier on dense text.
image_vecs = unit(topics + 0.6 * rng.normal(size=topics.shape))

def ranks(index, q):
    return list(np.argsort(-(index @ q)))

def rrf(rankings, k=60, top=5):
    score = {}
    for r in rankings:
        for pos, doc in enumerate(r):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=score.get, reverse=True)[:top]

hits = {"text": 0, "image": 0, "hybrid": 0}
queries = list(range(N_PAGES))
for target in queries:
    q = unit(topics[target] + 0.4 * rng.normal(size=D))
    rt, ri = ranks(text_vecs, q), ranks(image_vecs, q)
    hits["text"] += target in rt[:5]
    hits["image"] += target in ri[:5]
    hits["hybrid"] += target in rrf([rt, ri])
    if target == 17:
        print("chart-only page 17 -> text rank", rt.index(17) + 1,
              "| image rank", ri.index(17) + 1, "| in hybrid top-5:", 17 in rrf([rt, ri]))
for name, h in hits.items():
    print(f"recall@5 {name:6s} {h / len(queries):.2f}")
assert 17 in ranks(image_vecs, unit(topics[17]))[:5]
```

Generation step: send the retrieved page images to a VLM and ask for an answer
that cites the page. Model ID from [[anthropic-claude]].
<!-- skip-check: needs an API key and page images on disk -->
```python
# pip install anthropic
import base64, anthropic
client = anthropic.Anthropic()

def answer_from_pages(question: str, pages: list[tuple[str, int]]) -> str:
    content = []
    for path, page_no in pages:                       # top-k retrieved page renders (PNG)
        data = base64.standard_b64encode(open(path, "rb").read()).decode()
        content.append({"type": "text", "text": f"Page {page_no}:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
    content.append({"type": "text", "text":
        f"{question}\nAnswer only from these pages and cite the page number. "
        "If the pages do not contain the answer, say so."})
    r = client.messages.create(model="claude-sonnet-5-5", max_tokens=600,
                               messages=[{"role": "user", "content": content}])
    return r.content[0].text
```
Retriever options: `colpali` repo checkpoints (ColPali, ColQwen, ColSmol) for
late interaction; the `voyageai` client's `multimodal_embed()`; Cohere Embed v4
`inputs` with text and image parts; Gemini Embedding 2 for mixed media.

## Choosing / trade-offs
| Route | Good at | Costs / limits |
|---|---|---|
| Parse to text + caption figures | exact strings, citations by offset, cheap storage, any text vector DB | parsing quality caps recall; captions miss details; extra VLM calls at index time |
| Single-vector page embedding | simple, one vector per page, works in any vector DB | page-level granularity blurs dense pages; hosted API sends documents out |
| Multi-vector (ColPali/ColQwen) | best on charts, tables, layout | many vectors per page, needs multivector support (Qdrant, Vespa, Weaviate) or a reranking stage |
| Hybrid text + image | covers both failure modes | two indexes to keep in sync, fusion tuning |

- **Privacy**: open ColPali/ColQwen checkpoints run locally; hosted
  embedders and VLMs see the page images ([[ai-security-privacy-compliance]]).
- **Cost at query time** is dominated by the VLM reading images; fewer,
  cropped pages beat many full pages.
- **Index-time cost**: page rendering plus embedding is usually cheaper than
  per-figure captioning with a VLM, which can be the most expensive step.

## Gotchas
- Render resolution matters: too low and small table text is unreadable to
  both the embedder and the VLM; check each model's pixel limits.
- Page-level retrieval returns a whole page; questions spanning pages (a table
  continued over two pages) need neighbour pages added.
- VLMs misread numbers in charts with confidence; cross-check against the
  parsed table when one exists, and keep a human in the loop for high stakes.
- Text in images is untrusted input and can carry prompt injection
  ([[prompt-injection]]).
- Evaluate retrieval with page-level recall@k on real questions whose answers
  are in visuals; a text-only golden set hides exactly the failures this
  fixes ([[llm-evaluation]]).
- Duplicate pages (repeated headers, cover pages) crowd the top-k; dedupe or
  filter by metadata.

## Related
- [[late-interaction-retrieval]] - ColPali/ColBERT MaxSim scoring and storage maths.
- [[rag-basics]] - the text pipeline this extends.
- [[advanced-rag]] - hybrid search, reranking and fusion.
- [[chunking-strategies]] - the text-side alternative for parsed pages.
- [[document-parsing]] - parse-to-text route.
- [[ocr]] - text from scans.
- [[vision-language-models]] - the generator that reads page images.
- [[multimodal-models]] - how image encoders and VLMs work.
- [[embedding-models]] - multimodal embedding options.
- [[vector-databases]] - multivector support differs by engine.
- [[reciprocal-rank-fusion-from-scratch]] - fusing text and image rankings.
- [[document-processing]] - document archive scenarios.

## References
- Faysse et al., ColPali: Efficient Document Retrieval with Vision Language Models (ViDoRe benchmark): https://arxiv.org/abs/2407.01449
- ColPali / ColQwen code and model list: https://github.com/illuin-tech/colpali
- Voyage multimodal embeddings: https://docs.voyageai.com/docs/multimodal-embeddings
- Cohere Embed v4 request format (`inputs` with text and image parts), Amazon Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html
- Gemini Embedding 2 model page: https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2
- Radford et al., CLIP: https://arxiv.org/abs/2103.00020
- Claude vision: https://platform.claude.com/docs/en/build-with-claude/vision
