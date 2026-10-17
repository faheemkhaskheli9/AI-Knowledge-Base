---
title: Document layout analysis and table extraction
category: cv
tags: [document-ai, layout-analysis, table-extraction, table-structure-recognition, reading-order, docling, table-transformer, pp-structure, doclayout-yolo]
use_cases:
  - "extract tables from scanned or digital PDFs into CSV or DataFrames"
  - "detect titles, paragraphs, figures and tables on document pages"
  - "get the correct reading order of a multi-column page before chunking for RAG"
  - "turn financial statements or lab reports into structured rows"
  - "choose between a layout pipeline and a vision LLM for document extraction"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/docling-project/docling
  - https://docling-project.github.io/docling/
  - https://github.com/microsoft/table-transformer
  - https://huggingface.co/docs/transformers/model_doc/table-transformer
  - https://arxiv.org/abs/2110.00061
  - https://github.com/PaddlePaddle/PaddleOCR
  - https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html
  - https://github.com/opendatalab/DocLayout-YOLO
---

# Document layout analysis and table extraction

## Summary
Layout analysis finds the regions on a page (title, text block, list, table, figure, header/footer) and the order a human reads them; table structure recognition turns a table region into rows, columns and cells. Together with [[ocr]] they convert pages into structured Markdown, JSON or DataFrames, which is what RAG and extraction pipelines need. Open toolkits (Docling, PaddleOCR PP-StructureV3) bundle all three steps; vision-language models can do the same end to end with less control.

## Key concepts
- Layout detection: an object detector over page images with document classes (text, title, list, table, figure, caption, formula, page header/footer). Models: Docling's layout model, PP-DocLayout in PaddleOCR, DocLayout-YOLO, Table Transformer (table detection only).
- Reading order: sort regions into the order a reader follows. Multi-column pages, sidebars and footnotes break naive top-to-bottom sorting. Classic heuristic: recursive XY-cut (split on the widest horizontal or vertical whitespace gap); learned models do better on complex layouts.
- Table structure recognition (TSR): detect rows, columns, header rows and spanning cells inside a table crop, then fill cells with text from OCR or the PDF text layer. Table Transformer (DETR-based, trained on PubTables-1M) and Docling's TableFormer are common.
- Digital vs scanned: digital PDFs have a text layer, so only layout and table structure are needed (text comes from the PDF, no OCR errors); scanned pages also need OCR.
- Outputs: per-element type + box + text + order, exported as Markdown/HTML/JSON; tables as HTML or DataFrames with cell spans.
- Metrics: layout mAP; table TEDS (tree-edit-distance similarity) or GriTS; end-to-end field accuracy for your extraction.

## When to use / scenarios
- RAG over reports, manuals, papers: parse into ordered sections and keep tables intact before chunking ([[document-parsing]], [[rag-basics]]).
- Finance and accounting: balance sheets, invoices with line-item tables, bank statements ([[finance]]).
- Healthcare and labs: lab result tables; insurance forms ([[healthcare]]).
- Legal/regulatory: headings and clause structure for citation ([[legal]]).
- Use a VLM ([[vision-language-models]]) instead when layouts vary wildly, volume is low, and you need semantic extraction directly into a schema ([[structured-output]]); use a page-image retriever ([[late-interaction-retrieval]]) when you only need to find pages, not extract them.
- Not needed when the source is already structured (HTML, DOCX, spreadsheets) - parse those formats directly.

## Setup & code
Runnable core in pure Python + numpy: recursive XY-cut reading order for layout boxes, and rebuilding a table grid from detected cell boxes into CSV.
```python
import csv, io
import numpy as np

def xy_cut(boxes, ids=None, min_gap=8):
    """Reading order by recursive XY-cut. boxes: (N, 4) x1, y1, x2, y2. Returns ids in order."""
    ids = list(range(len(boxes))) if ids is None else ids
    if len(ids) <= 1:
        return ids
    b = boxes[ids]
    for axis in (1, 0):                         # try horizontal cuts (rows) first, then vertical (columns)
        lo, hi = b[:, axis], b[:, axis + 2]
        order = np.argsort(lo)
        reach, groups, cur = hi[order[0]], [], [order[0]]
        for k in order[1:]:
            if lo[k] - reach >= min_gap:        # whitespace gap -> cut here
                groups.append(cur); cur = []
            cur.append(k); reach = max(reach, hi[k])
        groups.append(cur)
        if len(groups) > 1:
            return [i for g in groups for i in xy_cut(boxes, [ids[k] for k in g], min_gap)]
    return [ids[k] for k in np.lexsort((b[:, 0], b[:, 1]))]   # no gap: top-to-bottom, left-to-right

# A page: title, two columns of text, then a full-width table, then a footer.
names = ["title", "colA-1", "colB-1", "colA-2", "colB-2", "table", "footer"]
boxes = np.array([[50, 30, 550, 60], [50, 80, 290, 200], [310, 80, 550, 160],
                  [50, 210, 290, 330], [310, 170, 550, 330], [50, 350, 550, 500],
                  [250, 770, 350, 785]], dtype=float)
order = [names[i] for i in xy_cut(boxes)]
print(order)
assert order == ["title", "colA-1", "colA-2", "colB-1", "colB-2", "table", "footer"]

def cells_to_grid(cells, tol=5):
    """cells: list of (x1, y1, x2, y2, text). Cluster row/column starts, return a 2-D list."""
    def cluster(vals):
        centers = []
        for v in sorted(vals):
            if not centers or v - centers[-1] > tol:
                centers.append(v)
        return centers
    rows, cols = cluster([c[1] for c in cells]), cluster([c[0] for c in cells])
    grid = [["" for _ in cols] for _ in rows]
    for x1, y1, x2, y2, text in cells:
        r = int(np.argmin([abs(y1 - v) for v in rows])); c = int(np.argmin([abs(x1 - v) for v in cols]))
        grid[r][c] = text
    return grid

cells = [(50, 350, 200, 380, "Item"), (201, 351, 380, 380, "Qty"), (381, 350, 550, 380, "Total"),
         (50, 381, 200, 410, "Widget"), (202, 382, 380, 410, "3"), (381, 381, 550, 410, "29.97"),
         (51, 411, 200, 440, "Gadget"), (201, 411, 380, 440, "1"), (382, 412, 550, 440, "15.00")]
grid = cells_to_grid(cells)
buf = io.StringIO(); csv.writer(buf).writerows(grid)
print(buf.getvalue())
assert grid[2] == ["Gadget", "1", "15.00"]
total = sum(float(r[2]) for r in grid[1:])
assert abs(total - 44.97) < 1e-9        # validate extracted numbers against the document's own totals
```

Docling end to end (`pip install docling`; MIT; downloads its layout and TableFormer models on first run), following the Docling README and table-export example:
<!-- skip-check: downloads layout/table models on first run -->
```python
from docling.document_converter import DocumentConverter

conv = DocumentConverter().convert("annual_report.pdf")
doc = conv.document
print(doc.export_to_markdown()[:500])           # ordered headings, paragraphs, tables
for i, table in enumerate(doc.tables):
    df = table.export_to_dataframe(doc=doc)      # pandas DataFrame per table
    df.to_csv(f"table_{i}.csv", index=False)
```
Docling also offers a VLM pipeline (`docling --pipeline vlm --vlm-model granite_docling <pdf>`). PaddleOCR's PP-StructureV3 (Apache 2.0) returns Markdown/JSON with table-cell and text coordinates; PaddleOCR-VL is its compact document VLM. See the PP-StructureV3 docs for the current API.

## Choosing / trade-offs
| Option | Good at | Watch out | Licence (code repo) |
|---|---|---|---|
| Docling | full PDF -> Markdown/JSON, tables to DataFrames, local | CPU speed on large batches; tune OCR settings for scans | MIT |
| PaddleOCR PP-StructureV3 / PaddleOCR-VL | multilingual scans, cell coordinates, formulas | PaddlePaddle runtime | Apache 2.0 |
| Table Transformer (detection + structure) | a focused table step you control | older; you build OCR and assembly yourself | MIT |
| DocLayout-YOLO | fast layout detection | layout only; AGPL-3.0 | AGPL-3.0 |
| VLM page-to-JSON ([[vision-language-models]]) | messy layouts, direct schema extraction | cost per page, hallucinated cells, no boxes unless asked | API terms |
| Cloud document AI services | managed, forms/key-value extraction | per-page cost, data residency | vendor terms |

- Accuracy vs control: pipelines give boxes and provenance for each value (needed for audit and highlighting); VLMs are simpler but harder to verify.
- Throughput: layout + TSR models on GPU for millions of pages; VLMs per page get expensive at that scale ([[cost-and-latency]]).
- Weights may carry a different licence from code; check model cards ([[model-licenses]]).

## Gotchas
- Spanning and multi-row header cells are where TSR fails most; check merged cells and header detection on your documents.
- Tables that cross pages come out as two tables; merge by matching column headers.
- Borderless tables and dense financial tables with indented sub-rows lose hierarchy; keep the indentation level as a column if it carries meaning.
- Reading-order errors (columns interleaved, footnotes in the middle of text) silently corrupt RAG chunks; spot-check multi-column pages.
- Rotated pages and landscape tables: detect orientation before layout.
- Numbers: OCR confuses `,`/`.`, `-`/`(` for negatives, `0`/`O`. Validate sums and totals against the document, as in the code above.
- Headers/footers and page numbers repeat on every page; drop them before chunking.
- Evaluate on your own documents; public benchmarks (PubTables-1M, DocLayNet, OmniDocBench) differ from invoices or scanned forms.

## Related
- [[document-parsing]] - PDF parsing pipelines and tools for RAG.
- [[ocr]] - text recognition for scanned pages and table cells.
- [[vision-language-models]] - end-to-end page reading and extraction.
- [[structured-output]] - schema-constrained extraction after parsing.
- [[object-detection]] - layout detection is object detection on pages.
- [[late-interaction-retrieval]] - retrieve page images without parsing.
- [[document-processing]] - end-to-end document scenarios.
- [[rag-basics]] - consumer of the ordered, chunked output.

## References
- Docling repo and docs: https://github.com/docling-project/docling, https://docling-project.github.io/docling/
- Table Transformer: https://github.com/microsoft/table-transformer and https://huggingface.co/docs/transformers/model_doc/table-transformer
- Smock et al., PubTables-1M: https://arxiv.org/abs/2110.00061
- PaddleOCR (PP-StructureV3, PaddleOCR-VL): https://github.com/PaddlePaddle/PaddleOCR, https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html
- DocLayout-YOLO: https://github.com/opendatalab/DocLayout-YOLO
