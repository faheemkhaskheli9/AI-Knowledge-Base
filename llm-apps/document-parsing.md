---
title: Document parsing (PDF and Office to text for LLMs)
category: llm-apps
tags: [document-parsing, pdf, ocr, docling, unstructured, markdown, tables, layout]
use_cases:
  - "convert thousands of PDF contracts into clean text for a RAG system"
  - "extract tables from financial statements and invoices"
  - "ingest Word, PowerPoint and Excel files into a knowledge base"
  - "handle scanned documents and forms with OCR"
  - "send PDFs straight to a multimodal LLM instead of building a parser"
status: draft
last_verified: 2026-10-03
sources:
  - https://github.com/docling-project/docling
  - https://pymupdf.readthedocs.io/
  - https://platform.claude.com/docs/en/build-with-claude/pdf-support
  - https://docs.unstructured.io/
---

# Document parsing (PDF and Office to text for LLMs)

## Summary
Document parsing turns PDFs, scans, and Office files into clean text or Markdown that preserves reading order, headings and tables. It decides RAG and extraction quality: bad parsing (merged columns, lost tables, headers repeated on every page) cannot be fixed downstream. Pick a parser by document type: digital vs scanned, simple vs layout-heavy, volume and privacy needs.

## Key concepts
- **Digital PDFs** have a text layer: fast extraction with PyMuPDF/pymupdf4llm, pdfplumber, pypdf. **Scanned PDFs/images** need OCR ([[ocr]]): Tesseract, PaddleOCR, docTR, cloud OCR (Textract, Document AI, Azure Document Intelligence).
- **Layout-aware parsers** detect columns, headings, tables, figures, formulas and output Markdown/JSON: Docling (IBM, open source), Unstructured, Marker, MinerU, LlamaParse (hosted), Mistral OCR, Azure Document Intelligence.
- **Office formats**: python-docx, python-pptx, openpyxl/pandas for DOCX/PPTX/XLSX; or Docling/Unstructured/MarkItDown to convert everything to Markdown uniformly.
- **Vision-LLM route**: send the PDF/page images to a multimodal model (Claude accepts PDF input; many models accept page images) to read text, tables and charts directly or to extract structured fields ([[vision-language-models]], [[structured-output]]). Best for complex pages and small volumes; cost scales per page.
- **Output target**: Markdown keeps headings/tables (good for chunking by structure, [[rag-basics]]); keep page numbers and bounding info as metadata for citations.
- **Tables**: serialize as Markdown/HTML/CSV and keep the caption; consider embedding a text summary of each table alongside it.
- **Cleaning**: drop headers/footers/page numbers, fix hyphenation, normalize whitespace, dedupe, detect language.

## When to use / scenarios
- RAG ingestion of manuals, policies, contracts, research papers ([[document-processing]]).
- Finance: statements, 10-Ks, invoices -> tables and key fields.
- Legal/healthcare: scanned records, forms, handwriting (needs strong OCR and human review).
- Mixed enterprise drives: batch convert all formats to Markdown, then chunk.

## Setup & code
Fast path for digital PDFs to Markdown:
```python
# pip install pymupdf4llm
import pymupdf4llm
md = pymupdf4llm.to_markdown("report.pdf")           # headings, tables, reading order
open("report.md", "w", encoding="utf-8").write(md)
```
Layout-aware, handles many formats (OCR optional):
```python
# pip install docling
from docling.document_converter import DocumentConverter
res = DocumentConverter().convert("contract.pdf")     # also .docx, .pptx, .xlsx, .html, images
print(res.document.export_to_markdown())
```
Let Claude read a PDF directly (text + page images; check size/page limits in the docs):
```python
# pip install anthropic
import anthropic, base64
client = anthropic.Anthropic()
pdf = base64.standard_b64encode(open("invoice.pdf", "rb").read()).decode()
r = client.messages.create(model="claude-sonnet-5-5", max_tokens=1000, messages=[{"role": "user", "content": [
    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf}},
    {"type": "text", "text": "Return vendor, invoice number, date and total as JSON."}]}])
print(r.content[0].text)
```

## Choosing / trade-offs
- **Open-source local vs hosted API**: local (Docling, Marker, PyMuPDF) = privacy, no per-page fees, you manage GPUs for OCR/layout models; hosted = best accuracy on hard layouts with zero ops but data leaves your network ([[ai-security-privacy-compliance]]).
- **Speed vs fidelity**: PyMuPDF is milliseconds per page; layout/vision models take seconds per page.
- **Parser vs vision LLM**: LLM wins on messy layouts, charts and forms; parser wins on volume, cost, determinism.
- **Hybrid**: parse fast, route low-confidence or table-heavy pages to a vision model.
- **Markdown vs JSON elements**: Markdown is simple; element JSON (with types and bboxes) enables structure-aware chunking and citations.

## Gotchas
- Multi-column PDFs read in the wrong order with naive extractors; compare against the rendered page.
- Scanned PDFs return empty text from digital extractors; detect (little text per page) and route to OCR.
- Headers/footers repeated on every page pollute chunks and retrieval.
- Tables flattened into run-on text destroy numeric meaning; verify on a sample.
- Handwriting, stamps, low-resolution scans: expect errors; add human review for high-stakes fields.
- Password-protected or malformed PDFs crash batch jobs; wrap each file and log failures.
- Files can contain hidden text with injected instructions ([[prompt-injection]]).
- Keep a source-to-chunk mapping (file, page) to allow citations and re-ingestion.

## Related
- [[rag-basics]] - consumer of parsed text and chunking.
- [[ocr]] - scanned input.
- [[vision-language-models]] - reading pages as images.
- [[structured-output]] - field extraction after parsing.
- [[document-processing]] - end-to-end scenarios.
- [[advanced-rag]] - contextual chunks.

## References
- Docling: https://github.com/docling-project/docling
- PyMuPDF: https://pymupdf.readthedocs.io/
- Unstructured: https://docs.unstructured.io/
- Anthropic PDF support: https://platform.claude.com/docs/en/build-with-claude/pdf-support
