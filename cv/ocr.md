---
title: OCR (optical character recognition)
category: cv
tags: [ocr, text-recognition, tesseract, easyocr, paddleocr, doctr, document-ai, handwriting]
use_cases:
  - "extract text from scanned documents, receipts and invoices"
  - "read license plates, serial numbers or meter displays from camera images"
  - "digitise multilingual or Arabic/Urdu/CJK documents"
  - "decide between classic OCR and a vision LLM for form extraction"
  - "make scanned PDFs searchable"
status: draft
last_verified: 2026-10-03
sources:
  - https://github.com/JaidedAI/EasyOCR
  - https://github.com/tesseract-ocr/tesseract
  - https://github.com/PaddlePaddle/PaddleOCR
  - https://mindee.github.io/doctr/
---

# OCR (optical character recognition)

## Summary
OCR converts text in images into machine-readable strings, usually with word/line positions. Classic engines (Tesseract, EasyOCR, PaddleOCR, docTR) are fast, cheap and run locally; vision-language models read messy layouts, handwriting and forms with reasoning but cost more and can hallucinate. Many production systems combine them: OCR for text and boxes, an LLM for field extraction.

## Key concepts
- Pipeline: detection (find text regions) then recognition (read each region); some tools add layout analysis and table recognition.
- Output: text plus bounding boxes plus confidence per word/line; boxes enable redaction, highlighting and layout reconstruction.
- Image quality dominates accuracy: resolution (roughly 300 DPI for scans), contrast, skew, noise.
- Language packs/models are per language and script; right-to-left and cursive scripts (Arabic, Urdu) are harder.
- Handwriting (HTR) is a separate, much harder problem than printed text.
- Document AI: layout/table/key-value extraction on top of OCR (see [[document-parsing]]).
- Metrics: character error rate (CER), word error rate (WER), field-level accuracy.

## When to use / scenarios
- Finance/back-office: invoices, receipts, bank statements, KYC documents (extract then validate totals).
- Logistics/industrial: labels, container codes, plates, meter reading (cropped, constrained text: tuned OCR works best).
- Archives, libraries, legal: bulk digitisation; searchable PDFs.
- Use a VLM instead when layout is irregular and you need understanding, not just text ([[vision-language-models]]); use PDF text extraction directly if the PDF already has a text layer.
- Not for: speech ([[speech-to-text]]); CAPTCHAs.

## Setup & code
```bash
pip install easyocr          # pulls PyTorch; downloads recognition models on first use
```
```python
import easyocr

reader = easyocr.Reader(["en"], gpu=False)       # add languages e.g. ["en", "ur"]; see EasyOCR's supported list
for box, text, conf in reader.readtext("receipt.jpg"):
    print(round(conf, 2), text, box)
```
Alternatives: Tesseract (`apt install tesseract-ocr` / Windows installer, then `pip install pytesseract`), PaddleOCR (strong multilingual and layout/table tooling), docTR (`pip install python-doctr`, PyTorch/TF). Check each project's README for the current install and API.

## Choosing / trade-offs
- Tesseract: mature, CPU-only, good on clean scans, needs preprocessing, weaker on photos and scene text.
- EasyOCR / PaddleOCR / docTR: deep-learning, better on scene text and photos; GPU helps; PaddleOCR is a common choice for multilingual and tables.
- Cloud OCR/document AI services: best out-of-the-box accuracy on forms and handwriting, per-page pricing, data leaves your network.
- VLM extraction: flexible and schema-driven, slower and costlier per page, may invent values; verify numbers.
- Hybrid: OCR text + boxes into an LLM with a JSON schema, then validate (sums, date formats, checksums).
- Local vs cloud is mainly a privacy and volume decision ([[ai-security-privacy-compliance]]).

## Gotchas
- Garbage in, garbage out: deskew, denoise, upscale small text, binarise only when it helps.
- Confidence scores are not calibrated; set thresholds empirically and route low-confidence to review.
- Digits confuse easily (0/O, 1/l/I, 5/S); constrain with character whitelists and format validation.
- Reading order of columns, tables and multi-box layouts is not guaranteed.
- Evaluate on your own documents, not benchmark scans; fonts, stamps and photos change results.
- Multi-language pages need the right language set; wrong set causes garbage.
- Personal data in documents: IDs, bank details need access control and retention rules.

## Related
- [[document-parsing]] - layout, tables and full PDF pipelines.
- [[vision-language-models]] - LLM-based reading and extraction.
- [[structured-output]] - schema-constrained extraction after OCR.
- [[document-processing]] - end-to-end scenarios.
- [[image-classification]] - route document types first.

## References
- https://github.com/JaidedAI/EasyOCR
- https://github.com/tesseract-ocr/tesseract
- https://github.com/PaddlePaddle/PaddleOCR
- https://mindee.github.io/doctr/
