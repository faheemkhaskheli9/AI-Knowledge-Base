---
title: Manufacturing and IoT AI
category: scenarios
tags: [manufacturing, iot, predictive-maintenance, visual-inspection, edge, sensors]
use_cases:
  - "detect product defects on a production line with cameras"
  - "predict equipment failure from sensor data"
  - "run AI on edge devices in a factory"
  - "forecast energy or production load"
  - "build a maintenance assistant over manuals and logs"
status: draft
last_verified: 2026-10-03
sources:
  - https://scikit-learn.org/stable/modules/outlier_detection.html
  - https://docs.ultralytics.com/
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
---

# Manufacturing and IoT AI

## Summary
Industrial AI is mostly perception (visual inspection) and time-series
(sensor anomalies, remaining useful life), deployed at the edge under tight
latency and uptime constraints. LLMs help around the edges: maintenance
manuals, work-order text, operator assistants.

## Key concepts
- Labels are scarce: failures are rare, defects are rare. Favour anomaly
  detection, few-shot and synthetic data ([[anomaly-detection]],
  [[data-labeling-and-synthetic-data]]).
- Edge constraints: latency, no/unstable network, limited accelerator
  ([[edge-on-device]], [[quantization]]).
- Domain shift: new lighting, new batch, new sensor firmware silently degrade
  models; monitor ([[mlops-lifecycle]]).
- Cost asymmetry: a missed defect or failure usually costs far more than a
  false alarm; set thresholds from cost, and keep a human or rule fallback.
- Safety: AI must not be the sole control for safety functions; follow
  functional-safety standards (check IEC 61508 / your sector's rules) and,
  in the EU, machinery and AI Act obligations where they apply.

## When to use / scenarios
1. **Visual defect inspection.** Problem: manual inspection misses defects.
   Approach: detection/segmentation model; anomaly-based approach when defect
   examples are few. Read: [[object-detection]], [[segmentation]],
   [[image-classification]], [[video-analytics]].
2. **Predictive maintenance.** Approach: windowed features from vibration,
   temperature, current; GBM or anomaly detector; alert with lead time.
   Read: [[anomaly-detection]], [[time-series-forecasting]],
   [[gradient-boosting-tabular]].
3. **Process quality prediction.** Approach: tabular model on process
   parameters to predict yield; explain with feature importance. Read:
   [[gradient-boosting-tabular]], [[classic-ml-scikit-learn]].
4. **Energy and demand forecasting.** Read: [[time-series-forecasting]].
5. **Worker safety / PPE detection.** Approach: detection on camera feeds;
   privacy and works-council constraints. Read: [[object-detection]],
   [[face-and-pose]], [[video-analytics]].
6. **Reading gauges and labels.** Approach: OCR/VLM on images. Read: [[ocr]],
   [[vision-language-models]].
7. **Maintenance copilot.** Approach: RAG over manuals, past work orders,
   with part numbers cited. Read: [[rag-basics]], [[advanced-rag]],
   [[document-parsing]].
8. **Fleet/telemetry log triage.** Approach: LLM summarises incident logs.
   Read: [[structured-output]], [[security-defensive]] (similar log triage).
9. **Voice-driven shop-floor assistants.** Read: [[speech-to-text]],
   [[voice-agents]].
10. **Supply chain and inventory.** Read: [[ecommerce-retail]],
    [[time-series-forecasting]].

## Setup & code
Reference architecture (flagship: camera-based inspection):

```
camera + trigger -> edge box (GPU/NPU) -> preprocess -> model (INT8/TensorRT)
  -> decision: pass | fail | uncertain -> PLC signal (reject gate)
  -> uncertain/fail images -> review queue -> relabel -> retrain
  -> central: model registry, drift stats, OTA model rollout with rollback
```

```python
from ultralytics import YOLO   # check licence terms (AGPL / enterprise)
model = YOLO("defects.pt")
res = model("part.jpg", conf=0.25)[0]
fail = any(res.names[int(c)] != "ok" for c in res.boxes.cls)
```
Licence note: see [[model-licenses]] before shipping detector code in a
product.

## Choosing / trade-offs
- Edge vs cloud: edge for latency and connectivity; cloud for training and
  fleet analytics ([[gpu-cloud-options]], [[edge-on-device]]).
- Supervised vs anomaly-based vision: supervised is more accurate when
  defects are well-labelled; anomaly methods cope with unseen defects at the
  cost of more false alarms.
- Accuracy vs speed: smaller or quantized models ([[quantization]],
  [[small-language-models]] for text parts).

## Gotchas
- Lighting, lens dirt and camera drift cause most "model" failures; fix
  the imaging rig first.
- Data leakage in time series: split by time or by machine, not randomly.
- Sensor timestamps and clock drift; calibrate and align streams.
- Alert fatigue kills predictive-maintenance projects; track precision of
  alerts and act on lead time.
- Video of workers is personal data in many jurisdictions; check privacy and
  labour rules ([[ai-security-privacy-compliance]]).
- OT networks are sensitive; keep ML systems read-only and segmented.

## Related
- [[experiment-tracking]] - tracking training runs and datasets.
- [[inference-servers-vllm]] - serving LLM parts when on premises.
- [[llm-observability]] - monitoring the language components.

## References
- scikit-learn outlier detection: https://scikit-learn.org/stable/modules/outlier_detection.html
- Ultralytics docs: https://docs.ultralytics.com/
- Regulation (EU) 2024/1689 (AI Act): https://eur-lex.europa.eu/eli/reg/2024/1689/oj
