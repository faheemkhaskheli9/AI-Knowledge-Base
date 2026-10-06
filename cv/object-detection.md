---
title: Object detection
category: cv
tags: [object-detection, yolo, ultralytics, rt-detr, open-vocabulary, bounding-boxes, tracking]
use_cases:
  - "detect people, vehicles or PPE in camera footage in real time"
  - "count items on shelves or products on a conveyor belt"
  - "train a custom detector on my own labelled images"
  - "detect arbitrary objects from a text prompt without training data"
  - "run a detector on an edge device or Jetson"
status: draft
last_verified: 2026-10-03
sources:
  - https://docs.ultralytics.com/quickstart/
  - https://docs.ultralytics.com/modes/train/
  - https://huggingface.co/docs/transformers/tasks/object_detection
---

# Object detection

## Summary
Object detection finds each object in an image and returns a class label, a bounding box and a confidence score. The practical default is a pretrained YOLO-family model (Ultralytics) fine-tuned on a custom dataset; transformer detectors (RT-DETR, DETR) and open-vocabulary detectors (Grounding DINO, YOLO-World) cover cases with no labelled data.

## Key concepts
- Output: boxes (xyxy or xywh), class, score. Postprocessing: confidence threshold and NMS (some modern models are NMS-free).
- Metrics: IoU, mAP@0.5 and mAP@0.5:0.95, per-class AP; precision/recall at your operating threshold matters more than mAP in production.
- Dataset formats: YOLO txt (class x_center y_center w h, normalised), COCO JSON, Pascal VOC.
- Model sizes (n/s/m/l/x style): trade accuracy for speed and memory.
- Transfer learning: start from COCO-pretrained weights and fine-tune; often a few hundred images per class suffice.
- Open-vocabulary detection: text-prompted ("a hard hat"); lower accuracy and speed than a trained detector but needs no labels.
- Tracking (assign IDs across frames) builds on detection; see [[video-analytics]].

## When to use / scenarios
- Retail: shelf and stock counting; Safety: PPE and intrusion detection; Traffic: vehicle counting; Agriculture: fruit/pest counting; Manufacturing: part presence and defect localisation.
- Need locations or counts: detection. Only an overall label: [[image-classification]]. Pixel-accurate shapes: [[segmentation]].
- No training data and a quick prototype: open-vocabulary detector or a [[vision-language-models]] call, then label with the output ([[data-labeling-and-synthetic-data]]).
- Small objects in large images: tile the image (SAHI-style slicing) or raise input size.

## Setup & code
```bash
pip install -U ultralytics
```
```python
from ultralytics import YOLO

# Inference with a pretrained COCO model (current family per Ultralytics docs: YOLO26)
model = YOLO("yolo26n.pt")
results = model("https://ultralytics.com/images/bus.jpg")
for r in results:
    for b in r.boxes:
        print(r.names[int(b.cls)], float(b.conf), b.xyxy.tolist())

# Fine-tune on a custom dataset described by data.yaml (train/val paths + class names)
# model.train(data="data.yaml", epochs=50, imgsz=640)
# model.export(format="onnx")
```
Model names change with releases; confirm the current family and weights in the Ultralytics docs before pinning (this file verified YOLO26 as current on 2026-10-03).

## Choosing / trade-offs
- Licence: Ultralytics code and weights are AGPL-3.0 with a separate commercial licence; closed-source commercial products must check this. Permissive alternatives exist (e.g. RT-DETR implementations, MMDetection, torchvision detectors); verify each licence.
- Speed vs accuracy: nano/small models for edge and real-time; large for offline accuracy.
- CNN-YOLO vs transformer detectors: YOLO is simpler and faster to deploy; DETR-family is attractive for crowded scenes and NMS-free output.
- Hosted APIs (cloud vision) vs self-hosted: APIs for low volume; self-host for video, privacy and cost.
- Export: ONNX, TensorRT, OpenVINO, CoreML for target hardware ([[edge-on-device]]).

## Gotchas
- Training and deployment images must look alike (resolution, angle, lighting); capture hard negatives (backgrounds without the object).
- Label quality (loose boxes, missed instances) caps mAP; audit labels first.
- Class imbalance and tiny objects need more data or tiling, not just more epochs.
- Default confidence threshold is not your operating point; tune on validation data for your precision/recall need.
- Dataset leakage from adjacent video frames split across train/val.
- Letterbox resizing changes box coordinates; use the library's postprocessing to map back.
- AGPL licensing is a frequent surprise at commercialisation time.

## Related
- [[segmentation]] - masks instead of boxes.
- [[video-analytics]] - tracking, counting, zones.
- [[image-classification]] - whole-image labels.
- [[data-labeling-and-synthetic-data]] - labelling boxes.
- [[edge-on-device]] - export and deployment.
- [[manufacturing-iot]] - inspection scenarios.
- [[hog-descriptor-from-scratch]] - HOG features built by hand, the classic pre-CNN detector descriptor.

## References
- https://docs.ultralytics.com/quickstart/
- https://docs.ultralytics.com/modes/train/
- https://huggingface.co/docs/transformers/tasks/object_detection
