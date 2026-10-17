---
title: Open-vocabulary detection (Grounding DINO, OWLv2, YOLO-World)
category: cv
tags: [open-vocabulary, zero-shot-detection, grounding-dino, owl-vit, owlv2, yolo-world, yoloe, llmdet, auto-labelling, transformers, ultralytics]
use_cases:
  - "detect objects described in plain text without collecting training data"
  - "auto-label a new image dataset with boxes, then review and train a fast detector"
  - "prototype a detector for a new product category in an afternoon"
  - "find every image in a folder that contains 'a forklift' and where it is"
  - "run a text-prompted detector at real-time speed on video"
status: draft
last_verified: 2026-10-15
sources:
  - https://huggingface.co/docs/transformers/tasks/zero_shot_object_detection
  - https://huggingface.co/docs/transformers/model_doc/grounding-dino
  - https://huggingface.co/docs/transformers/model_doc/owlv2
  - https://github.com/IDEA-Research/GroundingDINO
  - https://github.com/AILab-CVC/YOLO-World
  - https://docs.ultralytics.com/models/yolo-world/
  - https://docs.ultralytics.com/models/yoloe/
  - https://arxiv.org/abs/2303.05499
  - https://arxiv.org/abs/2306.09683
  - https://arxiv.org/abs/2401.17270
---

# Open-vocabulary detection (Grounding DINO, OWLv2, YOLO-World)

## Summary
Open-vocabulary (zero-shot) detectors take an image plus a list of text prompts ("hard hat", "forklift") and return boxes, scores and the matching prompt, with no training on those classes. They are the fastest way to prototype detection and to pre-label datasets; for a fixed class list in production, a detector fine-tuned on reviewed labels is usually faster and more accurate. Main families: Grounding DINO and its successors (strong accuracy, slower), OWL-ViT/OWLv2 (CLIP-style, also supports image-as-query), and YOLO-World/YOLOE (real-time, Ultralytics API).

## Key concepts
- How they work: an image encoder proposes regions; a text encoder embeds each prompt; each region is scored by similarity to each prompt embedding (CLIP-style alignment). A threshold turns scores into detections.
- Prompts are the "class list". Wording matters: "a red fire extinguisher" vs "extinguisher" gives different recall and false positives. Grounding DINO in Transformers takes a list of phrases (the raw model takes one lowercase string with phrases separated by periods, e.g. `"a cat. a dog."`).
- Two thresholds in Grounding DINO: box `threshold` (is there an object) and `text_threshold` (which prompt tokens it matches).
- Image-guided detection: OWL-ViT/OWLv2 can take an example crop instead of text ("find more of this").
- Real-time variants (YOLO-World, YOLOE) embed the prompt list once (`set_classes`) and then run like a normal YOLO; the vocabulary is fixed until you call it again.
- Output is boxes only; add [[segment-anything-sam2]] for masks (Grounded-SAM pattern), or use SAM 3, which takes text prompts directly.
- Auto-labelling loop: detect with prompts, filter by score and NMS, export to YOLO/COCO format, human review, train a small closed-set detector ([[object-detection]]).

## When to use / scenarios
- Cold start with no labels: retail ("detect the competitor's product on shelves"), safety ("person without helmet" style prompts, verified by review), logistics ("pallet", "forklift").
- Dataset bootstrapping: pre-label thousands of images, then correct in CVAT/Label Studio ([[data-labeling-and-synthetic-data]]).
- Long-tail or changing classes (search a photo archive for arbitrary objects) where retraining per class is impractical.
- Not for: fixed classes at high FPS on edge devices once you have labels (train a closed-set YOLO/RT-DETR); fine-grained identity ("this exact SKU", "this person") - use [[image-retrieval-clip-search]] or face models ([[face-and-pose]]); reading or reasoning about the scene - use [[vision-language-models]].

## Setup & code
Core idea, runnable with numpy only: score region embeddings against prompt embeddings, threshold, per-class NMS, and write YOLO-format labels for review.
```python
import numpy as np

rng = np.random.default_rng(0)
prompts = ["hard hat", "safety vest", "forklift"]
D = 64
text_emb = rng.normal(size=(len(prompts), D))
text_emb /= np.linalg.norm(text_emb, axis=1, keepdims=True)

# Fake region proposals: boxes (x1, y1, x2, y2) in pixels and their image-side embeddings.
boxes = np.array([[100, 50, 180, 120], [105, 55, 185, 125], [400, 300, 620, 470],
                  [300, 100, 360, 260], [10, 10, 40, 40]], dtype=float)
truth = [0, 0, 2, 1, None]                    # which prompt each region really shows
reg_emb = rng.normal(scale=0.25, size=(len(boxes), D))
for i, t in enumerate(truth):
    if t is not None:
        reg_emb[i] += text_emb[t] * 3
reg_emb /= np.linalg.norm(reg_emb, axis=1, keepdims=True)

def iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:, :2]).T
    x2, y2 = np.minimum(a[2:], b[:, 2:]).T
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    return inter / (area(a) + area(b) - inter)

def nms(bxs, scores, thr=0.5):
    order, keep = list(np.argsort(-scores)), []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(bxs[i], bxs[j:j + 1])[0] < thr]
    return keep

sim = reg_emb @ text_emb.T                     # cosine similarity, regions x prompts
cls, score = sim.argmax(1), sim.max(1)
dets = []
for c in range(len(prompts)):
    idx = np.where((cls == c) & (score > 0.5))[0]   # threshold: tune per prompt on a labelled sample
    for k in nms(boxes[idx], score[idx]):
        dets.append((c, score[idx[k]], boxes[idx[k]]))

W, H = 640, 480
for c, s, (x1, y1, x2, y2) in dets:   # YOLO txt: class x_center y_center w h (normalised)
    print(f"{c} {(x1 + x2) / 2 / W:.4f} {(y1 + y2) / 2 / H:.4f} {(x2 - x1) / W:.4f} {(y2 - y1) / H:.4f}"
          f"  # {prompts[c]} {s:.2f}")
assert {prompts[c] for c, _, _ in dets} == set(prompts) and len(dets) == 3
```

Grounding DINO via Transformers (`pip install transformers torch pillow`), from the Transformers model doc:
<!-- skip-check: downloads model weights from the Hugging Face Hub -->
```python
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, device_map="auto")

image = Image.open("site.jpg").convert("RGB")
text_labels = [["a hard hat", "a safety vest", "a forklift"]]
inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
result = processor.post_process_grounded_object_detection(
    outputs, inputs.input_ids, threshold=0.4, text_threshold=0.3,
    target_sizes=[image.size[::-1]])[0]
for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
    print(label, round(score.item(), 3), [round(x, 1) for x in box.tolist()])
```
The generic `pipeline(task="zero-shot-object-detection", model=...)` works with any Hub checkpoint tagged for the task (the Transformers guide uses `iSEE-Laboratory/llmdet_large`; OWLv2 is `google/owlv2-base-patch16-ensemble`) and takes `candidate_labels=[...]`, `threshold=...`.

Real-time with Ultralytics YOLO-World (`pip install -U ultralytics`); YOLOE uses the same `set_classes()` API and adds masks:
<!-- skip-check: downloads weights and a CLIP text encoder -->
```python
from ultralytics import YOLO

model = YOLO("yolov8s-world.pt")
model.set_classes(["hard hat", "safety vest", "forklift"])   # embeds prompts once
model.save("site_world.pt")                                  # reuse without re-embedding
for r in model.predict("site.jpg", conf=0.25):
    print(r.boxes.cls.tolist(), r.boxes.conf.tolist())
```

## Choosing / trade-offs
| Option | Strength | Weakness | Code licence (repo) |
|---|---|---|---|
| Grounding DINO (and successors such as LLMDet in Transformers) | best zero-shot accuracy here, phrase grounding | slower; needs a GPU for interactive use | Apache-2.0 (IDEA-Research/GroundingDINO, iSEE-Laboratory/LLMDet) |
| OWL-ViT / OWLv2 | image-as-query, simple CLIP-style API | lower recall on small objects | Apache-2.0 (google-research/scenic) |
| YOLO-World / YOLOE (Ultralytics) | real-time, export to ONNX/TensorRT, fine-tunable | fixed vocabulary per `set_classes`; licence | original YOLO-World repo GPL-3.0; Ultralytics AGPL-3.0 or commercial |
| VLM with box output ([[vision-language-models]]) | reasons about the prompt ("the broken one") | slow, costly per image, box precision varies | API terms |

- Weights can carry a different licence from the code; read each model card before commercial use ([[model-licenses]]).
- Auto-label then train a closed-set model when the class list is stable: cheaper inference and better accuracy on your domain.
- Keep the zero-shot model when classes change weekly or the long tail matters more than speed.

## Gotchas
- Scores are not calibrated across prompts: one global threshold gives many false positives for some prompts and misses for others. Tune per prompt on 50-100 labelled images.
- Synonyms and plurals compete: "car" and "vehicle" in one list split detections; dedupe prompts or merge classes after NMS.
- Negation does not work ("person without helmet" is not reliable); detect "person" and "helmet" separately and apply the rule in code.
- Long prompt lists slow real-time models and raise confusion; the Ultralytics YOLOE docs note per-frame cost grows with the prompt set.
- First `set_classes()` call in Ultralytics downloads a text encoder; pre-fetch it for offline deployments.
- Auto-labels inherit the model's bias (misses small or unusual instances); always review a sample and track label precision/recall before training on them ([[label-noise-and-data-cleaning]]).
- Grounding DINO `labels` are the matched text spans, not indices; map them back to your class list explicitly.

## Related
- [[object-detection]] - closed-set detectors to train on the auto-labels.
- [[segment-anything-sam2]] - turn detected boxes into masks (Grounded-SAM); SAM 3 takes text prompts.
- [[image-retrieval-clip-search]] - same CLIP-style embeddings, used for search instead of localisation.
- [[vision-language-models]] - prompts that need reasoning, not just a noun.
- [[data-labeling-and-synthetic-data]] - review and correct pre-labels.
- [[nms-iou-and-map-from-scratch]] - IoU, NMS and mAP used to filter and evaluate detections.
- [[contrastive-learning-infonce-from-scratch-numpy]] - the image-text alignment objective behind these models.
- [[video-analytics]] - tracking open-vocabulary detections across frames.

## References
- Transformers zero-shot object detection guide: https://huggingface.co/docs/transformers/tasks/zero_shot_object_detection
- Transformers Grounding DINO doc: https://huggingface.co/docs/transformers/model_doc/grounding-dino
- Transformers OWLv2 doc: https://huggingface.co/docs/transformers/model_doc/owlv2
- Liu et al., Grounding DINO: https://arxiv.org/abs/2303.05499 (code: https://github.com/IDEA-Research/GroundingDINO)
- Minderer et al., Scaling Open-Vocabulary Object Detection (OWLv2): https://arxiv.org/abs/2306.09683
- Cheng et al., YOLO-World: https://arxiv.org/abs/2401.17270 (code: https://github.com/AILab-CVC/YOLO-World)
- Ultralytics YOLO-World and YOLOE docs: https://docs.ultralytics.com/models/yolo-world/, https://docs.ultralytics.com/models/yoloe/
