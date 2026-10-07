---
title: IoU, non-maximum suppression and mAP from scratch (object detection post-processing and evaluation)
category: cv
tags: [nms, non-maximum-suppression, iou, map, average-precision, object-detection, yolo, coco, pascal-voc, numpy, from-scratch]
use_cases:
  - "remove duplicate boxes from an object detector's raw output"
  - "choose the NMS IoU threshold for crowded vs sparse scenes"
  - "compute mAP@0.5 / AP75 for a detector on my own labelled data"
  - "understand why a detector's mAP changes with post-processing settings"
status: stable
last_verified: 2026-10-04
sources:
  - https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html
  - https://cocodataset.org/#detection-eval
  - http://host.robots.ox.ac.uk/pascal/VOC/
  - https://arxiv.org/abs/1704.04503
---

# IoU, non-maximum suppression and mAP from scratch (object detection post-processing and evaluation)

## Summary
Object detectors emit many overlapping boxes per object. Non-maximum suppression (NMS) keeps the highest-scoring box and drops the others that overlap it by more than an IoU threshold. Average precision (AP) then scores the remaining ranked boxes against ground truth, and mAP averages AP over classes (and, for COCO, over IoU thresholds 0.5-0.95). This file implements IoU, greedy NMS and all-point-interpolated AP in NumPy, checks the NMS against `torchvision.ops.nms`, and runs them on synthetic detector output. NMS lifts AP50 from 0.82 to 0.98, and the threshold that is best for AP50 (0.5) is not the best for AP75 (0.7).

## Key concepts
- **IoU** = intersection area / union area of two boxes. 1 for identical boxes, 0 for disjoint ones. Used both for suppression and for deciding whether a detection matches a ground-truth box.
- **Greedy NMS.** Sort by score. Take the top box, remove every remaining box with IoU above the threshold, repeat. It runs per class in most detectors (class-aware NMS).
- **Matching for AP.** Go through detections in score order. A detection is a true positive if its best-overlapping ground truth has IoU ≥ the threshold and has not been matched already. A duplicate of an already-matched object is a false positive.
- **AP** is the area under the precision-recall curve after making precision monotone non-increasing. VOC 2010+ integrates at every recall point. COCO samples 101 recall points.
- **COCO mAP** (`AP@[.5:.95]`) averages AP over IoU thresholds 0.50, 0.55, ..., 0.95, so it rewards tight localisation. AP50 only asks whether the box is roughly right.

## When to use / scenarios
- Post-processing for any anchor-based or dense detector (YOLO up to v8/v11, SSD, RetinaNet, Faster R-CNN) before showing boxes or passing them to a tracker.
- Evaluating a detector on your own labelled set when you need to know what the mAP number in a report actually measures.
- Tuning the NMS threshold for the deployment scene: crowds and stacked products need a higher threshold, sparse scenes a lower one.
- Not needed for set-prediction detectors (DETR family, YOLOv10's NMS-free head), which output one box per object by design. See [[object-detection]].
- For production evaluation, use `pycocotools` or `torchmetrics.detection.MeanAveragePrecision` rather than your own AP code, so numbers are comparable with published results.

## Setup & code
NumPy only. The script builds 200 synthetic images, each with 1-5 objects. Each object gets 3-6 jittered duplicate boxes with decreasing scores, and each image gets a few random false positives.

```python
import numpy as np

rng = np.random.default_rng(0)


def iou(box, boxes):
    """IoU of one [x1, y1, x2, y2] box against an (N, 4) array."""
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter)


def nms(boxes, scores, thr=0.5):
    """Greedy NMS: keep the best box, drop everything overlapping it by more than thr, repeat."""
    order, keep = np.argsort(-scores), []
    while len(order):
        i, order = order[0], order[1:]
        keep.append(i)
        order = order[iou(boxes[i], boxes[order]) <= thr]
    return np.array(keep, dtype=int)


def average_precision(dets, gts, thr=0.5):
    """VOC/COCO-style AP for one class. dets: list of (image, score, box); gts: {image: (M, 4)}."""
    dets = sorted(dets, key=lambda d: -d[1])
    used = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tp = np.zeros(len(dets))
    for n, (img, _, box) in enumerate(dets):
        o = iou(box, gts[img])
        j = o.argmax()
        if o[j] >= thr and not used[img][j]:      # each ground truth may be matched once
            tp[n], used[img][j] = 1, True
    n_gt = sum(len(v) for v in gts.values())
    recall = np.cumsum(tp) / n_gt
    precision = np.cumsum(tp) / np.arange(1, len(dets) + 1)
    # all-point interpolation: make precision monotone, then integrate over recall
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return np.sum(np.diff(np.r_[0, recall]) * precision)


# synthetic detector output: 3-6 duplicate boxes around each object plus random false positives
gts, raw = {}, []
for img in range(200):
    xy = rng.uniform(0, 400, (rng.integers(1, 6), 2))
    wh = rng.uniform(30, 120, xy.shape)
    gts[img] = np.c_[xy, xy + wh]
    for g in gts[img]:
        for k in range(rng.integers(3, 7)):
            raw.append((img, rng.uniform(0.5, 1) - 0.08 * k, g + rng.normal(0, 0.06 * (g[2] - g[0]), 4)))
    for _ in range(rng.integers(0, 4)):
        xy = rng.uniform(0, 400, 2)
        raw.append((img, rng.uniform(0, 0.7), np.r_[xy, xy + rng.uniform(30, 120, 2)]))


def apply_nms(dets, thr):
    out = []
    for img in gts:
        d = [x for x in dets if x[0] == img]
        b, s = np.array([x[2] for x in d]), np.array([x[1] for x in d])
        out += [d[i] for i in nms(b, s, thr)]
    return out


print(f"no NMS         {len(raw):5d} boxes  AP50 {average_precision(raw, gts):.3f}  "
      f"AP75 {average_precision(raw, gts, 0.75):.3f}")
for thr in [0.3, 0.5, 0.7, 0.9]:
    kept = apply_nms(raw, thr)
    print(f"NMS iou<={thr:.1f}  {len(kept):5d} boxes  AP50 {average_precision(kept, gts):.3f}  "
          f"AP75 {average_precision(kept, gts, 0.75):.3f}")
```

Output (Python 3.14, NumPy 2.5):
```
no NMS          2890 boxes  AP50 0.819  AP75 0.662
NMS iou<=0.3    838 boxes  AP50 0.974  AP75 0.698
NMS iou<=0.5    893 boxes  AP50 0.984  AP75 0.704
NMS iou<=0.7   1249 boxes  AP50 0.935  AP75 0.749
NMS iou<=0.9   2707 boxes  AP50 0.822  AP75 0.665
```

The kept indices were checked against `torchvision.ops.nms` (torchvision 0.28) on 50 random box sets and matched every time. Without NMS, every duplicate after the first is a false positive, so AP50 sits at 0.82. A threshold of 0.5 removes almost all duplicates and gives the best AP50. AP75 prefers 0.7: a looser threshold keeps some second boxes, and when the top-scoring box is poorly localised, a better-placed duplicate can still match at IoU 0.75. At 0.9 almost nothing is suppressed. At 0.3 a few genuinely overlapping objects get merged, which costs recall.

## Choosing / trade-offs
- **NMS threshold.** 0.45-0.7 is the usual range (Ultralytics YOLO defaults `iou=0.7` at predict time). Crowded scenes with overlapping objects need higher values. Tune it on a validation set against the metric you report.
- **Score threshold** before NMS cuts work. For mAP evaluation keep it very low (around 0.001), because AP needs the full precision-recall curve. For deployment pick it from the operating point you need.
- **Class-aware vs class-agnostic.** Per-class NMS (`torchvision.ops.batched_nms`) keeps a "person" and a "bicycle" box that overlap. Class-agnostic NMS removes cross-class duplicates when the classifier is unsure between similar classes.
- **Soft-NMS** decays the scores of overlapping boxes instead of deleting them, which helps in crowds. **Weighted boxes fusion** averages overlapping boxes, common for ensembles.
- **Speed.** Greedy NMS is O(N²) in the worst case. Use `torchvision.ops.nms` (C++/CUDA) or the exporter's built-in NMS (TensorRT `EfficientNMS`, ONNX `NonMaxSuppression`) in production.
- **Which AP.** Report COCO `AP@[.5:.95]` for comparison with papers, AP50 when only rough location matters (counting, alerting), AP75 when boxes feed measurement or cropping.

## Gotchas
- Box format: `[x1, y1, x2, y2]` vs `[x, y, w, h]` vs centre format. A mismatch gives plausible-looking but wrong IoUs. Convert once at the boundary (`torchvision.ops.box_convert`).
- Pixel convention: old VOC code adds `+1` to widths and heights. Mixing conventions shifts IoU for small boxes.
- AP from different tools differs: VOC 11-point vs all-point vs COCO 101-point interpolation, and COCO ignores "crowd" boxes and splits by object size. Do not compare numbers computed with different tools.
- Evaluating after a high score threshold truncates the precision-recall curve and under-reports AP.
- A ground truth can only be matched once. Code that lets duplicates match the same object reports inflated precision.
- Exported models with NMS inside the graph have fixed `max_output_boxes` and thresholds. Changing them later requires re-export.

## Related
- [[object-detection]] - detector families, training and deployment, including NMS-free models.
- [[hungarian-algorithm-tracking-from-scratch]] - the IoU-based matching that runs after NMS in trackers.
- [[classification-metrics-and-cross-validation-from-scratch]] - precision, recall and PR curves in the classification setting.
- [[video-analytics]] - detection plus tracking pipelines where NMS settings matter.

## References
- torchvision `nms` / `batched_nms`: https://pytorch.org/vision/stable/generated/torchvision.ops.nms.html
- COCO detection evaluation: https://cocodataset.org/#detection-eval
- PASCAL VOC challenge and devkit: http://host.robots.ox.ac.uk/pascal/VOC/
- Bodla et al. (2017), "Soft-NMS - Improving Object Detection With One Line of Code": https://arxiv.org/abs/1704.04503
