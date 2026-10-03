---
title: Image segmentation (semantic, instance, SAM)
category: cv
tags: [segmentation, sam, sam2, instance-segmentation, semantic-segmentation, masks, ultralytics]
use_cases:
  - "cut out objects or remove backgrounds from product photos"
  - "measure area of lesions, crops or defects pixel by pixel"
  - "interactively label masks with click or box prompts to build a dataset"
  - "segment road scenes or aerial imagery into land-cover classes"
  - "track and segment an object across video frames"
status: draft
last_verified: 2026-10-03
sources:
  - https://github.com/facebookresearch/sam2
  - https://docs.ultralytics.com/tasks/segment/
  - https://huggingface.co/docs/transformers/tasks/semantic_segmentation
---

# Image segmentation (semantic, instance, SAM)

## Summary
Segmentation labels pixels. Semantic segmentation gives each pixel a class; instance segmentation separates individual objects of a class; panoptic combines both. Promptable foundation models such as Meta's Segment Anything (SAM 2) produce masks from a click, box or automatic grid with no task-specific training, and are widely used for labelling and interactive tools.

## Key concepts
- Semantic: class per pixel (road, sky), no object identity. Instance: one mask per object. Panoptic: both.
- Promptable segmentation: SAM/SAM 2 take points, boxes or masks as prompts and return class-agnostic masks; they do not name what they segment.
- Text-driven segmentation: combine an open-vocabulary detector (boxes from a text prompt) with SAM for masks.
- Video: SAM 2 propagates a prompted mask through frames with memory.
- Trained models: U-Net family for medical/industrial semantic work; Mask R-CNN, YOLO-seg, Mask2Former for instance/panoptic.
- Metrics: IoU / mIoU, Dice, mask AP.
- Mask formats: binary arrays, RLE (COCO), polygons.

## When to use / scenarios
- Photography/ecommerce: background removal and product cut-outs. Medicine: organ or lesion volume/area (decision support, regulated). Agriculture and remote sensing: land cover, canopy area. Manufacturing: defect area measurement. Robotics: grasp targeting.
- Labelling accelerator: SAM-assisted annotation in CVAT/Label Studio, then train a fast detector ([[data-labeling-and-synthetic-data]]).
- If boxes are enough, use [[object-detection]] (cheaper to label and run). If only a class per image, [[image-classification]].

## Setup & code
SAM 2 (from its repo; requires Python 3.10+, recent PyTorch; the repo notes Windows users should use WSL):
```bash
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
cd checkpoints && ./download_ckpts.sh
```
```python
import numpy as np, torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",
                                          "./checkpoints/sam2.1_hiera_large.pt"))
image = np.array(Image.open("photo.jpg").convert("RGB"))
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(point_coords=np.array([[500, 375]]),
                                         point_labels=np.array([1]))   # 1 = foreground click
print(masks.shape, scores)
```
Trained instance segmentation with Ultralytics: `YOLO("<seg-weights>.pt")` using a `-seg` model from the docs, then `model.train(data=..., epochs=...)`.

## Choosing / trade-offs
- Zero-shot SAM: no training, great mask quality, but class-agnostic and heavier than small detectors; needs a GPU for interactive speed (smaller SAM 2.1 variants trade accuracy for FPS).
- Trained YOLO-seg/Mask R-CNN: fast, knows your classes, needs polygon labels.
- U-Net/Mask2Former for dense semantic maps (medical, satellite): best with domain data; consider nnU-Net for medical volumes.
- Licences: SAM 2 is released under a permissive licence per its repo; Ultralytics models are AGPL-3.0 (see [[object-detection]]); check before commercial use.

## Gotchas
- SAM masks have no class; pair with a detector or classifier to know what they are.
- Ambiguous prompts return multiple masks (part vs whole); read all scores or request multimask output.
- Thin structures and transparent objects segment poorly; verify on your domain.
- Annotation cost: polygons take far longer than boxes; use SAM pre-labelling and review.
- Resolution: masks computed at model resolution then upscaled; edges blur on large images, so crop or tile.
- Medical use needs clinical validation and regulatory clearance; never present output as diagnosis.
- Class imbalance in dense tasks (tiny lesions): use Dice/focal losses and report per-class IoU.

## Related
- [[object-detection]] - boxes; detector + SAM pipeline.
- [[vision-language-models]] - grounding models that produce boxes from text.
- [[video-analytics]] - tracking segmented objects.
- [[data-labeling-and-synthetic-data]] - SAM-assisted labelling.
- [[pytorch-basics]] - training custom models.

## References
- https://github.com/facebookresearch/sam2
- https://docs.ultralytics.com/tasks/segment/
- https://huggingface.co/docs/transformers/tasks/semantic_segmentation
