---
title: Segment Anything (SAM 2, SAM 3) for images and video
category: cv
tags: [sam, sam2, sam3, segment-anything, promptable-segmentation, video-segmentation, auto-labelling, grounded-sam, masks, transformers]
use_cases:
  - "click on an object and get a pixel-accurate mask in my labelling tool"
  - "track and segment an object through a video from one click"
  - "auto-label instance masks for a dataset by combining a text detector with SAM"
  - "segment every object matching a text phrase in images or video"
  - "generate all masks in an image to build cut-outs or region proposals"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/facebookresearch/sam2
  - https://github.com/facebookresearch/sam2/blob/main/LICENSE
  - https://github.com/facebookresearch/sam3
  - https://github.com/facebookresearch/sam3/blob/main/LICENSE
  - https://arxiv.org/abs/2408.00714
  - https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
  - https://huggingface.co/docs/transformers/model_doc/sam2
  - https://huggingface.co/docs/transformers/model_doc/sam3
---

# Segment Anything (SAM 2, SAM 3) for images and video

## Summary
Meta's Segment Anything models return masks from prompts. SAM 2 takes points, boxes or masks on an image or a video frame and propagates the mask through the video with a memory module; its masks are class-agnostic. SAM 3 (Nov 2025) adds "promptable concept segmentation": a short noun phrase or image exemplar returns masks and identities for every matching instance, so it covers what used to need a text detector plus SAM. Use them for interactive labelling, auto-labelling and cut-outs; train a small task model when you need fixed classes at high speed.

## Key concepts
- Prompts: positive/negative points (label 1/0), a box, or a previous mask. One click is often ambiguous (part vs whole), so the model can return several masks with predicted-IoU scores.
- Image encoder runs once per image (`set_image`); each new prompt only runs the light prompt encoder and mask decoder, which is why clicking feels interactive.
- Video (SAM 2): `init_state` on a frame folder or video, add prompts on any frame with `add_new_points_or_box`, then `propagate_in_video` yields per-frame masks per object ID ("masklets"). Add corrective clicks on frames where it drifts.
- Automatic mask generation: a grid of point prompts over the image, then filtering by predicted IoU and stability, then de-duplicating overlapping masks.
- SAM 2 checkpoints: SAM 2.1 `hiera_tiny`, `hiera_small`, `hiera_base_plus`, `hiera_large` (larger = better masks, slower).
- SAM 3: text and/or image-exemplar prompts, returns all matching instances (masks, boxes, scores) on images, and tracks them in video. Checkpoints are gated on Hugging Face (request access, then `hf auth login`); the repo also publishes SAM 3.1 checkpoints.
- Grounded-SAM pattern (with SAM 2): text -> boxes from [[open-vocabulary-detection]] -> box prompts to SAM -> masks with class names.

## When to use / scenarios
- Annotation: SAM-assisted polygon/mask labelling in CVAT or Label Studio cuts labelling time versus hand polygons ([[data-labeling-and-synthetic-data]]).
- E-commerce and creative: background removal and product cut-outs; masks for inpainting ([[image-editing-and-controlnet]]).
- Video: rotoscoping, sports and wildlife tracking, robotics object masks.
- Measurement: area of defects, crops, wounds (with domain validation) - see [[segmentation]] for trained alternatives.
- Not for: fixed-class instance segmentation on an edge device at high FPS (train YOLO-seg or Mask R-CNN on SAM-generated labels); dense semantic maps of every pixel (U-Net/Mask2Former, [[segmentation]]); naming objects with SAM 2 alone (it has no classes).

## Setup & code
Runnable core with numpy: the bookkeeping around SAM output - mask to box, mask IoU, COCO-style RLE, and de-duplicating an automatic mask set by score.
```python
import numpy as np

def mask_to_box(m):
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

def mask_iou(a, b):
    return np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)

def rle_encode(m):
    """Uncompressed COCO RLE: column-major run lengths starting with a zero run."""
    flat = m.T.ravel().astype(np.uint8)
    change = np.flatnonzero(np.diff(np.r_[0, flat, 1 - flat[-1]]))   # run boundaries
    counts = np.diff(np.r_[0, change]).tolist()
    return {"size": list(m.shape), "counts": counts}

def rle_decode(rle):
    h, w = rle["size"]
    vals = np.concatenate([np.full(c, i % 2, np.uint8) for i, c in enumerate(rle["counts"])])
    return vals[: h * w].reshape(w, h).T.astype(bool)

def dedupe(masks, scores, iou_thr=0.7, min_score=0.8):
    """Keep high-score masks, dropping ones that mostly overlap a better mask (mask NMS)."""
    keep = []
    for i in np.argsort(-np.asarray(scores)):
        if scores[i] >= min_score and all(mask_iou(masks[i], masks[k]) < iou_thr for k in keep):
            keep.append(int(i))
    return keep

H, W = 60, 80
yy, xx = np.mgrid[:H, :W]
whole = (yy - 30) ** 2 + (xx - 25) ** 2 < 15 ** 2                    # an object
near_copy = (yy - 31) ** 2 + (xx - 25) ** 2 < 15 ** 2                 # duplicate from another grid point
part = whole & (yy < 25)                                              # a part of the object
other = (yy > 10) & (yy < 40) & (xx > 50) & (xx < 75)                 # a second object
masks, scores = [whole, near_copy, part, other, other & (xx > 70)], [0.95, 0.93, 0.88, 0.97, 0.6]

keep = dedupe(masks, scores)
print("kept", keep, [mask_to_box(masks[k]) for k in keep])
rle = rle_encode(whole)
assert np.array_equal(rle_decode(rle), whole)
assert keep == [3, 0, 2]          # near-duplicate and low-score masks dropped; the part survives
```
The part mask survives de-duplication because its IoU with the whole object is low; decide in your pipeline whether parts are wanted.

SAM 2 install and image prompting are shown in [[segmentation]]. Video tracking, per the SAM 2 README (Python 3.10+, PyTorch, a CUDA GPU for usable speed; `git clone https://github.com/facebookresearch/sam2 && cd sam2 && pip install -e .`, then `checkpoints/download_ckpts.sh`):
<!-- skip-check: needs torch, a GPU and downloaded SAM 2 checkpoints -->
```python
import numpy as np, torch
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml",
                                       "./checkpoints/sam2.1_hiera_small.pt")
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state("frames/")          # folder of JPEG frames 00000.jpg, 00001.jpg, ...
    predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1,
                                    box=np.array([120, 80, 340, 400], dtype=np.float32))
    masks = {}
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        masks[frame_idx] = (mask_logits[0] > 0).cpu().numpy()   # object 1's mask on this frame
print(len(masks), "frames segmented")
```

SAM 3 with a text prompt, via Transformers (gated checkpoint; accept the licence on the Hub first):
<!-- skip-check: gated model download, needs torch -->
```python
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
processor = Sam3Processor.from_pretrained("facebook/sam3")
image = Image.open("shelf.jpg").convert("RGB")
inputs = processor(images=image, text="cereal box", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)
res = processor.post_process_instance_segmentation(
    outputs, threshold=0.5, mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist())[0]
print(len(res["masks"]), "instances", res["boxes"][:3], res["scores"][:3])
```

## Choosing / trade-offs
| Need | Pick |
|---|---|
| Interactive clicks/boxes, video tracking, permissive licence | SAM 2.1 (Apache 2.0 for code and checkpoints per its README) |
| "All instances of X" from text, images or video | SAM 3 (SAM License, custom Meta terms; gated checkpoints) |
| Text prompts but want Apache-licensed parts | Grounding DINO/OWLv2 boxes + SAM 2 ([[open-vocabulary-detection]]) |
| Fixed classes, real-time, edge | Train YOLO-seg/Mask R-CNN on SAM-made labels ([[segmentation]], [[edge-on-device]]) |

- Model size: `tiny`/`small` for interactive tools on modest GPUs; `large` for offline auto-labelling where quality matters more than time.
- SAM 3's custom licence has its own conditions (including trade-control terms); read it before shipping a product. SAM 2's Apache 2.0 is simpler.
- Also available through Ultralytics and Transformers wrappers; the Transformers port avoids building the repo's CUDA extension but check feature parity (video options) per release.

## Gotchas
- SAM 2 masks have no labels; keep the prompt's class (from the detector or the annotator) attached to each mask.
- Single clicks are ambiguous; request multiple masks and pick by score, or add a negative click.
- Video drift and ID swaps after occlusion or when objects leave and re-enter; add correction prompts and check a few frames per object.
- Memory: long videos at full resolution exhaust GPU memory; downscale frames, process in chunks, or offload state to CPU.
- Automatic mask generation returns parts and wholes, shadows and background segments; filter by area, score and IoU (see code) for your use.
- Masks are produced at model resolution and upsampled; thin structures (cables, hair) and transparent objects are weak - crop and re-run on a zoomed region.
- SAM 3 checkpoints need a Hub token in CI and on servers; build that into deployment.
- Medical or safety-critical measurement needs validation on your data; never present a SAM mask as a diagnosis.

## Related
- [[segmentation]] - semantic/instance/panoptic overview, SAM 2 image install and trained alternatives.
- [[open-vocabulary-detection]] - text-prompted boxes to feed SAM 2 (Grounded-SAM).
- [[image-editing-and-controlnet]] - masks as inpainting input.
- [[video-analytics]] - tracking and counting across frames.
- [[data-labeling-and-synthetic-data]] - SAM-assisted annotation workflows.
- [[object-detection]] - when boxes are enough.
- [[model-licenses]] - Apache 2.0 vs custom model licences.

## References
- SAM 2 repo, README and release notes: https://github.com/facebookresearch/sam2 (licence: https://github.com/facebookresearch/sam2/blob/main/LICENSE)
- Ravi et al., SAM 2: Segment Anything in Images and Videos: https://arxiv.org/abs/2408.00714
- SAM 3 repo and SAM License: https://github.com/facebookresearch/sam3, https://github.com/facebookresearch/sam3/blob/main/LICENSE
- SAM 3: Segment Anything with Concepts: https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
- Transformers SAM 2 and SAM 3 docs: https://huggingface.co/docs/transformers/model_doc/sam2, https://huggingface.co/docs/transformers/model_doc/sam3
