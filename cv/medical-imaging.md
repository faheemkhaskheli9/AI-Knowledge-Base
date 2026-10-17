---
title: Medical imaging AI (X-ray, CT, MRI segmentation and classification)
category: cv
tags: [medical-imaging, radiology, ct, mri, x-ray, dicom, nifti, monai, nnu-net, totalsegmentator, medsam, dice, hipaa, fda]
use_cases:
  - "segment organs or tumours in CT or MRI volumes and report their volume in millilitres"
  - "classify chest X-rays for triage or worklist prioritisation"
  - "train a 3D segmentation model on our own labelled scans with MONAI or nnU-Net"
  - "de-identify DICOM studies before sending them to a model or vendor"
  - "plan validation and regulatory steps for an AI radiology tool"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/Project-MONAI/MONAI
  - https://github.com/MIC-DKFZ/nnUNet
  - https://github.com/wasserth/TotalSegmentator
  - https://github.com/bowang-lab/MedSAM
  - https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
  - https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices
  - https://ropesgray.com/en/insights/alerts/2024/12/fda-finalizes-guidance-on-predetermined-change-control-plans-for-ai-enabled-device
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
---

# Medical imaging AI (X-ray, CT, MRI segmentation and classification)

## Summary
Medical imaging AI segments anatomy and lesions in 3D CT/MRI volumes, classifies 2D radiographs, and measures sizes and volumes. The data differs from photos: images arrive as DICOM series with physical voxel spacing, CT intensities are calibrated Hounsfield units, volumes are too large for GPU memory in one pass, and labels are scarce and expensive. The usual open stack is MONAI (PyTorch domain library, Apache-2.0), nnU-Net (self-configuring segmentation baseline, Apache-2.0) and ready-made models such as TotalSegmentator. Anything that informs diagnosis or treatment is regulated as a medical device, and the images are protected health data.

## Key concepts
- **Formats.** DICOM (one file per slice plus metadata, including patient identifiers) from the scanner/PACS; NIfTI (`.nii.gz`, one file per volume) for research pipelines. Keep the affine/voxel spacing; a mask without spacing cannot give a volume.
- **Intensity.** CT is in Hounsfield units (air about -1000, water 0, bone high). Clip to a task window (soft tissue, lung, bone) and rescale. MRI has no absolute scale: normalise per scan (z-score within the body).
- **Spacing and orientation.** Resample to a common spacing and orientation (for example RAS) before training; anisotropic scans (thick slices) need care.
- **Patch-based 3D training and sliding-window inference.** Train on random 3D crops (oversampling foreground), infer by tiling the volume with overlap and blending. MONAI provides `sliding_window_inference`.
- **Architectures.** U-Net variants dominate segmentation; nnU-Net derives preprocessing, patch size and network configuration from a dataset "fingerprint" and is the baseline to beat. Classification uses 2D CNN/ViT backbones on radiographs or 3D CNNs on volumes.
- **Foundation and pre-trained models.** TotalSegmentator segments 117 classes in CT (`total` task) and 50 in MR (`total_mr`); MedSAM adapts Segment Anything to medical images with box prompts (Nature Communications 2024).
- **Metrics.** Dice and IoU for overlap, Hausdorff distance (often HD95) for boundary errors, volume error in mL; for classification AUROC, sensitivity at a fixed specificity, and calibration. Report per-site and per-scanner results.
- **Privacy.** In the US, HIPAA de-identification is either Safe Harbor (remove the listed identifier types) or Expert Determination. DICOM headers, burned-in text on the pixels, and 3D face reconstruction from head CT/MRI can all re-identify patients.
- **Regulation.** Software that diagnoses or drives treatment is a medical device (FDA in the US; MDR plus the EU AI Act, where such devices are high-risk). FDA's PCCP guidance (finalised December 2024) lets a manufacturer pre-authorise planned model updates.

## When to use / scenarios
- Radiology: organ and tumour volumetry, lesion follow-up, automatic contouring of organs at risk for radiotherapy planning, chest X-ray triage of urgent findings.
- Research cohorts: extracting body composition, organ volumes or biomarkers from thousands of scans ([[healthcare]]).
- Labelling acceleration: TotalSegmentator or MedSAM pre-labels, a clinician corrects ([[data-labeling-and-synthetic-data]]).
- Pathology whole-slide images and ophthalmology have their own tooling (tiling of gigapixel slides, fundus/OCT models); the ideas here transfer but the libraries differ.
- Not for: unsupervised diagnosis, free-text radiology reports from a general VLM without clinical validation ([[vision-language-models]]), or any use where the vendor contract does not cover protected health information.

## Setup & code
Install (versions per the repos; install PyTorch for your hardware first):
```bash
pip install monai            # Apache-2.0
pip install nnunetv2         # self-configuring U-Net; needs nnUNet_raw / nnUNet_preprocessed / nnUNet_results dirs
pip install TotalSegmentator # Python >= 3.10, PyTorch >= 2.0
TotalSegmentator -i ct.nii.gz -o segmentations                 # CT, 117 classes
TotalSegmentator -i mri.nii.gz -o segmentations --task total_mr
```

MONAI training skeleton (dictionary transforms, random foreground-weighted crops, sliding-window validation):
<!-- skip-check: needs torch, monai and NIfTI files on disk -->
```python
import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import (Compose, EnsureChannelFirstd, LoadImaged, Orientationd,
                              RandCropByPosNegLabeld, ScaleIntensityRanged, Spacingd)

files = [{"image": "img001.nii.gz", "label": "seg001.nii.gz"}]           # your data
tf = Compose([
    LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0, b_max=1, clip=True),  # soft-tissue CT window
    RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4),
])
loader = DataLoader(CacheDataset(files, tf), batch_size=2, shuffle=True)
model = UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128), strides=(2, 2, 2))
loss_fn, opt = DiceCELoss(to_onehot_y=True, softmax=True), torch.optim.AdamW(model.parameters(), 1e-4)
for batch in loader:
    opt.zero_grad()
    loss = loss_fn(model(batch["image"]), batch["label"])
    loss.backward()
    opt.step()
# inference on a whole volume: tiles with overlap, blended
# pred = sliding_window_inference(volume, roi_size=(96, 96, 96), sw_batch_size=4, predictor=model, overlap=0.5)
```

Self-contained NumPy version of the core ideas (CT windowing, patch-wise sliding-window inference with overlap blending, Dice, HD95 and volume in mL from voxel spacing) on a synthetic CT volume with a known lesion:
```python
import numpy as np

rng = np.random.default_rng(0)

# synthetic CT in Hounsfield units: air background, soft-tissue body, liver-like organ, small hypodense lesion
Z, Y, X = 40, 96, 96
spacing = (2.5, 0.8, 0.8)                                   # mm (z, y, x); thick slices, as in many clinical CTs
zz, yy, xx = np.mgrid[:Z, :Y, :X]
ct = np.full((Z, Y, X), -1000.0)
body = ((yy - 48) / 44) ** 2 + ((xx - 48) / 44) ** 2 < 1
ct[body] = 40.0
organ = body & (((zz - 20) / 15) ** 2 + ((yy - 45) / 22) ** 2 + ((xx - 40) / 25) ** 2 < 1)
ct[organ] = 60.0
lesion = ((zz - 20) * spacing[0]) ** 2 + ((yy - 40) * spacing[1]) ** 2 + ((xx - 35) * spacing[2]) ** 2 < 9.0 ** 2
ct[lesion] = 15.0                                           # 45 HU darker than the organ
ct += rng.normal(0, 12, ct.shape)                           # scanner noise


def window(hu, level=50, width=400):
    """Clip HU to a window and scale to [0, 1] (soft-tissue window by default)."""
    lo, hi = level - width / 2, level + width / 2
    return (np.clip(hu, lo, hi) - lo) / (hi - lo)


def toy_model(patch):
    """Stand-in for a trained network: smooth, then score 'darker than organ but not fat or air'."""
    k = np.ones(3) / 3
    for ax in range(3):
        patch = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), ax, patch)
    lo, hi = window(-10), window(30)
    return ((patch > lo) & (patch < hi)).astype(float)


def sliding_window(vol, roi=(16, 32, 32), overlap=0.5, predictor=toy_model):
    """Tile the volume with overlapping patches and average the predictions (what MONAI's helper does)."""
    out, count = np.zeros_like(vol), np.zeros_like(vol)
    steps = [max(1, int(r * (1 - overlap))) for r in roi]
    starts = [sorted(set(list(range(0, s - r + 1, st)) + [s - r])) for s, r, st in zip(vol.shape, roi, steps)]
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                sl = (slice(z, z + roi[0]), slice(y, y + roi[1]), slice(x, x + roi[2]))
                out[sl] += predictor(vol[sl])
                count[sl] += 1
    return out / count


def dice(a, b):
    return 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)


def surface(m):
    inner = m.copy()
    for ax in range(3):
        inner &= np.roll(m, 1, ax) & np.roll(m, -1, ax)
    return np.argwhere(m & ~inner) * np.array(spacing)       # surface points in mm


def hd95(a, b):
    pa, pb = surface(a), surface(b)
    d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1))
    return max(np.percentile(d.min(1), 95), np.percentile(d.min(0), 95))


def largest_component_near(mask, seed_mask):
    """Post-processing: keep predicted voxels inside the organ region (a common anatomical prior)."""
    return mask & seed_mask


voxel_ml = np.prod(spacing) / 1000.0
img = window(ct)
prob = sliding_window(img)
pred = largest_component_near(prob > 0.5, organ)
print(f"true lesion volume {lesion.sum() * voxel_ml:.2f} mL, predicted {pred.sum() * voxel_ml:.2f} mL")
print(f"Dice {dice(pred, lesion):.3f}   HD95 {hd95(pred, lesion):.1f} mm")
raw = window(ct, level=50, width=2000)                       # a too-wide window squashes the 45 HU contrast
print(f"lesion vs organ contrast: soft-tissue window {img[organ & ~lesion].mean() - img[lesion].mean():.3f}, "
      f"wide window {raw[organ & ~lesion].mean() - raw[lesion].mean():.3f}")
assert dice(pred, lesion) > 0.6
```
Reading the output: the soft-tissue window gives the model about five times the lesion contrast of a wide window; volume needs the voxel spacing; and a Dice around 0.7 still hides an underestimated volume (the toy model misses the lesion rim), which is why volumetry tools report volume error and boundary distance alongside Dice.

## Choosing / trade-offs
- **Off-the-shelf model vs train your own.** If TotalSegmentator (or a MONAI bundle) covers your structure and modality, validate it on your scanners first; training is for structures they lack. Some TotalSegmentator subtasks need a licence (free for non-commercial use, contact the authors for commercial), and `brain_aneurysm` is CC BY-NC 4.0, per its README.
- **nnU-Net vs MONAI custom code.** nnU-Net gives a strong segmentation baseline with almost no tuning; MONAI is a toolkit for custom pipelines (classification, detection, registration, federated, deployment). Teams often benchmark with nnU-Net and build production code in MONAI.
- **2D vs 3D.** 2D slice models need less memory and data; 3D models use context across slices and usually segment better on isotropic CT, at higher GPU cost.
- **Promptable (MedSAM) vs fully automatic.** Prompted models suit interactive labelling and tools with a human in the loop; automatic models suit batch volumetry.
- **Cloud API vs on-premises.** Images are protected health information: a cloud model needs a BAA/DPA and data residency; many hospitals require on-premises inference ([[ai-security-privacy-compliance]]). Federated training avoids moving data between sites ([[federated-learning-and-differential-privacy]]).

## Gotchas
- **Patient-level splits.** Split by patient (and ideally by site), never by slice; slices from one patient in train and test inflate every metric ([[data-leakage-and-validation-splits]]).
- **Shortcut learning.** Models learn hospital markers, scanner type, portable-X-ray labels or chest drains instead of disease. Check saliency ([[saliency-maps-and-neural-attribution]]) and test on an external site.
- **Domain shift.** Scanner vendor, protocol, contrast phase and slice thickness change performance; validate per scanner and monitor after deployment ([[transfer-learning-and-domain-adaptation]]).
- **Label quality.** Inter-reader variability is large for many structures; measure it and use it as the ceiling for your metric.
- **Tiny structures and class imbalance.** Use Dice+CE losses and foreground oversampling; report per-class Dice and detection-level sensitivity for small lesions.
- **Resampling labels.** Use nearest-neighbour for masks; linear interpolation creates invalid label values.
- **De-identification is not just headers.** Burned-in annotations on pixels, private DICOM tags and facial reconstruction from head scans can leak identity.
- **Outputs are decision support.** Never present a segmentation or score as a diagnosis; regulated use needs clinical validation, a quality system and clearance.

## Related
- [[segmentation]] - general segmentation and SAM, which MedSAM adapts.
- [[image-classification]] - 2D classifiers used for radiograph triage.
- [[healthcare]] - sector scenario, compliance and workflows.
- [[ai-security-privacy-compliance]] - PHI, DPAs and data residency.
- [[federated-learning-and-differential-privacy]] - multi-site training without pooling scans.
- [[transfer-learning-and-domain-adaptation]] - scanner and site shift.
- [[data-labeling-and-synthetic-data]] - pre-labelling and annotation workflows.
- [[satellite-and-remote-sensing]] - the other large-raster, tiled-inference domain.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/security/privacy-and-pii-handling.md - handling PII/PHI in the surrounding system.

## References
- MONAI: https://github.com/Project-MONAI/MONAI
- nnU-Net v2: https://github.com/MIC-DKFZ/nnUNet
- TotalSegmentator: https://github.com/wasserth/TotalSegmentator
- MedSAM, "Segment Anything in Medical Images", Nature Communications 15, 654 (2024): https://github.com/bowang-lab/MedSAM
- HHS HIPAA de-identification guidance: https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
- FDA AI-enabled medical devices list: https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-enabled-medical-devices
- FDA final PCCP guidance for AI-enabled devices (December 2024), law-firm summary: https://ropesgray.com/en/insights/alerts/2024/12/fda-finalizes-guidance-on-predetermined-change-control-plans-for-ai-enabled-device
- Regulation (EU) 2024/1689 (AI Act): https://eur-lex.europa.eu/eli/reg/2024/1689/oj
