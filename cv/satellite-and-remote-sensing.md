---
title: Satellite and remote sensing imagery (multispectral, change detection, EO foundation models)
category: cv
tags: [remote-sensing, satellite, earth-observation, geospatial, sentinel-2, multispectral, ndvi, change-detection, torchgeo, prithvi, clay, geotiff, stac]
use_cases:
  - "map crops, forest or water from Sentinel-2 imagery over a region"
  - "detect new buildings, deforestation or flood extent between two dates"
  - "fine-tune a geospatial foundation model on a small labelled land-cover dataset"
  - "build a training pipeline over large GeoTIFF rasters with matching label layers"
  - "compute vegetation and water indices as features for a yield or risk model"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/microsoft/torchgeo
  - https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2/Facts_and_figures
  - https://arxiv.org/abs/2412.02732
  - https://github.com/Clay-foundation/model
  - https://ar5iv.arxiv.org/html/1810.08468
  - https://stacspec.org/
---

# Satellite and remote sensing imagery (multispectral, change detection, EO foundation models)

## Summary
Earth observation (EO) imagery is large, georeferenced and multispectral: a Sentinel-2 tile has 13 bands from visible to shortwave infrared at 10, 20 and 60 m resolution, revisiting each place every few days. Typical tasks are land-cover segmentation, object detection (buildings, ships, vehicles in very-high-resolution imagery), regression (biomass, yield) and change detection between dates. Tooling differs from photo CV: coordinate reference systems, reprojection, cloud masks, tiling of huge rasters and more than three channels. TorchGeo handles geospatial datasets and samplers in PyTorch; foundation models such as Prithvi-EO-2.0 (IBM/NASA) and Clay are pretrained on multispectral, multi-temporal data and fine-tune with few labels.

## Key concepts
- **Sensors.** Optical multispectral (Sentinel-2, Landsat; free, 10-30 m), commercial very-high-resolution optical (sub-metre, paid licences), and SAR (Sentinel-1: radar, sees through clouds and at night, but speckled and hard to read). Choose by object size, revisit and cloud cover.
- **Sentinel-2 facts (ESA).** 13 bands: four at 10 m (blue, green, red, near-infrared), six at 20 m (red-edge, SWIR), three at 60 m (atmospheric). 290 km swath; 5-day revisit at the equator with two satellites.
- **Processing levels.** Top-of-atmosphere vs surface reflectance (Sentinel-2 L1C vs L2A). Train and infer on the same level; L2A includes a scene classification layer usable as a cloud/shadow mask.
- **Spectral indices.** NDVI = (NIR - Red)/(NIR + Red) for vegetation; NDWI = (Green - NIR)/(Green + NIR) for open water; NBR (NIR vs SWIR) for burn scars. Cheap, explainable features and baselines.
- **Georeferencing.** Rasters (GeoTIFF/COG) carry a CRS and affine transform; labels are often vector polygons. Reproject everything to one CRS and rasterise labels onto the image grid. STAC catalogues index scenes by place, time and cloud cover.
- **Tiling.** Scenes are thousands of pixels wide: sample fixed-size patches for training, predict with overlapping tiles and stitch (the same pattern as medical volumes, [[medical-imaging]]).
- **Change detection.** Compare co-registered images of two dates: index differencing, change vector analysis (magnitude of the per-pixel spectral difference) with a threshold such as Otsu, or Siamese/early-fusion CNNs trained on change labels (OSCD for Sentinel-2 urban change; LEVIR-CD for 0.5 m building change).
- **Foundation models for EO.** Masked-autoencoder style pretraining on multispectral time series. Prithvi-EO-2.0 is pretrained on NASA's Harmonized Landsat Sentinel-2 (HLS) data with six bands and released on Hugging Face with fine-tuning through IBM's TerraTorch; Clay (Apache-2.0 code and weights) produces embeddings for Sentinel-1/2 and other sources.

## When to use / scenarios
- Agriculture: crop type maps, field boundaries, vegetation health over the season, yield features ([[time-series-forecasting]]).
- Environment and climate: deforestation alerts, wetland and water mapping, burn scars, land-cover change reporting.
- Insurance and disaster response: flood extent (SAR works under cloud), building damage after storms or earthquakes.
- Urban planning and infrastructure: new construction, roads, solar panel and roof detection from very-high-resolution imagery ([[object-detection]]).
- Maritime and defence-adjacent monitoring: ship detection in SAR; check export-control and licensing terms of the imagery.
- Not for: detecting small objects (cars, people) in 10 m imagery, where each pixel is 100 square metres; buy higher resolution or change the question.

## Setup & code
```bash
pip install torchgeo          # MIT; PyTorch domain library for geospatial data
pip install rasterio          # read/write GeoTIFFs, CRS and transforms
```
TorchGeo combines raster datasets by location: `&` intersects (imagery that has labels), `|` unions (either sensor), and geospatial samplers draw patches from the overlap:
<!-- skip-check: needs torch, torchgeo and raster files on disk -->
```python
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import RandomGeoSampler

imagery = RasterDataset(paths="data/sentinel2/")            # GeoTIFFs; CRS/resolution handled for you
labels = RasterDataset(paths="data/landcover/")
labels.is_image = False                                     # treat as a mask
dataset = imagery & labels                                  # only places covered by both
sampler = RandomGeoSampler(dataset, size=256, length=1000)  # 256-pixel patches per epoch
loader = DataLoader(dataset, sampler=sampler, batch_size=8, collate_fn=stack_samples)
batch = next(iter(loader))
print(batch["image"].shape, batch["mask"].shape)
```

Self-contained NumPy demo of the classical baseline: two dates of synthetic 4-band (blue, green, red, NIR) imagery where some forest is cleared and a reservoir is filled, plus a cloud on date 2. It computes NDVI/NDWI, detects change with change vector analysis and an Otsu threshold, and shows why the cloud mask matters:
```python
import numpy as np

rng = np.random.default_rng(1)
H = W = 128
yy, xx = np.mgrid[:H, :W]

# surface reflectance per class: blue, green, red, NIR
SPEC = {"forest": [0.03, 0.06, 0.04, 0.35], "bare": [0.10, 0.14, 0.18, 0.25],
        "water": [0.06, 0.08, 0.05, 0.02], "crop": [0.04, 0.09, 0.06, 0.45], "cloud": [0.6, 0.6, 0.6, 0.6]}


def render(classes):
    img = np.zeros((4, H, W))
    for name, ref in SPEC.items():
        m = classes == name
        img[:, m] = np.array(ref)[:, None]
    return img + rng.normal(0, 0.01, img.shape)              # sensor noise


t1 = np.full((H, W), "crop", dtype=object)
t1[:, :64] = "forest"
t1[90:, 70:] = "bare"
t2 = t1.copy()
cleared = (yy > 20) & (yy < 60) & (xx > 10) & (xx < 50)     # deforestation
t2[cleared] = "bare"
filled = (yy > 95) & (xx > 80) & (xx < 120)                 # new reservoir
t2[filled] = "water"
truth = cleared | filled
cloud = (yy - 40) ** 2 + (xx - 100) ** 2 < 15 ** 2          # a cloud on date 2 over unchanged cropland
t2_obs = t2.copy()
t2_obs[cloud] = "cloud"
a, b = render(t1), render(t2_obs)


def ndvi(img):
    return (img[3] - img[2]) / (img[3] + img[2] + 1e-6)


def ndwi(img):
    return (img[1] - img[3]) / (img[1] + img[3] + 1e-6)


def otsu(values, bins=256):
    """Threshold that maximises between-class variance of a 1-D distribution."""
    hist, edges = np.histogram(values, bins=bins)
    p = hist / hist.sum()
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu0 = np.cumsum(p * mids)
    between = (mu0[-1] * w0 - mu0) ** 2 / (w0 * (1 - w0) + 1e-12)
    return mids[np.nanargmax(between)]


def f1_iou(pred, gt):
    tp, fp, fn = (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()
    return 2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)


cva = np.sqrt(((b - a) ** 2).sum(0))                        # change vector magnitude per pixel
for name, valid in [("no cloud mask", np.ones((H, W), bool)),
                    ("with cloud mask", ~(b.mean(0) > 0.4))]:   # crude brightness test; L2A ships a proper mask
    t = otsu(cva[valid])
    pred = (cva > t) & valid
    f1, iou = f1_iou(pred, truth)
    print(f"{name:16s} threshold {t:.3f}  F1 {f1:.3f}  IoU {iou:.3f}  false-change pixels {(pred & ~truth).sum()}")

d_ndvi, d_ndwi = ndvi(b) - ndvi(a), ndwi(b) - ndwi(a)
print(f"mean NDVI change: cleared forest {d_ndvi[cleared].mean():+.2f}, unchanged forest {d_ndvi[(t1 == 'forest') & ~cleared].mean():+.2f}")
print(f"mean NDWI change: new reservoir {d_ndwi[filled].mean():+.2f}")
```
Reading the output: without a mask, the very bright cloud dominates the histogram, Otsu puts the threshold between cloud and everything else, and the real changes are missed entirely (F1 0) while the cloud is reported as change; masking clouds first recovers the true changes. Index differences label what kind of change happened (vegetation loss vs new water), which raw change magnitude does not.

## Choosing / trade-offs
- **Indices + classical ML vs deep models.** Indices and gradient boosting on per-pixel time series ([[gradient-boosting-tabular]]) are strong, cheap baselines for crop and land-cover mapping. CNNs/ViTs win on spatial patterns (buildings, roads, field boundaries).
- **Foundation model vs training from ImageNet weights.** EO foundation models (Prithvi, Clay, TorchGeo's sensor-specific pretrained weights) handle extra bands and need fewer labels; ImageNet RGB backbones need band adaptation and ignore NIR/SWIR unless you widen the first layer ([[transfer-learning-and-domain-adaptation]]). Check each model's band order, resolution and licence before choosing.
- **Optical vs SAR.** Optical is easier to interpret and label; SAR works through clouds (floods, monsoon regions, high latitudes) at the cost of speckle and harder labelling. Fusing both is common.
- **Free 10-30 m vs commercial sub-metre.** Free data covers large areas often; commercial imagery is needed for small objects but adds cost, licence limits on derived products, and fewer revisits.
- **Pixel vs object vs scene outputs.** Segmentation for area statistics, detection for counts, scene classification for cheap screening ([[segmentation]], [[object-detection]], [[image-classification]]).

## Gotchas
- **Spatial autocorrelation leaks.** Random pixel or patch splits put neighbouring, near-identical pixels in train and test. Split by region or tile blocks ([[data-leakage-and-validation-splits]]).
- **Band order and scaling.** Products store reflectance as scaled integers with an offset that has changed between processing baselines; a model trained on one scaling silently degrades on another. Normalise with the statistics the pretrained model expects.
- **Clouds and shadows.** Unmasked clouds dominate change maps and training labels. Use the product's mask, composite several dates, or use SAR.
- **Seasonality.** Two dates in different seasons show phenology, not change. Compare the same season or model the full time series.
- **Co-registration.** A one-pixel misalignment between dates creates false change along every edge; check alignment, or use object-level comparison.
- **CRS mistakes.** Mixing degrees and metres, or labels in another CRS, shifts masks by kilometres. Reproject explicitly and overlay a sample visually.
- **Area statistics from classifiers are biased.** Pixel counting inherits classification errors; for official area estimates use a sample-based accuracy assessment.
- **Licences.** Imagery and derived-product licences differ (open Copernicus data vs commercial terms); model weights and datasets carry their own licences too ([[model-licenses]]).

## Related
- [[segmentation]] - land-cover and building masks.
- [[object-detection]] - ships, vehicles and buildings in very-high-resolution imagery.
- [[image-classification]] - scene classification and patch screening.
- [[transfer-learning-and-domain-adaptation]] - adapting pretrained backbones to new bands and regions.
- [[autoencoders-and-self-supervised-learning]] - masked-autoencoder pretraining behind EO foundation models.
- [[data-leakage-and-validation-splits]] - spatial blocking for honest evaluation.
- [[medical-imaging]] - the same tiled, large-raster inference pattern.
- [[model-licenses]] - checking weight and dataset licences.

## References
- TorchGeo: https://github.com/microsoft/torchgeo
- ESA Sentinel-2 facts and figures: https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2/Facts_and_figures
- Szwarcman et al., "Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications" (2024): https://arxiv.org/abs/2412.02732
- Clay foundation model: https://github.com/Clay-foundation/model
- Daudt et al., "Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks" (OSCD, IGARSS 2018): https://ar5iv.arxiv.org/html/1810.08468
- STAC specification: https://stacspec.org/
