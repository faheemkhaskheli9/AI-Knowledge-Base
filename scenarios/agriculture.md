---
title: Agriculture and agritech AI
category: scenarios
tags: [agriculture, crop-disease, yield-forecasting, remote-sensing, satellite, ndvi, precision-farming]
use_cases:
  - "detect crop diseases or pests from leaf photos taken on a phone"
  - "forecast crop yield per field from satellite imagery and weather"
  - "map crop types or field boundaries from satellite images"
  - "build an agronomy advice assistant for farmers in local languages"
  - "count plants, fruit or livestock from drone or camera images"
status: draft
last_verified: 2026-10-15
sources:
  - https://dataspace.copernicus.eu/explore-data/data-collections/sentinel-data/sentinel-2
  - https://arxiv.org/abs/1511.08060
  - https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
---

# Agriculture and agritech AI

## Summary
Agricultural AI is mostly perception (disease, weeds, counting from phone,
drone or satellite images) and spatio-temporal prediction (yield, crop type,
irrigation need) from remote sensing plus weather. Data is seasonal, so a
model sees few truly independent years. Imagery is noisy (clouds, sensor
differences) and users often have weak connectivity. LLMs help as advisory
assistants over extension material, in local languages and by voice.

## Key concepts
- Remote sensing basics: multispectral satellites such as Copernicus
  Sentinel-2 give free imagery with 13 bands at 10/20/60 m resolution and a
  revisit of a few days. Vegetation indices like NDVI = (NIR - Red) / (NIR + Red)
  summarise canopy vigour and are a strong baseline feature.
- Clouds: optical images are often cloudy in the growing season. Use cloud
  masks, temporal compositing, or radar (SAR) that sees through clouds.
- Spatial and temporal leakage: neighbouring pixels and fields are
  correlated, and so are dates within a season. Validate by held-out field,
  region and season ([[data-leakage-and-validation-splits]]), not random pixels.
- Ground truth is the bottleneck: yield monitors, field surveys and
  agronomist labels are expensive and biased toward large farms. Use active
  learning and weak labels ([[semi-supervised-and-active-learning]]).
- Lab-to-field gap: leaf-disease models trained on clean single-leaf photos
  (e.g. PlantVillage, Hughes & Salathe 2015) drop sharply on real field
  photos with soil, shadows and several leaves. Collect in-field images.
- Advice is regulated in places: pesticide use must follow the product label
  and local law, so an assistant must not recommend off-label products or
  doses. Have agronomists approve the knowledge base.

## When to use / scenarios
1. **Crop disease and pest detection from phone photos.** Problem: farmers
   lack quick diagnosis. Approach: image classifier on in-field photos,
   on-device model for offline use, "uncertain - ask an agronomist" band.
   Read: [[image-classification]], [[transfer-learning-and-domain-adaptation]],
   [[edge-on-device]], [[quantization]].
2. **Yield forecasting per field or region.** Approach: season-to-date
   NDVI/weather features -> gradient boosting or linear model, prediction
   intervals, leave-one-season-out validation. Read:
   [[gradient-boosting-tabular]], [[time-series-forecasting]],
   [[quantile-regression]], [[conformal-prediction-and-uncertainty]].
3. **Crop-type and field-boundary mapping.** Approach: per-pixel or
   per-parcel classification on multi-date image stacks; segmentation for
   boundaries. Read: [[segmentation]], [[image-classification]].
4. **Weed detection for targeted spraying.** Approach: real-time detection
   or segmentation on the sprayer; latency and false-negative cost dominate.
   Read: [[object-detection]], [[segmentation]], [[edge-on-device]].
5. **Counting plants, fruit or livestock.** Approach: detection with
   tracking on drone or fixed-camera video. Read: [[object-detection]],
   [[video-analytics]].
6. **Irrigation and soil-moisture prediction from IoT sensors.** Read:
   [[time-series-forecasting]], [[anomaly-detection]],
   [[manufacturing-iot]] (similar sensor pipelines).
7. **Livestock health and behaviour monitoring.** Approach: anomaly
   detection on activity and feeding data; pose or detection on video.
   Read: [[anomaly-detection]], [[face-and-pose]].
8. **Agronomy advice assistant.** Approach: RAG over approved extension
   guides and product labels, citations, local language and voice, refusal
   for off-label chemical advice. Read: [[rag-basics]],
   [[guardrails-and-safety]], [[speech-to-text]], [[text-to-speech]],
   [[small-language-models]].
9. **Commodity price and demand forecasting.** Read:
   [[time-series-forecasting]], [[logistics-and-supply-chain]].
10. **Crop insurance and credit for smallholders.** Approach: remote-sensing
    features as evidence, with human review of payouts. Read: [[insurance]],
    [[finance]].

When NOT to use ML: if a simple index threshold or agronomic rule
(growing degree days, NDVI drop vs last year) already drives the decision,
ship that first and use ML only where it beats the rule on held-out seasons.

## Setup & code
Reference architecture (flagship: field-level yield forecast):

```
satellite scenes (cloud-masked) + weather + field polygons + soil maps
  -> per-field, per-date features (NDVI stats, rainfall, heat days)
  -> model trained on past seasons -> forecast + interval per field
  -> validation: leave-one-season-out and held-out regions
  -> dashboard/map for agronomists; re-forecast as the season progresses
```

Runnable sketch: NDVI features and leave-one-season-out validation on
synthetic data (replace with your own field table).

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

rng = np.random.default_rng(0)

def ndvi(nir, red):
    return (nir - red) / np.clip(nir + red, 1e-6, None)

# 6 seasons x 80 fields; each season has its own weather shock
n_season, n_field = 6, 80
season = np.repeat(np.arange(n_season), n_field)
shock = rng.normal(0, 0.05, n_season)[season]
red = rng.uniform(0.03, 0.12, season.size)
nir = rng.uniform(0.25, 0.55, season.size) + shock
rain = rng.normal(400, 80, season.size) + 500 * shock
X = np.column_stack([ndvi(nir, red), rain])
yield_t_ha = 2 + 6 * X[:, 0] + 0.003 * rain + rng.normal(0, 0.3, season.size)

model = GradientBoostingRegressor(random_state=0)
scores = cross_val_score(model, X, yield_t_ha, groups=season,
                         cv=LeaveOneGroupOut(),
                         scoring="neg_mean_absolute_error")
print("MAE per held-out season (t/ha):", np.round(-scores, 2))
```

## Choosing / trade-offs
- Free vs commercial imagery: free public satellites cover large areas at
  10 m or coarser; small plots, single plants and weeds need commercial
  high-resolution, drone or ground images at higher cost per hectare.
- On-device vs cloud: on-device works offline in the field but limits model
  size; cloud allows bigger models and central updates.
- Simple indices vs deep models: indices plus gradient boosting are cheap,
  explainable and hard to beat with few seasons of labels; deep
  spatio-temporal models pay off with many labelled fields and regions.
- General VLM vs trained classifier for diagnosis: a
  [[vision-language-models]] call is quick to prototype but must be
  evaluated on local crops and diseases; a trained classifier is cheaper and
  more predictable at scale.

## Gotchas
- Random pixel splits inflate accuracy because adjacent pixels are near
  duplicates; split by field, region and season.
- A model trained in one region or climate often fails in another; check
  per-region error before rollout.
- Mixing sensors or processing levels (top-of-atmosphere vs surface
  reflectance) silently shifts index values.
- A bad year (drought, flood) may be outside training data exactly when the
  forecast matters most; show intervals and flag out-of-range inputs
  ([[out-of-distribution-detection]]).
- Farm data ownership and consent: field boundaries, yields and locations
  are commercially sensitive and can be personal data; agree data terms
  with growers ([[ai-security-privacy-compliance]]).
- Hallucinated chemical names, rates or pre-harvest intervals are a real
  harm; ground every recommendation in the label text ([[hallucination-and-grounding]]).

## Related
- [[logistics-and-supply-chain]] - harvest logistics and demand forecasting.
- [[insurance]] - index-based and parametric crop insurance.
- [[government-public-sector]] - subsidy checks and land monitoring by agencies.
- [[manufacturing-iot]] - sensor pipelines and edge deployment patterns.
- [[document-processing]] - certificates, invoices and compliance paperwork.
- [[satellite-and-remote-sensing]] - multispectral imagery, cloud masking and change detection in depth.

## References
- Copernicus Data Space, Sentinel-2 collection: https://dataspace.copernicus.eu/explore-data/data-collections/sentinel-data/sentinel-2
- Hughes & Salathe, "An open access repository of images on plant health to enable the development of mobile disease diagnostics" (PlantVillage), 2015: https://arxiv.org/abs/1511.08060
- scikit-learn group-based cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold
- Regulation (EU) 2024/1689 (AI Act): https://eur-lex.europa.eu/eli/reg/2024/1689/oj
