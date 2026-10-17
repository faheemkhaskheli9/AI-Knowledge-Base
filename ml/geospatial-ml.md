---
title: Geospatial ML (spatial features, spatial CV, H3)
category: ml
tags: [geospatial, spatial, h3, geopandas, spatial-cross-validation, spatial-autocorrelation, morans-i, kriging, haversine, location-features]
use_cases:
  - "predict house prices, demand or risk from location without leaking neighbours into validation"
  - "build distance and neighbourhood features from latitude/longitude"
  - "aggregate GPS points or orders into hexagonal cells with H3"
  - "choose a cross-validation scheme for spatially autocorrelated data"
status: draft
last_verified: 2026-10-15
sources:
  - https://h3geo.org/docs/
  - https://uber.github.io/h3-py/
  - https://geopandas.org/en/stable/
  - https://pysal.org/esda/
  - https://scikit-learn.org/stable/modules/cross_validation.html
  - https://doi.org/10.1111/ecog.02881
---

# Geospatial ML (spatial features, spatial CV, H3)

## Summary
Geospatial ML predicts something tied to a place: property value, delivery
time, store demand, crop yield, flood or crime risk. Location is usually the
strongest feature, but it needs encoding (distances, neighbourhood
aggregates, grid cells such as H3 hexagons), and nearby points are
correlated, so random cross-validation leaks neighbours into the validation
set and overstates accuracy. Use spatial features with a gradient-boosted
model, validate with spatial blocks, and check residuals for remaining
spatial patterns.

## Key concepts
- **Coordinates are not Euclidean.** Latitude/longitude are angles; use the
  haversine (great-circle) distance or project to a local metric CRS (e.g.
  UTM) before computing distances or buffers. Store the CRS with the data.
- **Spatial autocorrelation.** Near things are more alike (Tobler's first
  law). Measure it with Moran's I on values or on model residuals; positive
  residual autocorrelation means the model misses spatial structure.
- **Spatial features.**
  - Raw lat/lon (trees can split on them) and rotated coordinates.
  - Distances to points of interest: centre, coast, station, nearest
    competitor store.
  - Counts and densities within a radius or cell: shops, crimes, orders.
  - Neighbourhood aggregates of the target (spatial lag), computed out of
    fold only, e.g. median price of the k nearest *training* sales.
  - Region attributes joined by polygon: census tract income, zoning.
  - Raster values at the point: elevation, land cover, satellite indices.
- **Discrete global grids (H3).** H3 indexes the globe into hexagonal cells
  at 16 resolutions (0 coarsest to 15 finest). A cell ID is a categorical
  feature and a join key; `grid_disk` gives k-ring neighbours for smoothing
  and neighbourhood features. Hexagons have uniform neighbour distances,
  unlike squares. Alternatives: S2 cells, geohash.
- **Spatial cross-validation.** Group rows into spatial blocks (grid cells,
  H3 parents, clusters of coordinates, or administrative areas) and use
  `GroupKFold`; optionally leave a buffer around each test block. Roberts
  et al. (2017) recommend block CV whenever data have spatial dependence and
  the model must predict for new areas.
- **Interpolation models.** Kriging and Gaussian processes model the spatial
  covariance directly and give uncertainty
  ([[gaussian-process-regression-from-scratch]]); GBMs with spatial features
  scale better and use many covariates.

## When to use / scenarios
- **Real estate valuation:** location features + structure features in a
  GBM, spatial CV by neighbourhood ([[real-estate]]).
- **Delivery and mobility:** ETA and demand per H3 cell and hour; surge or
  courier positioning.
- **Retail site selection:** demand prediction for candidate locations from
  footfall, demographics, competitors ([[ecommerce-retail]]).
- **Insurance and climate risk:** flood, wildfire or hail exposure from
  rasters and historic events.
- **Agriculture and environment:** yield or species-distribution models
  from satellite imagery and weather.
- **Not when:** location only identifies an entity you already model (use an
  ID or entity features), or the task is pixel-level imagery
  ([[segmentation]], [[image-classification]]).

## Setup & code
```bash
pip install numpy scikit-learn             # example below
pip install h3 geopandas shapely pyproj    # grids, vector data, projections
pip install esda libpysal rasterio         # spatial statistics, rasters
```
Random K-fold vs spatial block CV on autocorrelated data, plus Moran's I on
residuals:
```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_score, cross_val_predict

rng = np.random.default_rng(0)
n = 2000
lat = rng.uniform(51.3, 51.7, n); lon = rng.uniform(-0.5, 0.3, n)   # one city

def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))

# Smooth spatial field (unobserved "neighbourhood effect") + one observed feature
centres = np.column_stack([rng.uniform(51.3, 51.7, 30), rng.uniform(-0.5, 0.3, 30)])
w = rng.normal(0, 1, 30)
field = sum(wi * np.exp(-haversine_km(lat, lon, c[0], c[1]) ** 2 / 20)
            for wi, c in zip(w, centres))
size = rng.normal(0, 1, n)
y = 2 * field + size + rng.normal(0, 0.3, n)

dist_centre = haversine_km(lat, lon, 51.5074, -0.1278)             # distance feature
X = np.column_stack([lat, lon, dist_centre, size])

# Spatial blocks: ~4 km grid cells as groups (H3 parent cells work the same way)
block = (np.floor((lat - 51.3) / 0.036) * 100 + np.floor((lon + 0.5) / 0.058)).astype(int)
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=0, n_jobs=1)
r2_random = cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
r2_block = cross_val_score(rf, X, y, cv=GroupKFold(5), groups=block, scoring="r2")
print(f"random KFold R2: {r2_random.mean():.3f}   spatial block R2: {r2_block.mean():.3f}")

def morans_i(values, lat, lon, k=8):
    """Moran's I with a row-standardised k-nearest-neighbour weight matrix."""
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    z = values - values.mean()
    lag = z[nn].mean(axis=1)                       # spatial lag (W z)
    return len(z) * (z * lag).sum() / (len(z) * (z ** 2).sum())   # S0 = n for row-standardised W

resid_no_loc = y - cross_val_predict(RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1),
                                     size[:, None], y, cv=GroupKFold(5), groups=block)
print(f"Moran's I of target: {morans_i(y, lat, lon):.2f}, "
      f"of residuals without location features: {morans_i(resid_no_loc, lat, lon):.2f}")
```
The random-split score is higher because test points have training
neighbours a few hundred metres away; the block score is closer to what a new
area would see. Residual Moran's I well above 0 says location is missing.

H3 indexing (h3-py v4 API):
<!-- skip-check: needs the h3 package -->
```python
import h3
cell = h3.latlng_to_cell(51.5074, -0.1278, 8)      # resolution 8 cell id (string)
parent = h3.cell_to_parent(cell, 6)                  # coarser block for spatial CV
ring = h3.grid_disk(cell, 1)                         # cell + its 6 neighbours
print(cell, parent, len(ring), h3.cell_to_latlng(cell))
```

## Choosing / trade-offs
| Choice | Options | Pushes one way |
|---|---|---|
| Location encoding | raw lat/lon, distances, H3 cell IDs, target spatial lag | Trees handle raw coords; cell IDs need enough rows per cell; spatial lag is strong but leak-prone |
| Grid system | H3, S2, geohash, admin areas | H3 for uniform neighbours and analytics; admin areas when decisions are per region |
| Resolution | coarse vs fine cells | Fine cells are sparse and noisy; pick by rows per cell, or use several resolutions |
| Model | GBM + features vs kriging/GP vs spatial GNN | GBM for many covariates and scale; GP for small data with uncertainty; GNN for network-shaped space (roads) |
| CV | random, spatial block, buffered block, leave-region-out | Block CV whenever deploying to unseen areas; random only for interpolation inside dense coverage |

- **Block size:** at least the range over which residuals stay correlated
  (inspect a semivariogram or Moran's I by distance); too small leaks, too
  large leaves training folds unrepresentative.
- **Tooling:** GeoPandas/Shapely for vector data, rasterio for rasters,
  PySAL (esda, libpysal) for spatial statistics, a spatial database
  (PostGIS) for large joins.

## Gotchas
- Mixing lat/lon order (GeoJSON uses lon, lat) or CRSs silently produces
  wrong distances and joins; assert bounds after every load.
- Euclidean distance on degrees distorts with latitude; at high latitudes a
  degree of longitude is much shorter than a degree of latitude.
- Target-derived neighbourhood features computed on all rows leak the label;
  compute them from training folds only
  ([[data-leakage-and-validation-splits]]).
- Random CV on clustered samples (many sales in one block, many images of
  one field) inflates scores; group by location.
- Data density bias: models are confident where data are dense and wrong in
  sparse rural areas; report error by region.
- Location as a proxy for protected attributes (redlining): lending,
  insurance and housing models need fairness checks on geographic features.
- Location data is personal data; aggregate to cells and apply k-anonymity
  style thresholds before sharing ([[federated-learning-and-differential-privacy]]).
- H3 cells are not equal-area across the globe; normalise counts by cell
  area when comparing distant regions.

## Related
- [[feature-engineering]] - general encoding and aggregation patterns.
- [[gradient-boosting-tabular]] - the usual model for spatial features.
- [[data-leakage-and-validation-splits]] - grouped and blocked splits.
- [[gaussian-process-regression-from-scratch]] - kriging-style interpolation with uncertainty.
- [[distance-metrics-and-similarity]] - distance choices for neighbours.
- [[dbscan-from-scratch]] - density clustering of GPS points with haversine.
- [[graph-neural-networks]] - road and mobility networks as graphs.
- [[real-estate]] - valuation models that rely on these features.
- See also (SE KB): https://github.com/faheemkhaskheli9/Software-Engineering-KnowledgeBase/blob/main/case-studies/design-ride-sharing.md

## References
- H3 documentation: https://h3geo.org/docs/
- h3-py (Python bindings, v4 API): https://uber.github.io/h3-py/
- GeoPandas documentation: https://geopandas.org/en/stable/
- PySAL esda (Moran's I and other spatial statistics): https://pysal.org/esda/
- scikit-learn cross-validation (GroupKFold): https://scikit-learn.org/stable/modules/cross_validation.html
- Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure", Ecography 40(8), 2017: https://doi.org/10.1111/ecog.02881
