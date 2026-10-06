---
title: HOG descriptor from scratch (gradients, orientation bins, L2-Hys block normalisation)
category: cv
tags: [hog, histogram-of-oriented-gradients, feature-descriptor, pedestrian-detection, classical-cv, scikit-image, numpy, from-scratch]
use_cases:
  - "detect people or simple objects on a CPU without a neural network"
  - "build hand-crafted image features for a small dataset and a linear classifier"
  - "understand the feature pipeline that CNNs replaced, and what its parameters mean"
status: draft
last_verified: 2026-10-06
sources:
  - https://ieeexplore.ieee.org/document/1467360
  - https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
---

# HOG descriptor from scratch (gradients, orientation bins, L2-Hys block normalisation)

## Summary
Histogram of Oriented Gradients (Dalal & Triggs 2005) describes an image window by how strongly its edges point in each direction, cell by cell. The steps are: take image gradients, vote each pixel's gradient magnitude into an orientation histogram for its 8x8 cell, normalise groups of 2x2 cells against lighting changes, and concatenate everything. A linear SVM on HOG of 64x128 windows was the standard pedestrian detector before CNNs, and OpenCV still ships it. This file implements the descriptor in NumPy, classifies synthetic person windows vs background, and compares the features to `skimage.feature.hog`.

## Key concepts
- **Gradients.** Centred `[-1, 0, 1]` differences with no Gaussian smoothing first. Dalal & Triggs found that smoothing hurts. Magnitude is `sqrt(gx^2 + gy^2)` and orientation is `atan2(gy, gx)`.
- **Unsigned orientation.** Angles folded into 0–180°, so a dark-to-light edge and a light-to-dark edge count the same. 9 bins of 20° each.
- **Interpolated voting.** Each pixel splits its magnitude between the two nearest bin centres in proportion to distance, so a small rotation does not flip a whole vote from one bin to the next. (The original also interpolates spatially between cells; this version skips that.)
- **Block normalisation.** 2x2 cells (36 values) are L2-normalised, clipped at 0.2 and normalised again (L2-Hys). Blocks slide by one cell, so each cell appears in up to 4 blocks under different normalisations. That redundancy is deliberate and helped in the paper.
- **Size.** A 64x128 window has 8x16 cells, 7x15 block positions, and 7·15·4·9 = 3780 features.

## When to use / scenarios
- CPU-only or microcontroller-class detection of rigid, upright objects (pedestrians, vehicles from a fixed camera, signs) where a small CNN is too heavy or there is no training data for one.
- Features for small datasets: HOG + linear SVM can beat a CNN trained from scratch on a few hundred images. A pretrained CNN embedding usually beats both ([[image-classification]]).
- Pre-filters and baselines: a fast HOG detector proposing windows for a heavier model, or a sanity baseline for a new detection task ([[object-detection]]).
- Not for: deformable or rotated objects, heavy occlusion, or any task where a pretrained detector (YOLO, DETR) fits the compute budget. Those win by a wide margin.

## Setup & code
NumPy only. `scikit-image` (`pip install scikit-image`) is used only as an optional cross-check.

```python
import numpy as np


def hog(img, cell=8, block=2, bins=9, eps=1e-5):
    """Histogram of Oriented Gradients (Dalal & Triggs 2005) for a 2-D float image.
    Unsigned orientations 0-180, linear vote split between the two nearest bins,
    L2-Hys block normalisation, blocks slide by one cell."""
    img = img.astype(np.float64)
    gx, gy = np.zeros_like(img), np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]              # centred [-1, 0, 1], no smoothing
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180

    ch, cw = img.shape[0] // cell, img.shape[1] // cell
    width = 180 / bins
    pos = ang / width - 0.5                              # bin centres at 10, 30, ..., 170 deg
    lo = np.floor(pos).astype(int)
    frac = pos - lo
    hist = np.zeros((ch, cw, bins))
    for r in range(ch * cell):
        for c in range(cw * cell):
            h = hist[r // cell, c // cell]
            h[lo[r, c] % bins] += mag[r, c] * (1 - frac[r, c])
            h[(lo[r, c] + 1) % bins] += mag[r, c] * frac[r, c]

    feats = []
    for r in range(ch - block + 1):
        for c in range(cw - block + 1):
            v = hist[r:r + block, c:c + block].ravel()
            v = v / np.sqrt((v ** 2).sum() + eps ** 2)
            v = np.minimum(v, 0.2)                       # "Hys": clip then renormalise
            v = v / np.sqrt((v ** 2).sum() + eps ** 2)
            feats.append(v)
    return np.concatenate(feats), hist


# Synthetic 64x128 "pedestrian window" vs background, the classic HOG input size
rng = np.random.default_rng(0)


def person(h=128, w=64):
    im = rng.normal(0.5, 0.05, (h, w))
    yy, xx = np.mgrid[:h, :w]
    head = (yy - 22) ** 2 + (xx - 32) ** 2 < 9 ** 2
    torso = (abs(xx - 32) < 11) & (yy > 32) & (yy < 80)
    legs = ((abs(xx - 25) < 4) | (abs(xx - 39) < 4)) & (yy >= 80) & (yy < 120)
    im[head | torso | legs] -= rng.uniform(0.25, 0.4)
    return np.roll(im, rng.integers(-3, 4), axis=1)


def background(h=128, w=64):
    im = rng.normal(0.5, 0.05, (h, w))
    for _ in range(3):                                   # a few random blobs
        y0, x0, r = rng.integers(10, h - 10), rng.integers(5, w - 5), rng.integers(4, 12)
        yy, xx = np.mgrid[:h, :w]
        im[(yy - y0) ** 2 + (xx - x0) ** 2 < r * r] -= rng.uniform(0.2, 0.4)
    return im


f, hist = hog(person())
print(f"feature length {f.size} = 7x15 blocks x 2x2 cells x 9 bins = {7 * 15 * 4 * 9}")
torso_edge = hist[6, 2]                                  # cell on the torso's left edge
print("cell (6,2) histogram, bin centres 10..170 deg:", np.round(torso_edge / torso_edge.max(), 2))

# Linear classifier on HOG features (least squares, the lazy stand-in for a linear SVM)
Xtr = np.array([hog(person())[0] for _ in range(60)] + [hog(background())[0] for _ in range(60)])
ytr = np.r_[np.ones(60), -np.ones(60)]
Xte = np.array([hog(person())[0] for _ in range(40)] + [hog(background())[0] for _ in range(40)])
yte = np.r_[np.ones(40), -np.ones(40)]
A = np.c_[Xtr, np.ones(len(Xtr))]
w = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ ytr)   # ridge
acc = (np.sign(np.c_[Xte, np.ones(len(Xte))] @ w) == yte).mean()
print(f"test accuracy person vs background: {acc:.2%}")

# Cross-check with scikit-image using the same settings
try:
    from skimage.feature import hog as sk_hog
    img = person()
    ref = sk_hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                 block_norm="L2-Hys", feature_vector=True)
    ours = hog(img)[0]
    print(f"skimage length {ref.size}; correlation with ours {np.corrcoef(ours, ref)[0, 1]:.3f}")
except ImportError:
    pass

assert hog(np.ones((128, 64)))[0].max() == 0           # flat image has no gradients
```

Output (Python 3.14, NumPy 2.5, scikit-image 0.26):
```
feature length 3780 = 7x15 blocks x 2x2 cells x 9 bins = 3780
cell (6,2) histogram, bin centres 10..170 deg: [0.97 0.33 0.21 0.17 0.23 0.16 0.14 0.24 1.  ]
test accuracy person vs background: 100.00%
skimage length 3780; correlation with ours 0.943
```

How to read it:
- 3780 is the classic Dalal–Triggs descriptor length for a 64x128 window. OpenCV's `HOGDescriptor` default has the same size.
- The cell on the torso's left edge puts nearly all its mass in the 10° and 170° bins. Those are the two ends of the same near-horizontal gradient direction (unsigned angles wrap around), i.e. a vertical edge. The other bins hold only noise.
- A ridge-regression linear classifier separates the synthetic person shape from blobs perfectly. The point is that the silhouette is linearly visible in HOG space. Real pedestrian data is much harder: the original paper reports miss rates, not 100%.
- The features correlate 0.943 with scikit-image's. The gap is binning detail: scikit-image does not interpolate votes between orientation bins and handles the image borders differently. Both are valid HOGs, but they are not interchangeable. Train and run inference with the same implementation.

## Choosing / trade-offs
- **Use a library.** `skimage.feature.hog` for features and visualisation, and `cv2.HOGDescriptor` (with `getDefaultPeopleDetector()`) for a ready-made, multi-scale pedestrian detector. The pure-Python voting loop above is for understanding, and it is slow.
- **HOG vs CNN features.** On anything beyond rigid, upright, well-lit objects, an ImageNet-pretrained backbone gives much better features for the same effort. HOG still wins on transparency, CPU cost and having no weights to ship.
- **Parameters.** Cell 8 px for a 64x128 person; scale cell size with object size. 9 unsigned bins is the standard choice. Signed 18 bins help when contrast direction matters (e.g. cars vs road) and hurt for clothing.
- **Classifier.** A linear SVM is the standard pairing (HOG is high-dimensional and roughly linearly separable). Kernel SVMs add little for the cost. Detection needs a sliding window over an image pyramid plus [[nms-iou-and-map-from-scratch]].

## Gotchas
- Colour images: the original takes, for each pixel, the gradient of the colour channel with the largest magnitude. Converting to grey first loses some edges. Pick one approach and use it everywhere.
- A fixed window aspect means people must be roughly 64x128 at some pyramid level. Sitting, lying or partly visible people are missed.
- Hard-negative mining matters more than the features. The paper retrains on false positives found by scanning person-free images, and skipping that gives many false alarms.
- Feature vectors grow fast: a 128x256 window at the same settings is 4x larger. Keep the window small and use the pyramid for scale.
- Implementations disagree on details (border cells, interpolation, `sqrt` gamma compression), so features from different libraries are not compatible.

## Related
- [[object-detection]] - modern detectors that replaced HOG + SVM.
- [[canny-edge-detector-from-scratch]] - the same image-gradient step, used for thin edges instead of histograms.
- [[harris-corner-detector-from-scratch]] - another classical gradient-based feature.
- [[nms-iou-and-map-from-scratch]] - merging and scoring sliding-window detections.
- [[svm-from-scratch]] - the linear classifier usually trained on HOG features.

## References
- Dalal & Triggs (2005), "Histograms of Oriented Gradients for Human Detection", CVPR: https://ieeexplore.ieee.org/document/1467360
- scikit-image `feature.hog`: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog
