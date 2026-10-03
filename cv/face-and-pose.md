---
title: Face detection, recognition and pose estimation
category: cv
tags: [face-detection, face-recognition, pose-estimation, keypoints, biometrics, privacy, gdpr, insightface, mediapipe]
use_cases:
  - "detect faces to blur them for privacy in images or video"
  - "count people and estimate body pose for fitness or ergonomics apps"
  - "verify that a selfie matches an ID photo (1:1 face verification)"
  - "detect falls or unsafe posture from camera footage"
  - "assess legal risk before building face recognition"
status: draft
last_verified: 2026-10-03
sources:
  - https://github.com/deepinsight/insightface
  - https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
  - https://docs.ultralytics.com/tasks/pose/
  - https://eur-lex.europa.eu/eli/reg/2024/1689/oj
  - https://gdpr-info.eu/art-9-gdpr/
---

# Face detection, recognition and pose estimation

## Summary
Three distinct tasks: face detection (where are faces), face recognition (who is this: 1:1 verification or 1:N identification via embeddings), and pose estimation (body keypoints for skeletons, used in fitness, ergonomics, fall detection and action recognition). Detection and pose are low-risk technical tools; face recognition processes biometric data and is heavily regulated. Read the legal cautions below before building any recognition feature.

## Key concepts
- Face detection: boxes plus landmarks (RetinaFace, SCRFD, MediaPipe, YuNet).
- Face recognition: a network maps an aligned face to an embedding; compare by cosine distance against a threshold. Verification (1:1) is lower risk than identification (1:N, search a gallery).
- Pose estimation: top-down (detect people then keypoints) or bottom-up; outputs 17-33 keypoints per person with confidences (COCO 17 keypoints; MediaPipe 33 landmarks).
- Liveness / anti-spoofing: separate models to defeat photos and screens; required for secure verification.
- Metrics: FAR/FRR and ROC at a threshold; accuracy differs by demographic group, so test fairness explicitly.
- Pose downstream: joint angles, rep counting, fall rules, or skeleton-based action recognition ([[video-analytics]]).

## When to use / scenarios
- Privacy: auto-blur faces and plates before sharing footage (detection only; no identity needed).
- Fitness/physio, sports analysis, ergonomics at work, fall detection for elder care: pose estimation.
- KYC onboarding: 1:1 selfie-to-ID verification with liveness, explicit consent and a fallback path.
- Prefer non-biometric alternatives where possible (badges, PINs, QR codes); use person detection and counts instead of identification.
- Avoid: mass identification in public spaces, inferring emotion/age/ethnicity/sexuality from faces, scraping face galleries.

## Legal and privacy cautions
- Face templates/embeddings used to identify a person are biometric data: special-category data under GDPR Art. 9 (generally prohibited unless an exception such as explicit consent applies); Illinois BIPA, Texas CUBI, Washington and similar US laws require notice, written consent and retention schedules, and BIPA has private-action damages; other jurisdictions (UK, Canada, India DPDP, Pakistan, Gulf states, China PIPL) have their own rules. This is not legal advice; get counsel.
- EU AI Act (Regulation 2024/1689): prohibits certain practices, including untargeted scraping of facial images to build databases, emotion inference in workplaces and schools, and real-time remote biometric identification in public spaces by law enforcement (narrow exceptions); biometric identification systems are otherwise high-risk with conformity duties. Verify current obligations and dates at the EUR-Lex text.
- Obtain informed, specific, revocable consent; state purpose, retention and who has access; offer an alternative; minimise (store embeddings not photos, delete on request, short retention); run a DPIA.
- Bias: error rates vary by skin tone, gender and age; measure per group and set thresholds accordingly. Never use as sole basis for decisions with legal or similar effects.
- Security: embeddings are irrevocable identifiers; encrypt, restrict access, and never leave them in logs or analytics.

## Setup & code
```bash
pip install insightface onnxruntime opencv-python    # detection + recognition models download on first use
```
```python
import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis()                    # default pack; check the model licence terms before commercial use
app.prepare(ctx_id=-1)                  # -1 = CPU, 0 = first GPU
faces = app.get(cv2.imread("photo.jpg"))
for f in faces:
    print(f.bbox.astype(int).tolist(), f.det_score)     # f.normed_embedding exists for recognition
```
Pose with Ultralytics: load a pose-variant weight from the docs (e.g. a `-pose.pt` model), `results = model("img.jpg")`, then `results[0].keypoints.xy`. MediaPipe Pose Landmarker (`pip install mediapipe`) runs well on CPU and mobile.

## Choosing / trade-offs
- Licences: InsightFace code is permissive but its pretrained models are restricted to non-commercial use per its README; check before shipping. Commercial face APIs exist but move the same legal duties to you as controller.
- MediaPipe: light, mobile/web friendly, single-person focus; Ultralytics pose: multi-person, real time; AGPL caveat ([[object-detection]]).
- On-device processing avoids transmitting biometrics ([[edge-on-device]]).
- Accuracy on masks, low light, angles and children is much worse; test in your conditions.

## Gotchas
- A threshold tuned on a benchmark gives very different false-match rates in a larger gallery (1:N error grows with N).
- Photos, screens and masks fool systems without liveness detection.
- Datasets built from scraped faces carry legal and ethical risk.
- Pose keypoints are noisy on occluded joints; smooth over time and use confidence gating.
- Storing raw face crops "just in case" creates breach liability.
- Sending face images to a third-party API/VLM is a data transfer of biometric data ([[vision-language-models]]).

## Related
- [[video-analytics]] - tracking and pose-based activity.
- [[object-detection]] - person detection without identity.
- [[ai-security-privacy-compliance]] - broader compliance guidance.
- [[guardrails-and-safety]] - responsible-use controls.
- [[edge-on-device]] - local processing.

## References
- https://github.com/deepinsight/insightface
- https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- https://docs.ultralytics.com/tasks/pose/
- https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- https://gdpr-info.eu/art-9-gdpr/
