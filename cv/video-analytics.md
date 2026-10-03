---
title: Video analytics
category: cv
tags: [video, tracking, counting, rtsp, opencv, action-recognition, ultralytics, surveillance]
use_cases:
  - "count people or vehicles crossing a line from CCTV streams"
  - "detect safety violations or intrusions in real time from IP cameras"
  - "track objects across frames and measure dwell time in zones"
  - "recognise actions or activities in video clips"
  - "summarise or search long videos with a vision-language model"
status: draft
last_verified: 2026-10-03
sources:
  - https://docs.ultralytics.com/modes/track/
  - https://docs.opencv.org/4.x/dd/de7/group__videoio.html
  - https://github.com/roboflow/supervision
---

# Video analytics

## Summary
Video analytics turns camera streams or files into events and metrics: detection per frame, tracking to give objects persistent IDs, then logic on top (line crossing, zone dwell, counting, alerts). Action recognition and video-language models add activity understanding. Most real-world cost and difficulty is in the stream plumbing, latency and false alarms, not the model.

## Key concepts
- Pipeline: capture (RTSP/file/webcam) -> decode -> detect -> track -> rules/analytics -> alert/store.
- Multi-object tracking (MOT): ByteTrack, BoT-SORT and similar link detections across frames; Ultralytics `track` mode bundles them.
- Zones and lines: polygon/line geometry on tracked centroids gives counts, dwell time, direction; the `supervision` library has ready-made annotators and zone counters.
- Frame skipping and resolution: process every Nth frame or lower resolution to hit real-time throughput.
- Action recognition: clip-based models (e.g. video transformers, SlowFast) classify activities; pose-based approaches use skeleton sequences ([[face-and-pose]]).
- Video-language models and VLMs on sampled frames answer questions about footage ([[vision-language-models]]).
- Metrics: detection mAP, tracking (MOTA/IDF1), and operational: false alerts per camera per day, latency.

## When to use / scenarios
- Retail: footfall and queue length; Traffic/smart city: vehicle counting, parking occupancy; Industrial safety: PPE and restricted-zone alerts; Sports: player tracking; Agriculture/livestock monitoring.
- Search over recorded video: sample frames, embed or caption them, index ([[embedding-models]]).
- Not for: identifying individuals by face without legal basis (see [[face-and-pose]]); tasks where a periodic still photo is enough.

## Setup & code
```bash
pip install -U ultralytics opencv-python
```
```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")      # current family per Ultralytics docs; confirm before pinning
# source may be a file, webcam index, or an rtsp:// URL
for r in model.track(source="traffic.mp4", stream=True, persist=True, classes=[2, 5, 7]):
    for b in r.boxes:
        if b.id is not None:
            print(int(b.id), r.names[int(b.cls)], b.xyxy.tolist())
```
`stream=True` yields results frame by frame instead of loading the whole video into memory. Add line/zone counting with `supervision` (`pip install supervision`; see its docs for `LineZone`/`PolygonZone`).

## Choosing / trade-offs
- Edge (Jetson, camera-side) vs server: edge cuts bandwidth and latency and keeps video local; server scales GPUs across many streams ([[edge-on-device]]).
- Real-time vs batch: real-time needs small models, frame skipping, TensorRT; batch analysis can use larger models.
- Per-frame VLM calls are too expensive for live streams; use detection to trigger a VLM only on events.
- Tracker choice: ByteTrack-style for speed; appearance re-ID trackers for crowded scenes and occlusions.
- Storage: keep event clips and metadata, not continuous raw video, to control cost and privacy.
- Licensing of the detector (AGPL for Ultralytics) matters for products ([[object-detection]]).

## Gotchas
- RTSP streams drop and stall; implement reconnects and a bounded frame queue so latency does not grow.
- Reading frames slower than the camera rate builds lag; drop old frames and read the latest.
- ID switches under occlusion corrupt counts; place counting lines where objects are separated and unobstructed.
- Camera angle, night IR and weather shift accuracy; test per camera and fine-tune on site data.
- Double counting at line crossing needs debouncing and direction logic.
- Privacy law: video of people is personal data; signage, retention limits, access control and DPIA/legal review are often required.
- GPU decoding (NVDEC) often matters as much as the model for many-stream throughput.

## Related
- [[object-detection]] - the per-frame detector.
- [[face-and-pose]] - pose and face with legal cautions.
- [[vision-language-models]] - event explanation and video Q&A.
- [[edge-on-device]] - on-camera deployment.
- [[manufacturing-iot]] - industrial monitoring scenarios.

## References
- https://docs.ultralytics.com/modes/track/
- https://github.com/roboflow/supervision
- https://docs.opencv.org/4.x/dd/de7/group__videoio.html
