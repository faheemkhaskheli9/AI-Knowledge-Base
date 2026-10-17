---
title: 3D generation models (text/image to mesh and splats)
category: models
tags: [3d-generation, image-to-3d, text-to-3d, mesh, gaussian-splatting, gltf, glb, pbr, trellis, hunyuan3d, game-assets]
use_cases:
  - "turn concept art into textured 3D props for a Unity or Godot game"
  - "generate product 3D models from photos for an e-commerce viewer"
  - "prototype a level with AI-generated placeholder assets"
  - "self-host an image-to-3D model with a licence that allows commercial use"
status: draft
last_verified: 2026-10-15
sources:
  - https://github.com/microsoft/TRELLIS.2
  - https://huggingface.co/microsoft/TRELLIS.2-4B
  - https://github.com/microsoft/TRELLIS
  - https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
  - https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/LICENSE
  - https://huggingface.co/stabilityai/stable-fast-3d
  - https://stability.ai/license
  - https://github.com/VAST-AI-Research/TripoSR
  - https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
  - https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md
  - https://docs.unity3d.com/Packages/com.unity.cloud.gltfast@6.11/manual/
---

# 3D generation models (text/image to mesh and splats)

## Summary
3D generation models turn an image (or text, via a text-to-image step) into a
3D asset: a textured triangle mesh, or a 3D Gaussian splat. Open models such as
Microsoft TRELLIS.2, Tencent Hunyuan3D 2.1, Stability's SF3D and TripoSR export
meshes (usually GLB) that game engines and web viewers import directly. Output
is good for props and prototypes; hero assets still need an artist's cleanup.

## Key concepts
- Most pipelines are image-to-3D. Text-to-3D is usually text-to-image
  ([[image-generation-models]]) followed by image-to-3D, which also lets you
  pick the best concept image first.
- Two stages are common: shape (geometry) generation, then texture/material
  generation. Hunyuan3D 2.1 ships separate shape (3.3B) and paint (2B) models.
- Output representations:
  - Mesh + UV texture: what engines want. PBR materials (base colour,
    roughness, metallic) look right under engine lighting.
  - 3D Gaussian splats: millions of coloured blobs; photoreal for captured
    scenes but no clean surfaces for physics or editing. Standardised in glTF
    as `KHR_gaussian_splatting` (ratified per the extension README).
  - Radiance fields (NeRF-style): mostly an intermediate, baked to mesh.
- Interchange format: glTF 2.0 / GLB (binary glTF) is the default; it carries
  mesh, PBR materials and textures in one file. OBJ/FBX/USD are alternatives.
- Single-image input is ambiguous: the back of the object is invented, so
  multi-view or clean front-on images on a plain background work best.

Open models as of 2026-10-15 (sources: each repo / model card):

| Model | Input -> output | Licence (summary) |
|---|---|---|
| TRELLIS.2-4B (Microsoft) | image -> mesh with PBR, GLB export | MIT (deps such as nvdiffrast have own licences) |
| TRELLIS (v1, Microsoft) | image/text -> Gaussians, radiance field or mesh | MIT (some submodules differ) |
| Hunyuan3D 2.1 (Tencent) | image -> mesh + PBR texture | Tencent Hunyuan 3D 2.1 Community License: excludes EU, UK, South Korea; >1M MAU needs a licence |
| Stable Fast 3D (Stability) | image -> UV-unwrapped textured mesh | Stability AI Community License: free under US$1M annual revenue |
| SPAR3D (Stability) | image -> point cloud -> editable mesh | Stability AI Community License |
| TripoSR (Tripo AI + Stability) | image -> mesh | MIT |

Hosted text/image-to-3D APIs and apps also exist (for example Meshy and Tripo);
Their current models, formats and prices are not covered here; read the vendor's docs.

## When to use / scenarios
- Game studios: background props, greybox placeholders, game-jam assets;
  then retopologise and fix UVs for anything the player looks at closely.
- E-commerce and AR: quick 3D previews from product photos for a web viewer
  (`<model-viewer>`, three.js) or AR Quick Look (convert GLB to USDZ).
- Architecture/interiors: furniture and decor fillers for visualisation.
- Real places or real objects you can photograph from many angles: use
  photogrammetry or Gaussian-splat capture instead of single-image generation.
- Not for: precise CAD parts with tolerances (use parametric CAD), rigged
  animated characters out of the box (rigging is a separate step), or anything
  where the licence territory excludes you (Hunyuan3D in the EU/UK).

## Setup & code
Pipeline: (1) text-to-image or a clean photo; (2) background removal;
(3) image-to-3D model (GPU, see the repo README for the CUDA stack);
(4) decimate/retopologise and check scale in Blender; (5) export GLB;
(6) import into the engine. Godot imports `.glb` natively, Unity uses the
glTFast package (default importer for `.gltf`/`.glb` once installed), and
Unreal has a glTF importer.

Running the models needs a CUDA GPU and large downloads, so see each repo
(TRELLIS.2 exports `sample.glb` through its `o_voxel` post-processing step;
TripoSR runs `python run.py examples/chair.png --output-dir output/`).

The block below (runs offline, numpy + stdlib) writes a valid minimal GLB, the
format every model above exports, so you can test an import pipeline or
validator without a GPU:
```python
import json, struct
import numpy as np

# One triangle, metres, Y-up (glTF convention)
pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
idx = np.array([0, 1, 2], dtype=np.uint16)

def pad4(b, fill=b"\x00"):
    return b + fill * (-len(b) % 4)

bin_chunk = pad4(idx.tobytes()) + pos.tobytes()
gltf = {
    "asset": {"version": "2.0"},
    "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
    "meshes": [{"primitives": [{"attributes": {"POSITION": 1}, "indices": 0, "material": 0}]}],
    "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.3, 0.2, 1], "metallicFactor": 0, "roughnessFactor": 0.8}}],
    "buffers": [{"byteLength": len(bin_chunk)}],
    "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": idx.nbytes, "target": 34963},
        {"buffer": 0, "byteOffset": len(pad4(idx.tobytes())), "byteLength": pos.nbytes, "target": 34962},
    ],
    "accessors": [
        {"bufferView": 0, "componentType": 5123, "count": 3, "type": "SCALAR"},
        {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3",
         "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
    ],
}
json_chunk = pad4(json.dumps(gltf).encode(), b" ")
glb = b"".join([
    struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)),
    struct.pack("<I4s", len(json_chunk), b"JSON"), json_chunk,
    struct.pack("<I4s", len(bin_chunk), b"BIN\x00"), bin_chunk,
])
with open("triangle.glb", "wb") as f:
    f.write(glb)

# Read it back: header check and triangle count, the kind of gate to run on generated assets
magic, version, length = struct.unpack_from("<4sII", glb, 0)
jlen = struct.unpack_from("<I", glb, 12)[0]
doc = json.loads(glb[20:20 + jlen])
tris = doc["accessors"][doc["meshes"][0]["primitives"][0]["indices"]]["count"] // 3
print(magic, version, length == len(glb), "triangles:", tris)
```

## Choosing / trade-offs
- Licence first: MIT (TRELLIS, TripoSR) is simplest for commercial games;
  Stability's licence caps free use by revenue; Hunyuan3D's excludes whole
  regions ([[model-licenses]]).
- Quality vs speed: SF3D and TripoSR are fast feed-forward models (sub-second
  per asset on a GPU per their authors) with simpler geometry; TRELLIS.2 and
  Hunyuan3D 2.1 give higher detail and PBR materials at more GPU time.
- Mesh vs splats: meshes for interactive games, physics and editing; splats
  for photoreal viewing of captured scenes.
- Self-host vs hosted API: self-hosting keeps concept art private and has no
  per-asset fee; APIs avoid the CUDA setup ([[gpu-cloud-options]]).

## Gotchas
- Polygon counts are often far too high for real-time use; budget and decimate.
- Generated UVs and topology are messy: fine for static props, poor for
  deformation and rigging.
- Lighting is often baked into the albedo texture, so assets look wrong under
  engine lighting; prefer models that output separate PBR maps.
- Scale and orientation vary; glTF is Y-up in metres, so normalise before
  import.
- TRELLIS.2's GLB is exported in OPAQUE mode; transparency in the texture alpha
  must be enabled manually in your 3D tool.
- Check the licence of your input images too: generating from someone else's
  concept art does not clear their rights.

## Related
- [[image-generation-models]] - the text-to-image step before image-to-3D.
- [[diffusion-models]] - the generative backbone of most 3D generators.
- [[multimodal-models]] - other cross-modal generation.
- [[model-licenses]] - MIT vs community licences with revenue/territory limits.
- [[gpu-cloud-options]] - renting a GPU to run the open models.

## References
- https://github.com/microsoft/TRELLIS.2
- https://huggingface.co/microsoft/TRELLIS.2-4B
- Xiang et al., "Structured 3D Latents for Scalable and Versatile 3D Generation" (CVPR 2025), https://github.com/microsoft/TRELLIS
- https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- https://huggingface.co/tencent/Hunyuan3D-2.1/blob/main/LICENSE
- https://huggingface.co/stabilityai/stable-fast-3d
- https://stability.ai/news-updates/stable-point-aware-3d
- https://stability.ai/license
- Tochilkin et al., "TripoSR: Fast 3D Object Reconstruction from a Single Image" (2024), https://arxiv.org/abs/2403.02151
- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023), https://arxiv.org/abs/2308.04079
- https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md
- https://docs.unity3d.com/Packages/com.unity.cloud.gltfast@6.11/manual/
- https://docs.godotengine.org/en/stable/tutorials/assets_pipeline/importing_3d_scenes/available_formats.html
