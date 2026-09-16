---
license: apache-2.0
base_model: depth-anything/Video-Depth-Anything-Small
tags:
- depth-estimation
- video
- onnx
- webgpu
library_name: onnxruntime
---

# Video Depth Anything (Small) — ONNX for TheoreticallyPose

Browser-ready ONNX export of [Video-Depth-Anything-Small](https://huggingface.co/depth-anything/Video-Depth-Anything-Small)
(Apache-2.0), used by the **"Consistent (VDA)" depth engine** in
[TheoreticallyPose](https://www.youtube.com/@TheoreticallyMedia) — a single-file browser
app that turns a source clip into pose / depth / silhouette control videos for
video-to-video AI generation. The app downloads these two files automatically the first
time you use the Consistent engine (~59 MB, cached by your browser).

## Files

| file | what it is | input → output |
|---|---|---|
| `vda_vits_encoder_518x910_fp16.onnx` (45 MB) | per-frame DINOv2-S encoder | `[1,3,518,910]` normalized RGB → 4 × `[1,2405,384]` ViT features |
| `vda_vits_head_T32_37x65_fp16.onnx` (14 MB) | temporal DPT head, 32-frame window | 4 × `[32,2405,384]` stacked features → `[32,1,518,910]` depth |

Conversion notes: exported with the PyTorch legacy tracer at opset 17, static shapes
(518×910, T=32), split at the ViT-token boundary so per-frame encoder features can be
cached across overlapping windows; vanilla attention throughout (no xformers);
weights converted to fp16 with fp32 graph I/O. Runs on onnxruntime-web's WebGPU
execution provider.

## Attribution & license

Original model and architecture: **Video Depth Anything** by ByteDance
([repo](https://github.com/DepthAnything/Video-Depth-Anything),
[paper](https://arxiv.org/abs/2501.12375), CVPR 2025 Highlight).
Only the **Small** variant is redistributed here, under its original
**Apache-2.0** license (see `LICENSE`). The Base/Large variants are CC-BY-NC and are
**not** included. This repo contains a format conversion (ONNX/fp16) of the original
checkpoint; no weights were retrained or fine-tuned.

```bibtex
@inproceedings{video_depth_anything,
  title={Video Depth Anything: Consistent Depth Estimation for Super-Long Videos},
  author={Chen, Sili and Guo, Hengkai and Zhu, Shengnan and Zhang, Feihu and Huang, Zilong and Feng, Jiashi and Kang, Bingyi},
  booktitle={CVPR},
  year={2025}
}
```
