# VDA-Small → ONNX (TheoreticallyPose V4 depth engine)

**Status: ALL PHASES COMPLETE (2026-07-07).** Exported, browser-validated, integrated
into theoreticallypose.html as the "Consistent (VDA)" depth engine, and flicker-A/B'd
on lossless PNG-seq. Ship config = the two `*_fp16.onnx` (59.4 MB total).

## Bottom line (84f 960×544 clip, M4 Pro, Claude Preview Electron)
- **Bake: ~16 s** on WebGPU fp16 (encoder 94 ms/frame, temporal head ~1.3 s/window,
  4 windows) — ~2.6× FASTER than the per-frame DAv2 "Fast" engine (~42 s).
- **Flicker (lossless PNG-seq, top-90-rows static ROI, vs raw DAv2):**
  raw grays: VDA −75% bg shimmer at 101% edge energy; V3 stack (DAv2+stabilize+smooth2)
  −62% at 51% edge energy. Contrast-normalized (each seq scaled to equal global std —
  the honest cross-config view since stabilized display mapping changes gray scale):
  VDA −65% shimmer / −85% pulse with stabilize, vs V3 stack only −22% shimmer —
  i.e. much of the V3 stack's raw-gray win was contrast softening, VDA's is real.
- Stabilize on VDA output costs zero real edge energy and flattens residual global
  pulse (−85%) → recommended default: VDA + stabilize ON, smooth to taste.
- MP4 export of VDA depth: 466 KB for 3.5 s (h264 84f @24 exact) — temporally
  stable depth compresses absurdly well.

## Phase 2/3 gotchas (hard-won, keep)
- **WebGPU-vs-CPU numerics:** browser WebGPU output correlates r=0.996–0.999 with
  python ORT CPU on IDENTICAL pixels (~2–2.7% RMSE of range) — Dawn compiles Metal
  shaders with fast-math; deterministic run-to-run, not a bug, invisible after u8
  quantization. ort-web wasm EP was abandoned: single-threaded w/o COOP/COEP, it
  froze the renderer for minutes per window (~100× slower). VDA engine requires WebGPU.
- **fp16 conversion:** onnxconverter-common chokes on identity Cast(float→float) nodes
  the tracer emits (`.to(dtype)` no-ops) — session creation fails with a type mismatch.
  Strip them first (see the surgery in this repo's history / re-run export then
  convert). onnxsim segfaults on these graphs — don't bother.
- **Browser seek offset:** Electron/Chrome shows frame i−1 at t=(i+0.5)/24 for this
  container (PTS/edit-list shift) and duplicates frame 0 — cv2 sequential read
  disagrees with browser seeks by one frame. IRRELEVANT in-app (all engines + pose
  share the same seekTo convention) but validation must compare like-for-like
  sequences. Decode IS deterministic across runs.
- **Flicker-metric contrast trap:** stabilize/smooth render through the ALIGNED global
  range (sgmin/sgmax ≠ gmin/gmax) → per-config gray contrast differs → raw edge/shimmer
  numbers partly measure contrast, not noise. Always ALSO report contrast-normalized
  metrics (scale each seq to equal global std).
- In-app JS heap during VDA bake ≈ 1 GB (fp32 head-input buffers 473 MB + feature
  cache; fp16 graph I/O stays fp32 by design). Fine on 48 GB; revisit fp16 I/O if
  end users complain.

## Integration (theoreticallypose.html — live V4)
Depth layer → engine select: "Fast (per-frame)" (DAv2-S, default, exact V3 behavior)
vs "Consistent (VDA)". VDA fills the same {u8,min,max} store → stabilize/smooth/
invert/import/export untouched. Engine switch resets the store (value spaces differ).
Models load from VDA.BASES (relative `vda_v4/onnx_out/` first, then localhost:8123,
then the public HF repo). PUBLIC HOSTING IS LIVE (2026-07-07):
https://huggingface.co/TheoreticallyTim/theoreticallypose-vda (apache-2.0 model card,
attribution + LICENSE included; upload staging kept in `hf_upload/`). Audience path
verified end-to-end: local sources hidden → app downloaded both models from HF and
completed a full Consistent bake. Distribution = share theoreticallypose.html itself;
a double-clicked file:// copy reaches HF + CDNs fine (verified CORS from file:// origin).

## Files added in Phases 2–4
- `vda_harness.html` — standalone browser test harness (drove all Phase 2 probes)
- `flicker_ab.py` — the A/B metrics (bg-ROI shimmer/pulse decomposition + edge energy)
- `compare_browser.py`, `make_browser_ref.py` — browser-vs-python validation tooling
- `tp_test_server.py` — Desktop static server + PUT /upload receiver (port 8123)
- `validate_out/export_vda.mp4`, `export_eyeball_f42.png` — evidence

---

# Phase 1 record (export + python validation)

**Phase 1 PASSED (2026-07-07).** Video-Depth-Anything-Small exports to ONNX and
validates bit-close against PyTorch on a real clip.

## License gate
Small model = **Apache-2.0** (repo LICENSE + README: "Video-Depth-Anything-Small model is
under the Apache-2.0 license"). Base/Large are CC-BY-NC-4.0 — **never ship those**.

## Artifacts
- `onnx_out/vda_vits_encoder_518x910.onnx` (88.3 MB fp32) — per-frame DINOv2-S.
  Input `pixels` [1,3,518,910] (ImageNet-normalized RGB). Outputs `feat1..feat4`
  [1,2405,384] (ViT layers 2/5/8/11, final-norm applied; cls tokens dropped — head
  doesn't use them).
- `onnx_out/vda_vits_head_T32_37x65.onnx` (27.9 MB fp32) — temporal DPT head, T=32 baked.
  Inputs `feat1..feat4` [32,2405,384] (32 frames' encoder features concatenated along
  axis 0). Output `depth` [32,1,518,910], ReLU applied in-graph.
- Opset 17, legacy TorchScript exporter (`dynamo=False`), `do_constant_folding=True`,
  all shapes static (chosen for WebGPU EP friendliness).
- 518×910 = exactly what VDA's own preprocessing picks for a 960×544 clip
  (lower-bound resize to 518, multiple-of-14). Other sizes: re-run `export_vda.py --h H --w W`
  (~2 min, self-contained in this folder; needs torch+onnx+onnxruntime+einops+easydict+torchvision).

## Validation (validate_vda.py, tp_test_2char.mp4, 84f 960×544 @24fps)
Full `infer_video_depth` pipeline run twice — identical preprocessing, 32-frame windows,
OVERLAP=10, KEYFRAMES stitching — torch fp32 CPU vs ONNX (ort CPU) forward:
- per-frame Pearson r: **min 1.000000**; PSNR (peak = ref range): **min 120.8 dB**
- global max|diff| 1.3e-3 on 22.2 depth range (rel 6e-5)
- eyeball strips in `validate_out/` (ref | onnx | 10× diff — diff panel is black)
- CPU M4 Pro timings: torch 77s, ONNX 64s for 84 frames (4 windows; per window:
  enc 12.2s = 0.38s/frame, head 3.6s)

## Export gotchas handled (in export_vda.py)
- `CrossAttention._attention` uses `baddbmm(torch.empty(...), beta=0)` — the ONNX
  decomposition can materialize the empty tensor (NaN/Inf propagation). Patched to plain
  bmm·scale→softmax→bmm before export; verified exactly equivalent in torch (diff 0.0).
- `micro_batch_size` chunking in `DPTHeadTemporal.forward`: pass `micro_batch_size=T`
  to trace the single-path branch.
- xformers absent ⇒ all attention takes vanilla fallbacks automatically (that's what
  makes export possible); do NOT install xformers in the export env.
- Positional encoding buffers (APE sinusoid, max_len=32) live in the checkpoint and
  trace to constants at fixed T.

## Phase 2 planning facts
- Windowing to replicate in JS (from `video_depth.py`): INFER_LEN=32, OVERLAP=10,
  KEYFRAMES=[0,12,24..31], INTERP_LEN=8, scale/shift alignment on the first
  OVERLAP−INTERP_LEN=2 keyframe slots + linear crossfade over the next 8
  (`get_interpolate_frames`), then append frames 10..31. Window slot 0 always carries
  **source frame 0** (global scale anchor) — overlap slots are copies of *source frames*,
  so per-frame encoder features are cacheable across windows: encoder runs = unique
  frames, head runs = ceil((F−32)/22)+1.
- Head input cache per window: 4×[32,2405,384] fp32 ≈ **473 MB** (fp16 ≈ 236 MB) at
  518×910. If WebGPU memory is tight, drop processing res (e.g. 378×658) or fp16 the
  features.
- Model download for users: 116 MB fp32 total today; fp16 conversion (Phase 2) ≈ 58 MB —
  comparable to the current DAv2-S (~94 MB).
- Suspect ops for WebGPU EP fallback profiling: bicubic Resize (pos-embed interp is
  constant-folded away — verify), Erf (GELU), GroupNorm/LayerNorm, the [19240,32,32]
  batched MatMuls in temporal attention.
