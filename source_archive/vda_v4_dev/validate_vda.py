#!/usr/bin/env python3
"""Phase 1 gate: validate split ONNX export against PyTorch on a real clip.

Runs VideoDepthAnything.infer_video_depth twice — identical preprocessing,
windowing (INFER_LEN=32/OVERLAP=10/KEYFRAMES) and cross-window stitching —
once with the torch forward, once with a forward shim that calls the two ONNX
graphs (per-frame encoder -> stacked features -> T=32 temporal head).
"""
import os, sys, time

import cv2
import numpy as np
import torch

SCRATCH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(SCRATCH, 'Video-Depth-Anything')
sys.path.insert(0, REPO)

from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.motion_module import attention as mm_attn
from export_vda import _attention_clean, build_model, VITS

CLIP = '/Users/tim/Desktop/tp_test_2char.mp4'
ENC_ONNX = os.path.join(SCRATCH, 'onnx_out', 'vda_vits_encoder_518x910.onnx')
HEAD_ONNX = os.path.join(SCRATCH, 'onnx_out', 'vda_vits_head_T32_37x65.onnx')
OUT_DIR = os.path.join(SCRATCH, 'validate_out')


def read_clip(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames), fps


class OnnxForwardShim:
    """Drop-in replacement for VideoDepthAnything.forward backed by the two ONNX graphs."""

    def __init__(self, enc_path, head_path):
        import onnxruntime as ort
        self.enc = ort.InferenceSession(enc_path, providers=['CPUExecutionProvider'])
        self.head = ort.InferenceSession(head_path, providers=['CPUExecutionProvider'])
        self.enc_time = 0.0
        self.head_time = 0.0
        self.windows = 0

    def __call__(self, x):  # x [1,T,3,H,W] float32, already ImageNet-normalized
        B, T, C, H, W = x.shape
        assert B == 1
        pixels = x.flatten(0, 1).numpy()
        feats = [[], [], [], []]
        t0 = time.time()
        for i in range(T):
            out = self.enc.run(None, {'pixels': pixels[i:i + 1]})
            for j in range(4):
                feats[j].append(out[j])
        self.enc_time += time.time() - t0
        stacked = {f'feat{j + 1}': np.concatenate(feats[j], axis=0) for j in range(4)}
        t0 = time.time()
        depth = self.head.run(None, stacked)[0]  # [T,1,H,W], relu applied in-graph
        self.head_time += time.time() - t0
        self.windows += 1
        return torch.from_numpy(depth).squeeze(1).unsqueeze(0)  # [1,T,H,W]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mm_attn.CrossAttention._attention = _attention_clean

    frames, fps = read_clip(CLIP)
    print(f'== clip: {frames.shape} @ {fps:.3f} fps')

    model = build_model()

    t0 = time.time()
    with torch.no_grad():
        ref, _ = model.infer_video_depth(frames, fps, input_size=518, device='cpu', fp32=True)
    t_ref = time.time() - t0
    print(f'== torch reference: {ref.shape} in {t_ref:.0f}s')

    shim = OnnxForwardShim(ENC_ONNX, HEAD_ONNX)
    model.forward = shim  # instance attribute shadows the method for self.forward(...)
    t0 = time.time()
    with torch.no_grad():
        got, _ = model.infer_video_depth(frames, fps, input_size=518, device='cpu', fp32=True)
    t_onnx = time.time() - t0
    print(f'== onnx pipeline:   {got.shape} in {t_onnx:.0f}s '
          f'({shim.windows} windows; enc {shim.enc_time / shim.windows:.1f}s + head {shim.head_time / shim.windows:.1f}s per window)')

    assert ref.shape == got.shape
    n = ref.shape[0]
    rng = float(ref.max() - ref.min())
    pearsons, psnrs = [], []
    for i in range(n):
        a, b = ref[i].ravel(), got[i].ravel()
        r = float(np.corrcoef(a, b)[0, 1])
        rmse = float(np.sqrt(np.mean((a - b) ** 2)))
        psnr = 20 * np.log10(rng / rmse) if rmse > 0 else float('inf')
        pearsons.append(r)
        psnrs.append(psnr)
    pearsons, psnrs = np.array(pearsons), np.array(psnrs)
    print(f'== per-frame Pearson r: min {pearsons.min():.6f}  mean {pearsons.mean():.6f}')
    finite = psnrs[np.isfinite(psnrs)]
    print(f'== per-frame PSNR (peak=ref range {rng:.2f}): min {psnrs.min():.1f} dB  mean {(finite.mean() if len(finite) else float("inf")):.1f} dB')
    print(f'== global max|diff|: {np.abs(ref - got).max():.4e}  (rel {np.abs(ref - got).max() / rng:.2e})')

    # eyeball strips: ref | onnx | 10x abs diff, normalized to ref min/max
    lo, hi = ref.min(), ref.max()
    for idx in [0, n // 2, n - 1]:
        a = (np.clip((ref[idx] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        b = (np.clip((got[idx] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        d = (np.clip(np.abs(ref[idx] - got[idx]) / (hi - lo) * 10, 0, 1) * 255).astype(np.uint8)
        strip = np.concatenate([a, b, d], axis=1)
        cv2.imwrite(os.path.join(OUT_DIR, f'compare_f{idx:03d}.png'), strip)
    print(f'== eyeball strips in {OUT_DIR}')

    ok = pearsons.min() > 0.995 and psnrs.min() > 40.0
    print(f'== VERDICT: {"PASS" if ok else "FAIL"} (gate: min r > 0.995 and min PSNR > 40 dB)')
    np.savez_compressed(os.path.join(OUT_DIR, 'depths.npz'),
                        ref=ref[[0, n // 2, n - 1]], onnx=got[[0, n // 2, n - 1]])
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
