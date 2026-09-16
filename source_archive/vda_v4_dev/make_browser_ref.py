#!/usr/bin/env python3
"""Regenerate the python ONNX reference on the browser's EFFECTIVE frame sequence.

The preview browser's seek lands one frame early on this container (grab(i) =
src[max(0, i-1)]) — same mapping the app's own seekTo/frameTime uses, so all
in-app engines share it. Like-for-like validation = run the python pipeline on
that sequence.
"""
import os, sys, time

import numpy as np
import torch

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(S, 'Video-Depth-Anything'))
sys.path.insert(0, S)

from video_depth_anything.motion_module import attention as mm_attn
from export_vda import _attention_clean, build_model
from validate_vda import read_clip, OnnxForwardShim, CLIP, ENC_ONNX, HEAD_ONNX

mm_attn.CrossAttention._attention = _attention_clean

frames, fps = read_clip(CLIP)
eff = np.array([max(0, i - 1) for i in range(len(frames))])
frames_eff = frames[eff]
print(f'== effective sequence: {frames_eff.shape} (indices {eff[:4]}...{eff[-2:]})')

model = build_model()
model.forward = OnnxForwardShim(ENC_ONNX, HEAD_ONNX)
t0 = time.time()
with torch.no_grad():
    depths, _ = model.infer_video_depth(frames_eff, fps, input_size=518, device='cpu', fp32=True)
print(f'== python onnx reference on effective sequence: {depths.shape} in {time.time()-t0:.0f}s')

n = len(depths)
np.savez_compressed(os.path.join(S, 'validate_out', 'depths_browser_ref.npz'),
                    ref=depths[[0, n // 2, n - 1]])
print('== saved validate_out/depths_browser_ref.npz (frames 0, 42, 83 of effective sequence)')
