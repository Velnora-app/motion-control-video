#!/usr/bin/env python3
"""Compare browser-harness VDA depths (518x910, uploaded .bin) vs python torch reference (544x960 npz)."""
import json, os

import cv2
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
UP = os.path.join(S, 'uploads')
NPZ = np.load(os.path.join(S, 'validate_out', 'depths.npz'))
ref_frames = NPZ['ref']  # [3, 544, 960] for frames 0, 42, 83

NET_H, NET_W = 518, 910
names = ['vda_browser_f000.bin', 'vda_browser_f042.bin', 'vda_browser_f083.bin']
labels = [0, 42, 83]

print(json.dumps(json.load(open(os.path.join(UP, 'vda_browser_stats.json')))['adapter']))
for i, (name, lab) in enumerate(zip(names, labels)):
    raw = np.fromfile(os.path.join(UP, name), dtype='<f4')
    assert raw.size == NET_H * NET_W, f'{name}: {raw.size}'
    browser = raw.reshape(NET_H, NET_W)
    up = cv2.resize(browser, (960, 544), interpolation=cv2.INTER_LINEAR)
    ref = ref_frames[i]
    r = float(np.corrcoef(up.ravel(), ref.ravel())[0, 1])
    # scale/shift-align browser to ref before RMSE (browser preprocessing resample differs slightly)
    A = np.stack([up.ravel(), np.ones(up.size)], axis=1)
    sol, *_ = np.linalg.lstsq(A, ref.ravel(), rcond=None)
    rmse = float(np.sqrt(np.mean((A @ sol - ref.ravel()) ** 2)))
    rng = float(ref.max() - ref.min())
    print(f'frame {lab:3d}: pearson r = {r:.6f}   aligned RMSE = {rmse:.4f} ({100*rmse/rng:.2f}% of range {rng:.2f})')
    strip = np.concatenate([
        (np.clip((ref - ref.min()) / (rng + 1e-9), 0, 1) * 255).astype(np.uint8),
        (np.clip((up - ref.min()) / (rng + 1e-9), 0, 1) * 255).astype(np.uint8),
        (np.clip(np.abs(A.reshape(544, 960, 2) @ sol - ref) / (rng + 1e-9) * 10, 0, 1) * 255).astype(np.uint8),
    ], axis=1)
    cv2.imwrite(os.path.join(S, 'validate_out', f'browser_compare_f{lab:03d}.png'), strip)
print('strips written to validate_out/browser_compare_f*.png')
