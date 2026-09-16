#!/usr/bin/env python3
"""Phase 4: flicker A/B on lossless PNG-seq exports (V3 methodology).

Configs: vda_raw (V4, stabilize off/smooth 0), vda_v3stack (V4 + V3 filters),
dav2_raw (V2 baseline), dav2_v3 (V3 best: stabilize + smooth 2).

Metrics (per V3 session methodology):
- flicker decomposition on gray frames: global pulse = mean|Δmean_t|,
  per-pixel shimmer = mean_t mean_px |Δ_t - Δmean_t|
- background-ROI shimmer: same, restricted to rows the people never enter
  (auto-picked as the lowest-motion rows on the SOURCE clip)
- edge-energy retention: mean Sobel magnitude vs the dav2_raw baseline
  (smoothing that kills edges would show up here)
"""
import io, os, zipfile

import cv2
import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
UP = os.path.join(S, 'uploads')
CONFIGS = ['vda_raw', 'vda_v3stack', 'dav2_raw', 'dav2_v3']


def load_seq(tag):
    zf = zipfile.ZipFile(os.path.join(UP, f'export_{tag}.zip'))
    names = sorted(n for n in zf.namelist() if n.endswith('.png'))
    frames = []
    for n in names:
        arr = cv2.imdecode(np.frombuffer(zf.read(n), np.uint8), cv2.IMREAD_GRAYSCALE)
        frames.append(arr.astype(np.float32))
    return np.stack(frames)


def pick_bg_rows(strip_h=90):
    """Rows with least motion on the SOURCE clip = static background band."""
    cap = cv2.VideoCapture('/Users/tim/Desktop/tp_test_2char.mp4')
    prev, motion = None, None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            d = np.abs(g - prev)
            motion = d if motion is None else motion + d
        prev = g
    cap.release()
    row_motion = motion.mean(axis=1)
    best = min(range(0, len(row_motion) - strip_h), key=lambda y: row_motion[y:y + strip_h].mean())
    return best, best + strip_h, row_motion


def metrics(seq, y0=None, y1=None):
    d = np.diff(seq, axis=0)                     # [T-1, H, W]
    if y0 is not None:
        d = d[:, y0:y1, :]
    dmean = d.mean(axis=(1, 2), keepdims=True)
    pulse = float(np.abs(dmean).mean())
    shimmer = float(np.abs(d - dmean).mean())
    return pulse, shimmer


def edge_energy(seq):
    e = 0.0
    for f in seq:
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        e += float(np.sqrt(gx * gx + gy * gy).mean())
    return e / len(seq)


def main():
    seqs = {}
    for tag in CONFIGS:
        seqs[tag] = load_seq(tag)
        print(f'{tag}: {seqs[tag].shape}')
    shape0 = next(iter(seqs.values())).shape
    assert all(s.shape == shape0 for s in seqs.values()), 'shape mismatch across configs'

    # scale ROI rows from source height (544) to export height
    y0s, y1s, _ = pick_bg_rows()
    H = shape0[1]
    y0, y1 = int(y0s * H / 544), int(y1s * H / 544)
    print(f'background ROI: source rows {y0s}-{y1s} -> export rows {y0}-{y1}')

    base_edge = edge_energy(seqs['dav2_raw'])
    print(f'\n{"config":14s} {"full pulse":>11s} {"full shim":>10s} {"bg pulse":>9s} {"bg shim":>8s} {"edgeE":>7s} {"edge%":>6s}')
    results = {}
    for tag in CONFIGS:
        fp, fs = metrics(seqs[tag])
        bp, bs = metrics(seqs[tag], y0, y1)
        ee = edge_energy(seqs[tag])
        results[tag] = dict(full_pulse=fp, full_shimmer=fs, bg_pulse=bp, bg_shimmer=bs,
                            edge_energy=ee, edge_retention=ee / base_edge)
        print(f'{tag:14s} {fp:11.4f} {fs:10.4f} {bp:9.4f} {bs:8.4f} {ee:7.2f} {100*ee/base_edge:5.1f}%')

    print('\nrelative to dav2_raw (V2 baseline):')
    for tag in CONFIGS:
        r = results[tag]
        b = results['dav2_raw']
        print(f"{tag:14s} bg shimmer {100*(r['bg_shimmer']/b['bg_shimmer']-1):+6.1f}%   "
              f"full shimmer {100*(r['full_shimmer']/b['full_shimmer']-1):+6.1f}%   "
              f"bg pulse {100*(r['bg_pulse']/b['bg_pulse']-1):+6.1f}%")
    np.savez(os.path.join(S, 'validate_out', 'flicker_ab.npz'), **{f'{k}_{m}': v for k, r in results.items() for m, v in r.items()})


if __name__ == '__main__':
    main()
