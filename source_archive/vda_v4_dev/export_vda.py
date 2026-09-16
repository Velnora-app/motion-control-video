#!/usr/bin/env python3
"""Phase 1: split ONNX export of Video-Depth-Anything-Small (Apache-2.0).

Split = per-frame DINOv2-S encoder + temporal head with fixed T=32 over cached
ViT tokens. Fixed shapes throughout (browser WebGPU EP prefers static graphs).

Usage: python3 export_vda.py [--h 518] [--w 910] [--t 32] [--out DIR] [--dynamo]
"""
import argparse, os, sys, time

import numpy as np
import torch
import torch.nn as nn

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Video-Depth-Anything')
sys.path.insert(0, REPO)

from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.motion_module import attention as mm_attn

CKPT = os.path.join(REPO, 'checkpoints', 'video_depth_anything_vits.pth')
VITS = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
LAYER_IDX = [2, 5, 8, 11]

_orig_attention = mm_attn.CrossAttention._attention

def _attention_clean(self, query, key, value, attention_mask=None):
    # baddbmm with beta=0 reads a torch.empty tensor; the ONNX decomposition can
    # materialize that empty (NaN/Inf) and propagate it. Same math, no empty.
    attention_scores = torch.bmm(query, key.transpose(-1, -2)) * self.scale
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask
    attention_probs = attention_scores.softmax(dim=-1)
    attention_probs = attention_probs.to(value.dtype)
    hidden_states = torch.bmm(attention_probs, value)
    return self.reshape_batch_dim_to_heads(hidden_states)


class EncoderWrapper(nn.Module):
    """One frame in -> 4 normalized patch-token tensors out (cls token unused: use_clstoken=False)."""
    def __init__(self, model):
        super().__init__()
        self.backbone = model.pretrained

    def forward(self, pixels):  # [1,3,H,W], ImageNet-normalized
        feats = self.backbone.get_intermediate_layers(pixels, LAYER_IDX, return_class_token=True)
        return feats[0][0], feats[1][0], feats[2][0], feats[3][0]  # each [1,N,384]


class HeadWrapper(nn.Module):
    """4 stacked feature tensors [T,N,384] -> depth [T,1,ph*14,pw*14] (relu'd, as in model.forward)."""
    def __init__(self, model, ph, pw, t):
        super().__init__()
        self.head = model.head
        self.ph, self.pw, self.t = ph, pw, t

    def forward(self, f1, f2, f3, f4):
        feats = [(f1,), (f2,), (f3,), (f4,)]
        # micro_batch_size=T -> single (non-chunked) refinenet path
        depth = self.head(feats, self.ph, self.pw, self.t, micro_batch_size=self.t)[0]
        return torch.relu(depth)


def build_model():
    model = VideoDepthAnything(**VITS)
    sd = torch.load(CKPT, map_location='cpu')
    model.load_state_dict(sd, strict=True)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, default=518)
    ap.add_argument('--w', type=int, default=910)
    ap.add_argument('--t', type=int, default=32)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'onnx_out'))
    ap.add_argument('--dynamo', action='store_true', help='use torch.onnx dynamo exporter instead of legacy tracer')
    ap.add_argument('--skip-parity-check', action='store_true')
    args = ap.parse_args()

    assert args.h % 14 == 0 and args.w % 14 == 0, 'H and W must be multiples of 14'
    ph, pw, T = args.h // 14, args.w // 14, args.t
    N = ph * pw
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    print(f'== building model (vits), input {args.h}x{args.w} -> grid {ph}x{pw} (N={N}), T={T}')
    model = build_model()

    # --- sanity: clean attention == original attention on the full forward ---
    if not args.skip_parity_check:
        small_h = small_w = 14 * 8  # 112px, T=4: fast full-graph parity probe
        x = torch.randn(1, 4, 3, small_h, small_w)
        with torch.no_grad():
            mm_attn.CrossAttention._attention = _orig_attention
            ref = model(x)
            mm_attn.CrossAttention._attention = _attention_clean
            got = model(x)
        diff = (ref - got).abs().max().item()
        print(f'== clean-attention parity: max|diff| = {diff:.3e}')
        assert diff < 1e-4, 'clean attention rewrite is NOT equivalent'
    mm_attn.CrossAttention._attention = _attention_clean

    enc = EncoderWrapper(model)
    head = HeadWrapper(model, ph, pw, T)

    enc_in = torch.randn(1, 3, args.h, args.w)
    with torch.no_grad():
        t0 = time.time()
        enc_ref = enc(enc_in)
        print(f'== torch encoder forward: {time.time()-t0:.1f}s, feature shapes {[tuple(f.shape) for f in enc_ref]}')
        head_ins = [torch.randn(T, N, 384) * 0.5 for _ in range(4)]
        t0 = time.time()
        head_ref = head(*head_ins)
        print(f'== torch head forward: {time.time()-t0:.1f}s, depth shape {tuple(head_ref.shape)}')

    enc_path = os.path.join(args.out, f'vda_vits_encoder_{args.h}x{args.w}.onnx')
    head_path = os.path.join(args.out, f'vda_vits_head_T{T}_{ph}x{pw}.onnx')

    exporter = 'dynamo' if args.dynamo else 'legacy'
    print(f'== exporting with {exporter} exporter, opset 17')

    def export(mod, inputs, path, in_names, out_names):
        t0 = time.time()
        if args.dynamo:
            torch.onnx.export(mod, inputs, path, opset_version=17,
                              input_names=in_names, output_names=out_names,
                              dynamo=True, external_data=False)
        else:
            torch.onnx.export(mod, inputs, path, opset_version=17,
                              input_names=in_names, output_names=out_names,
                              do_constant_folding=True, dynamo=False)
        print(f'   exported {os.path.basename(path)} ({os.path.getsize(path)/1e6:.1f} MB) in {time.time()-t0:.1f}s')

    export(enc, (enc_in,), enc_path, ['pixels'], ['feat1', 'feat2', 'feat3', 'feat4'])
    export(head, tuple(head_ins), head_path, ['feat1', 'feat2', 'feat3', 'feat4'], ['depth'])

    # --- validate: onnx.checker + ORT parity vs torch on the same inputs ---
    import onnx
    import onnxruntime as ort
    for p in (enc_path, head_path):
        onnx.checker.check_model(onnx.load(p))
    print('== onnx.checker passed for both graphs')

    so = ort.SessionOptions()
    enc_sess = ort.InferenceSession(enc_path, so, providers=['CPUExecutionProvider'])
    head_sess = ort.InferenceSession(head_path, so, providers=['CPUExecutionProvider'])

    t0 = time.time()
    enc_got = enc_sess.run(None, {'pixels': enc_in.numpy()})
    t_enc = time.time() - t0
    for i, (r, g) in enumerate(zip(enc_ref, enc_got)):
        d = np.abs(r.numpy() - g).max()
        rel = d / (np.abs(r.numpy()).max() + 1e-9)
        print(f'   encoder feat{i+1}: max|diff|={d:.3e} (rel {rel:.2e})')
        assert rel < 1e-3, f'encoder output {i} mismatch'

    t0 = time.time()
    head_got = head_sess.run(None, {f'feat{i+1}': head_ins[i].numpy() for i in range(4)})[0]
    t_head = time.time() - t0
    r = head_ref.numpy()
    d = np.abs(r - head_got).max()
    rng = r.max() - r.min() + 1e-9
    print(f'   head depth: max|diff|={d:.3e} (range {rng:.2f}, rel {d/rng:.2e})')
    assert d / rng < 1e-3, 'head output mismatch'
    print(f'== ORT CPU parity passed. enc {t_enc:.1f}s/frame-batch, head {t_head:.1f}s/window (CPU, fp32)')
    print(f'== ARTIFACTS:\n   {enc_path}\n   {head_path}')


if __name__ == '__main__':
    main()
