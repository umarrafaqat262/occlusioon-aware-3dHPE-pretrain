"""Real-video monocular 3D pose demo for KinFK-Mamba.

video → RTMPose (2D COCO-17 + scores) → coco2h36m → normalize → KinFK-Mamba lift
→ side-by-side render (2D overlay | 3D skeleton) → mp4/gif.

Occlusion showcase:
  * real occlusions surface as low detector confidence → the confidence-gated SSM
    coasts on temporal memory;
  * --occlude limb   injects synthetic occlusion (zeros a limb's 2D + confidence);
  * --conf-off       feeds confidence=1 everywhere (ablation: gate disabled) so you
    can A/B the benefit of novelty B under the same occlusion.

Usage:
  python demo.py --video clip.mp4 --checkpoint checkpoints/best_kinfk_cpn_tiny.pth
                 [--occlude limb] [--conf-off] [--out out.mp4] [--max-frames 300]
"""

import os, sys, argparse
import numpy as np
import torch
import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common.utils import load_config
from common.coco2h36m import coco2h36m_kpts, coco2h36m_conf, normalize_screen
from common.skeleton import H36M_PARENTS
from model.bsmamba import BoneStateMamba

_LEFT = [4, 5, 6, 11, 12, 13]
LIMBS = {'rarm': [14, 15, 16], 'rleg': [1, 2, 3], 'larm': [11, 12, 13], 'lleg': [4, 5, 6]}


def detect_2d(frames, device):
    """frames: list of BGR images → coco kpts (T,17,2), scores (T,17)."""
    from rtmlib import Body
    try:
        body = Body(mode='balanced', backend='onnxruntime', device=device)
    except Exception:
        body = Body(mode='balanced', backend='onnxruntime', device='cpu')
    K, S = [], []
    prev_k, prev_s = np.zeros((17, 2), np.float32), np.zeros(17, np.float32)
    for f in frames:
        kpts, scores = body(f)                      # (N,17,2),(N,17)
        if len(kpts) == 0:
            K.append(prev_k.copy()); S.append(np.zeros(17, np.float32)); continue
        i = scores.mean(-1).argmax()                # most-confident person
        prev_k, prev_s = kpts[i].astype(np.float32), scores[i].astype(np.float32)
        K.append(prev_k.copy()); S.append(prev_s.copy())
    return np.stack(K), np.stack(S)


@torch.no_grad()
def lift(model, pose2d, conf, device):
    """pose2d (T,17,2) normalized, conf (T,17,1) → 3D (T,17,3), seq2seq windows."""
    T = len(pose2d); F = model.cfg.num_frames
    out = np.zeros((T, 17, 3), np.float32)
    for s in range(0, T, F):
        e = min(s + F, T); n = e - s
        win2d = np.zeros((F, 17, 2), np.float32); winc = np.zeros((F, 17, 1), np.float32)
        win2d[:n] = pose2d[s:e]; winc[:n] = conf[s:e]
        if n < F:                                   # edge-pad short tail
            win2d[n:] = pose2d[e-1]; winc[n:] = conf[e-1]
        x = torch.from_numpy(win2d)[None].to(device); c = torch.from_numpy(winc)[None].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            pred, *_ = model(x, c)   # model returns (P, bone_dir, bone_len, P0)
        pred = (pred - pred[:, :, :1]).float().cpu().numpy()[0]   # root-relative
        out[s:e] = pred[:n]
    return out


def draw_2d(frame, kpt_h36m, conf):
    img = frame.copy()
    for j, p in enumerate(H36M_PARENTS):
        if p < 0:
            continue
        col = (0, 255, 0) if min(conf[j], conf[p]) > 0.3 else (0, 0, 255)  # red = low-conf
        a, b = kpt_h36m[j].astype(int), kpt_h36m[p].astype(int)
        cv2.line(img, tuple(a), tuple(b), col, 2)
    for j in range(17):
        cv2.circle(img, tuple(kpt_h36m[j].astype(int)), 3, (255, 255, 0), -1)
    return img


def render(frames, kpts_h36m, confs, pose3d, out_path, fps):
    H, W = frames[0].shape[:2]
    writer = imageio.get_writer(out_path, fps=fps, macro_block_size=None)
    for t in range(len(frames)):
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1); ax1.axis('off'); ax1.set_title('2D (red=low conf)')
        ax1.imshow(cv2.cvtColor(draw_2d(frames[t], kpts_h36m[t], confs[t]), cv2.COLOR_BGR2RGB))
        ax2 = fig.add_subplot(1, 2, 2, projection='3d'); ax2.set_title('KinFK-Mamba 3D')
        p = pose3d[t]
        for j, par in enumerate(H36M_PARENTS):
            if par < 0:
                continue
            c = 'b' if j in _LEFT else 'r'
            ax2.plot([p[j, 0], p[par, 0]], [p[j, 2], p[par, 2]], [-p[j, 1], -p[par, 1]], c)
        ax2.set_xlim(-0.6, 0.6); ax2.set_ylim(-0.6, 0.6); ax2.set_zlim(-0.6, 0.6)
        ax2.view_init(elev=10, azim=-70)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]   # mpl>=3.10
        writer.append_data(np.ascontiguousarray(buf)); plt.close(fig)
    writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--config', default='configs/cpn_tiny.yaml')
    ap.add_argument('--checkpoint', default='checkpoints/best_kinfk_cpn_tiny.pth')
    ap.add_argument('--weights', default='ema', choices=['ema', 'model'])
    ap.add_argument('--out', default='demo_out.mp4')
    ap.add_argument('--max-frames', type=int, default=300)
    ap.add_argument('--occlude', default='none', choices=['none'] + list(LIMBS))
    ap.add_argument('--conf-off', action='store_true', help='disable confidence gate (ablation)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = load_config(args.config)
    model = BoneStateMamba(cfg).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck['ema'] if (args.weights == 'ema' and 'ema' in ck) else ck['model'])
    model.eval()

    # Read via imageio/ffmpeg (robust to mp4/webm/ogv); downscale wide frames.
    reader = imageio.get_reader(args.video)
    try:
        fps = reader.get_meta_data().get('fps', 25) or 25
    except Exception:
        fps = 25
    frames = []
    for i, rgb in enumerate(reader):
        if len(frames) >= args.max_frames:
            break
        rgb = np.asarray(rgb)[..., :3]
        h, w = rgb.shape[:2]
        if w > 960:                                  # downscale for CPU detector speed
            s = 960 / w
            rgb = cv2.resize(rgb, (960, int(round(h * s))))
        frames.append(rgb[..., ::-1].copy())          # RGB→BGR for the detector
    reader.close()
    fps = min(fps, 30)
    print(f"{len(frames)} frames @ {fps:.0f} fps")

    coco_k, coco_s = detect_2d(frames, device)
    kp = coco2h36m_kpts(coco_k); cf = coco2h36m_conf(coco_s)        # (T,17,2),(T,17)

    if args.occlude != 'none':                                     # synthetic occlusion
        j = LIMBS[args.occlude]; kp[:, j] = 0.0; cf[:, j] = 0.0
        print(f"occluded {args.occlude} joints {j}")

    H, W = frames[0].shape[:2]
    norm = normalize_screen(kp.copy(), W, H).astype(np.float32)
    conf_in = np.ones_like(cf) if args.conf_off else cf
    pose3d = lift(model, norm, conf_in[..., None].astype(np.float32), device)

    render(frames, kp, cf, pose3d, args.out, fps)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
