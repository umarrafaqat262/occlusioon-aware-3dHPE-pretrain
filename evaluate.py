"""KinFK-Mamba evaluation — Human3.6M, VideoPose3D/CPN protocol.

Protocol #1 MPJPE and #2 P-MPJPE, root-relative, in millimetres, over ALL test
frames (seq2seq), with optional horizontal-flip test-time augmentation and a
per-action breakdown. Comparable to published CPN numbers.
"""

import os, sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.utils import load_config
from common.dataset_vp3d import VP3DDataset
from common.augmentation import _LEFT, _RIGHT
from model.bsmamba import BoneStateMamba


def p_mpjpe_np(pred, gt):
    """Procrustes-aligned MPJPE per frame (canonical). pred/gt: (N,17,3) → (N,)."""
    muX, muY = gt.mean(1, keepdims=True), pred.mean(1, keepdims=True)
    X0, Y0 = gt - muX, pred - muY
    nX = np.sqrt((X0 ** 2).sum((1, 2), keepdims=True))
    nY = np.sqrt((Y0 ** 2).sum((1, 2), keepdims=True))
    X0 /= (nX + 1e-8); Y0 /= (nY + 1e-8)
    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))
    sign_detR = np.sign(np.linalg.det(R))                  # (N,)
    V[:, :, -1] *= sign_detR[:, None]
    s[:, -1] *= sign_detR
    R = np.matmul(V, U.transpose(0, 2, 1))
    a = s.sum(1)[:, None, None] * nX / nY                   # (N,1,1)
    t = muX - a * np.matmul(muY, R)
    pred_aligned = a * np.matmul(pred, R) + t
    return np.linalg.norm(pred_aligned - gt, axis=-1).mean(-1)


def _unflip_3d(pred):
    """Mirror a 3D prediction back: negate x, swap left/right joints."""
    out = pred.clone()
    out[..., 0] = -out[..., 0]
    tmp = out[:, :, _LEFT].clone()
    out[:, :, _LEFT] = out[:, :, _RIGHT]; out[:, :, _RIGHT] = tmp
    return out


def _flip_2d(pose_2d, conf):
    p = pose_2d.clone(); p[..., 0] = -p[..., 0]
    tmp = p[:, :, _LEFT].clone(); p[:, :, _LEFT] = p[:, :, _RIGHT]; p[:, :, _RIGHT] = tmp
    c = conf.clone(); tc = c[:, :, _LEFT].clone(); c[:, :, _LEFT] = c[:, :, _RIGHT]; c[:, :, _RIGHT] = tc
    return p, c


@torch.no_grad()
def run_eval(model, cfg, device, flip_tta=True):
    ds = VP3DDataset(cfg.data_dir, 'test', cfg.num_frames, cfg.num_frames, cfg.keypoints_file)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    print(f"  {len(ds)} test clips × {cfg.num_frames} frames  (flip_tta={flip_tta})")

    e1_all, e2_all, act_all = [], [], []
    ptr = 0
    for pose_2d, pose_3d, conf in tqdm(loader, desc="Evaluating", ncols=80):
        b = len(pose_2d)
        sources = ds.sources[ptr:ptr + b]; ptr += b
        pose_2d, pose_3d, conf = pose_2d.to(device), pose_3d.to(device), conf.to(device)
        pred, *_ = model(pose_2d, conf)
        if flip_tta:
            fp, fc = _flip_2d(pose_2d, conf)
            pred_f, *_ = model(fp, fc)
            pred = (pred + _unflip_3d(pred_f)) / 2
        pred = pred - pred[:, :, :1]; gt = pose_3d - pose_3d[:, :, :1]
        e1 = (pred - gt).norm(dim=-1).mean(dim=-1) * 1000.0      # (B,T) mm
        e1_all.append(e1.flatten().cpu().numpy())
        pf = pred.reshape(-1, 17, 3).cpu().numpy() * 1000.0
        gf = gt.reshape(-1, 17, 3).cpu().numpy() * 1000.0
        for s in range(0, len(pf), 20000):
            e2_all.append(p_mpjpe_np(pf[s:s+20000], gf[s:s+20000]))
        # action label per frame (action = middle token of "Subj_Action_Cam")
        acts = ['_'.join(src.split('_')[1:-1]).split(' ')[0] for src in sources]
        act_all.append(np.repeat(acts, cfg.num_frames))

    e1 = np.concatenate(e1_all); e2 = np.concatenate(e2_all)
    acts = np.concatenate(act_all)
    # Score each unique test frame once (cover_tail overlaps duplicate tail frames).
    keep = ds.dedup_mask().reshape(-1)
    if keep.shape[0] == e1.shape[0]:
        e1, e2, acts = e1[keep], e2[keep], acts[keep]
    print(f"\n{'='*52}\n  Protocol #1  MPJPE   : {e1.mean():.1f} mm")
    print(f"  Protocol #2  P-MPJPE : {e2.mean():.1f} mm\n{'='*52}")
    print("  Per-action MPJPE (mm):")
    for a in sorted(set(acts.tolist())):
        m = e1[acts == a].mean()
        print(f"    {a:<14s} {m:6.1f}")
    print(f"{'='*52}")
    print("  SOTA (H3.6M CPN, 243f, seq2seq):")
    print("    SasMamba-base 0.64M -> 41.5 / 34.8")
    print("    PoseMamba-S   0.86M -> ~38-39")
    print("    PoseMamba-L   6.71M -> 38.1 / 32.5")
    print(f"    Ours          -> {e1.mean():.1f} / {e2.mean():.1f}\n{'='*52}")
    return float(e1.mean()), float(e2.mean())


# Limb joint groups (H36M 17j) for the structured-occlusion study.
_LIMBS = {'rleg': [1, 2, 3], 'lleg': [4, 5, 6],
          'larm': [11, 12, 13], 'rarm': [14, 15, 16]}


@torch.no_grad()
def run_occlusion_eval(model, cfg, device, occ_frac=0.5):
    """Structured-occlusion robustness study (headline for novelty A+B).

    For each test clip, occlude one limb (conf→0 over a contiguous span covering
    `occ_frac` of frames). The model must coast (gated SSM) and reconstruct the
    missing joints (anatomical projection). Reports MPJPE on the OCCLUDED joints
    (reconstruction quality) and overall — per limb and averaged. Run the same
    command on a gate-off / FK-baseline checkpoint to get the ablation columns.
    """
    ds = VP3DDataset(cfg.data_dir, 'test', cfg.num_frames, cfg.num_frames, cfg.keypoints_file)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    T = cfg.num_frames
    span = max(1, int(T * occ_frac))
    t0 = (T - span) // 2                       # centred occlusion window
    print(f"  Occlusion study: {len(ds)} clips, span={span}/{T} frames (centred)")

    results = {}
    for limb, joints in _LIMBS.items():
        occ_err, all_err = [], []
        for pose_2d, pose_3d, conf in tqdm(loader, desc=f"occ:{limb}", ncols=80):
            pose_2d, pose_3d, conf = pose_2d.to(device), pose_3d.to(device), conf.to(device)
            c = conf.clone()
            c[:, t0:t0 + span][:, :, joints] = 0.0          # mark limb occluded
            pred, *_ = model(pose_2d, c)
            pred = pred - pred[:, :, :1]; gt = pose_3d - pose_3d[:, :, :1]
            d = (pred - gt).norm(dim=-1) * 1000.0           # (B,T,J) mm
            occ_err.append(d[:, t0:t0 + span][:, :, joints].flatten().cpu().numpy())
            all_err.append(d.flatten().cpu().numpy())
        results[limb] = (float(np.concatenate(occ_err).mean()),
                         float(np.concatenate(all_err).mean()))

    print(f"\n{'='*56}\n  STRUCTURED-OCCLUSION MPJPE (mm)")
    print(f"  {'limb':<8s}{'occluded-joints':>18s}{'overall':>12s}")
    occ_means = []
    for limb, (oe, ae) in results.items():
        print(f"  {limb:<8s}{oe:>18.1f}{ae:>12.1f}"); occ_means.append(oe)
    print(f"  {'MEAN':<8s}{np.mean(occ_means):>18.1f}")
    print(f"{'='*56}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/cpn.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/best_kinfk_mamba_cpn.pth')
    parser.add_argument('--no-flip', action='store_true')
    parser.add_argument('--weights', default='ema', choices=['ema', 'model'])
    parser.add_argument('--occlusion', action='store_true',
                        help='run the structured-occlusion robustness study')
    parser.add_argument('--occ-frac', type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BoneStateMamba(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt['ema'] if (args.weights == 'ema' and 'ema' in ckpt) else ckpt['model']
    model.load_state_dict(state); model.eval()
    print(f"Loaded {args.checkpoint} (weights={args.weights}, epoch {ckpt.get('epoch','?')})")
    run_eval(model, cfg, device, flip_tta=not args.no_flip)
    if args.occlusion:
        run_occlusion_eval(model, cfg, device, occ_frac=args.occ_frac)


if __name__ == '__main__':
    main()
