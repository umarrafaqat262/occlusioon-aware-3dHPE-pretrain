"""Throwaway diagnostics (Tier-0 steps 3 & 4) — read-only, eval the existing ckpt.

 3) Raw P0 (pre-projection regression) vs DAP-projected output MPJPE on the S9/S11
    test set. Tests "DAP fixes outputs, not features": if P0 ~= DAP the projection
    adds anatomy but ~0 accuracy; if DAP << P0 the projection is load-bearing.
 4) Train-clean fit: MPJPE on the TRAIN subjects (S1/5/6/7/8) with NO augmentation
    (the dataset returns clean data; aug lives in train.py). Underfit vs generalization:
    train-clean ~30 & test ~49 -> generalization/capacity gap;
    train-clean ~45        -> underfitting/optimization.

Both use EMA weights (matching the training-time val). conf is passed through as-is.
"""
import sys, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.utils import load_config
from common.dataset_vp3d import VP3DDataset
from model.bsmamba import BoneStateMamba


@torch.no_grad()
def _mpjpe_pair(model, loader, device, max_batches=None):
    """Returns (mpjpe_final_mm, mpjpe_p0_mm) root-relative over the loader."""
    ef, e0, n = 0.0, 0.0, 0
    for i, (pose_2d, pose_3d, conf) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        pose_2d, pose_3d, conf = pose_2d.to(device), pose_3d.to(device), conf.to(device)
        pred, _bdir, _blen, p0 = model(pose_2d, conf)
        gt = pose_3d - pose_3d[:, :, :1]
        for out, acc in ((pred, 'f'), (p0, '0')):
            o = out - out[:, :, :1]
            err = (o - gt).norm(dim=-1).mean(dim=-1)      # (B,T)
            if acc == 'f':
                ef += err.sum().item()
            else:
                e0 += err.sum().item()
        n += (pose_3d.shape[0] * pose_3d.shape[1])
    return ef / n * 1000.0, e0 / n * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/anatproj_sota.yaml')
    ap.add_argument('--checkpoint', default='checkpoints/best_anatproj_sota.pth')
    ap.add_argument('--train-batches', type=int, default=50,
                    help='cap train-clean batches for speed (0 = all)')
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BoneStateMamba(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt['ema'] if 'ema' in ckpt else ckpt['model']
    model.load_state_dict(state); model.eval()
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch','?')}, EMA)")

    # --- Step 3: P0 vs DAP on the TEST set (non-overlapping windows) ---
    test_ds = VP3DDataset(cfg.data_dir, 'test', cfg.num_frames, cfg.num_frames,
                          cfg.keypoints_file)
    test_ld = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    f_mm, p0_mm = _mpjpe_pair(model, test_ld, device)
    print("\n" + "=" * 52)
    print("  [3] TEST set (S9/S11), no flip-TTA")
    print(f"    DAP-projected (final) MPJPE : {f_mm:.1f} mm")
    print(f"    Raw P0 (pre-projection)     : {p0_mm:.1f} mm")
    print(f"    DAP - P0 delta              : {f_mm - p0_mm:+.1f} mm")
    print("=" * 52)

    # --- Step 4: train-clean fit (no aug; dataset returns clean data) ---
    tr_ds = VP3DDataset(cfg.data_dir, 'train', cfg.num_frames, cfg.num_frames,
                        cfg.keypoints_file, subset='all', val_fraction=0.0)
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    mb = args.train_batches or None
    tf_mm, tp0_mm = _mpjpe_pair(model, tr_ld, device, max_batches=mb)
    seen = (mb * cfg.batch_size) if mb else len(tr_ds)
    print("\n" + "=" * 52)
    print(f"  [4] TRAIN-CLEAN fit (S1/5/6/7/8, no aug, ~{seen} clips)")
    print(f"    DAP-projected MPJPE         : {tf_mm:.1f} mm")
    print(f"    Raw P0 MPJPE                : {tp0_mm:.1f} mm")
    print(f"    train-clean vs test gap     : {tf_mm:.1f}  vs  {f_mm:.1f}")
    print("=" * 52)


if __name__ == '__main__':
    main()
