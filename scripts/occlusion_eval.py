"""Occlusion-robustness study — recognized protocol (arXiv 2504.10350), adapted
for our 2D->3D lifter on Human3.6M (run from repo root).

Occlusion = realistic detector behaviour on occluded joints: their 2D gets
zero-mean Gaussian noise AND their confidence drops. A confidence-aware model
(novelty B) should coast on those; a confidence-off model trusts the noise.

Reports, per noise level sigma in {0,.001,.005,.01,.03,.05} (fraction of image
resolution; in our width-normalized coords sigma_norm ~= 2*sigma):
  - Overall / Visible / Occluded MPJPE (mm), root-relative, all frames
  - confidence-aware (ours) vs confidence-off ablation
Plus a per-keypoint sensitivity pass (corrupt one joint at sigma=0.03).

Usage:
  python scripts/occlusion_eval.py --checkpoint checkpoints/best_kinfk_cpn_sota.pth
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.utils import load_config
from common.dataset_vp3d import VP3DDataset
from model.bsmamba import BoneStateMamba

SIGMAS = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05]
OCC_CONF = 0.1            # confidence assigned to occluded/noised joints
OCC_FRAC = 1.0 / 3.0     # fraction of joints occluded per clip (~BlendMimic3D 34%)


def _mpjpe_mm(model, loader, device, sigma, occ_mask_fn, conf_aware, gen):
    """occ_mask_fn(B,J)->bool mask of occluded joints. Returns (overall,vis,occ) mm."""
    so = sv = soc = 0.0
    no = nv = noc = 0
    for pose_2d, pose_3d, conf in loader:
        pose_2d, pose_3d = pose_2d.to(device), pose_3d.to(device)
        B, T, J, _ = pose_2d.shape
        occ = occ_mask_fn(B, J).to(device)                       # (B,J) bool
        occ_t = occ[:, None, :, None]                            # (B,1,J,1)
        x = pose_2d.clone()
        if sigma > 0:
            noise = torch.randn(B, T, J, 2, generator=gen, device='cpu').to(device) * (2 * sigma)
            x = x + noise * occ_t                                # noise only occluded joints
        c = torch.ones(B, T, J, 1, device=device)
        if conf_aware:
            c = torch.where(occ_t, torch.full_like(c, OCC_CONF), c)
        pred, *_ = model(x, c)   # model returns (P, bone_dir, bone_len, P0)
        pred = pred - pred[:, :, :1]; gt = pose_3d - pose_3d[:, :, :1]
        err = (pred - gt).norm(dim=-1) * 1000.0                  # (B,T,J) mm
        om = occ[:, None, :].expand(B, T, J)
        so += err.sum().item(); no += err.numel()
        sv += err[~om].sum().item(); nv += (~om).sum().item()
        soc += err[om].sum().item(); noc += om.sum().item()
    return so / no, (sv / nv if nv else 0), (soc / noc if noc else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/cpn_tiny_sota.yaml')
    ap.add_argument('--checkpoint', default='checkpoints/best_kinfk_cpn_sota.pth')
    ap.add_argument('--weights', default='ema', choices=['ema', 'model'])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BoneStateMamba(cfg).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck['ema'] if (args.weights == 'ema' and 'ema' in ck) else ck['model'])
    model.eval()
    ds = VP3DDataset(cfg.data_dir, 'test', cfg.num_frames, cfg.num_frames, cfg.keypoints_file)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    J = cfg.num_joints

    def rand_occ(B, j, seed):
        g = np.random.RandomState(seed)
        m = np.zeros((B, j), bool); n = max(1, int(j * OCC_FRAC))
        for b in range(B):
            m[b, g.choice(j, n, replace=False)] = True
        return torch.from_numpy(m)

    print(f"Occlusion study | {args.checkpoint}\n{'='*64}")
    print(f"{'sigma':>6} | {'conf-aware (ours)':>26} | {'conf-off':>26}")
    print(f"{'':>6} | {'overall  vis    occ':>26} | {'overall  vis    occ':>26}")
    with torch.no_grad():
        for sig in SIGMAS:
            mask_fn = lambda B, j, s=sig: rand_occ(B, j, int(s * 1e5))
            g = torch.Generator().manual_seed(0)
            oa = _mpjpe_mm(model, loader, device, sig, mask_fn, True, g)
            g = torch.Generator().manual_seed(0)
            of = _mpjpe_mm(model, loader, device, sig, mask_fn, False, g)
            print(f"{sig:6.3f} | {oa[0]:7.1f} {oa[1]:6.1f} {oa[2]:6.1f}      | "
                  f"{of[0]:7.1f} {of[1]:6.1f} {of[2]:6.1f}")
    print(f"{'='*64}")
    print("Per-keypoint sensitivity (corrupt one joint, sigma=0.03): occ-MPJPE mm")
    with torch.no_grad():
        for j in range(J):
            mask_fn = lambda B, jj, jidx=j: torch.nn.functional.one_hot(
                torch.full((B,), jidx), jj).bool()
            g = torch.Generator().manual_seed(0)
            a = _mpjpe_mm(model, loader, device, 0.03, mask_fn, True, g)[2]
            g = torch.Generator().manual_seed(0)
            f = _mpjpe_mm(model, loader, device, 0.03, mask_fn, False, g)[2]
            print(f"  joint {j:2d}: conf-aware {a:6.1f}  | conf-off {f:6.1f}  | gain {f-a:+5.1f}")


if __name__ == '__main__':
    main()
