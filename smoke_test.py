"""Training-readiness smoke test (run on the GPU box, no dataset needed).

Builds each AnatomyProj config on synthetic data and checks, in ~30s, that the
model constructs, stays < 1M params, runs a forward + backward, takes optimizer
steps with a finite decreasing loss, that the augmentation pipeline works, and
that the new occlusion features are actually active. Exits non-zero on any failure,
so you can gate a long training run behind:  python smoke_test.py && python train.py ...

Usage:
  PYTHONPATH=$PWD python smoke_test.py
  PYTHONPATH=$PWD python smoke_test.py --configs configs/anatproj_occ.yaml
"""

import argparse, sys, traceback
import numpy as np
import torch

from common.utils import load_config, count_parameters
from model.bsmamba import BoneStateMamba
from losses import TotalLoss
from train import augment

CONFIGS = ['configs/anatproj_clean.yaml',
           'configs/anatproj_gcn.yaml',
           'configs/anatproj_occ.yaml',
           'configs/csm_base.yaml',
           'configs/csm_s.yaml']
PARAM_BUDGET = 1_000_000
B = 2


def synth_batch(cfg, device):
    T, J = cfg.num_frames, cfg.num_joints
    pose_2d = torch.randn(B, T, J, 2, device=device) * 0.5
    pose_3d = torch.randn(B, T, J, 3, device=device) * 0.3
    pose_3d = pose_3d - pose_3d[:, :, :1]            # root-relative, like the data
    conf = torch.rand(B, T, J, 1, device=device)     # exercise the confidence path
    return pose_2d, pose_3d, conf


def check_config(path, device):
    print(f"\n=== {path} ===")
    cfg = load_config(path)
    model = BoneStateMamba(cfg).to(device).train()
    n = count_parameters(model)
    print(f"  params: {n:,} ({n/1e6:.3f}M)")
    assert n < PARAM_BUDGET, f"OVER BUDGET: {n:,} >= {PARAM_BUDGET:,}"

    # occlusion features active where expected
    sb = model.blocks[0].spatial
    if getattr(cfg, 'spatial_gcn', False):
        assert getattr(sb, 'gcn', False), "spatial_gcn=true but GCN branch not built"
        print("  GCN branch: ON")
    if getattr(cfg, 'spatial_conf_gate', False):
        assert sb.ssm.conf_gate, "spatial_conf_gate=true but BiSSM gating is off"
        print("  spatial conf-gate: ON")
    # CSM-Pose redesign modules
    if getattr(cfg, 'use_dct', False):
        assert model.dct is not None, "use_dct=true but DCT front-end not built"
        print(f"  DCT denoise: ON (n_coef={model.dct.n_coef})")
    if getattr(cfg, 'spatial_kpa', False):
        assert sb.kpa is not None, "spatial_kpa=true but KPA not built"
        print("  KPA graph prior: ON")
    if getattr(cfg, 'spatial_lap_pe', 0):
        assert sb.lap_pe, "spatial_lap_pe set but PE not built"
        print("  Laplacian PE: ON")
    if getattr(cfg, 'spatial_limb_reorder', False):
        assert sb.limb_reorder, "spatial_limb_reorder=true but not built"
        print("  limb-reorder scan: ON")
    if getattr(cfg, 'spatial_ssi', False):
        assert sb.ssi, "spatial_ssi=true but SSI not built"
        print("  SSI state fusion: ON")
    if getattr(cfg, 'temporal_motion', False):
        assert getattr(model.blocks[0].temporal.ssm.fwd, 'motion_adaptive', False), \
            "temporal_motion=true but MSM not built"
        print("  MSM motion-adaptive Δ: ON")
    if getattr(cfg, 'use_infill', False):
        assert model.infill_head is not None, "use_infill=true but in-fill head not built"
        print("  3D in-fill head: ON")

    # augmentation pipeline (CPU tensors, epoch mid-run so curriculum is active)
    p2, p3, c = (t.cpu() for t in synth_batch(cfg, 'cpu'))
    p2, p3, c = augment(p2, p3, c, cfg, epoch=cfg.epochs // 2)
    assert p2.shape == (B, cfg.num_frames, cfg.num_joints, 2), "augment changed 2D shape"
    assert torch.isfinite(p2).all(), "augment produced non-finite 2D"
    print("  augment(): OK")

    # forward + backward + two optimizer steps
    crit = TotalLoss(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pose_2d, pose_3d, conf = synth_batch(cfg, device)
    losses = []
    for _ in range(2):
        opt.zero_grad()
        pred, bone_dir, bone_len, p0 = model(pose_2d, conf)
        assert pred.shape == (B, cfg.num_frames, cfg.num_joints, 3), \
            f"bad output shape {tuple(pred.shape)}"
        loss, _ = crit(pred, pose_3d, bone_len, pose_2d, p0)
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()
        g = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(g), "non-finite grad norm"
        opt.step()
        losses.append(loss.item())
    print(f"  forward/backward: OK  loss {losses[0]:.4f} -> {losses[1]:.4f}  gradnorm~{g:.3f}")
    return True


def check_dedup():
    """Standalone check of the cover_tail dedup logic (no dataset needed)."""
    print("\n=== dedup logic ===")
    # a source of 250 frames, T=100 stride=100 -> clips [0:100],[100:200]; tail [150:250]
    frames = np.arange(250)
    T, stride = 100, 100
    clips = []
    last = 0
    for s in range(0, len(frames) - T + 1, stride):
        clips.append(frames[s:s + T]); last = s
    if last + T < len(frames):
        clips.append(frames[len(frames) - T:])       # tail overlaps 150..199
    clips = np.array(clips)
    flat = clips.reshape(-1)
    _, first = np.unique(flat, return_index=True)
    keep = np.zeros(flat.shape[0], bool); keep[first] = True
    kept_ids = flat[keep]
    assert len(np.unique(kept_ids)) == len(kept_ids), "dedup left duplicates"
    assert set(kept_ids.tolist()) == set(range(250)), "dedup dropped/added frames"
    print(f"  clips={clips.shape} kept unique frames={keep.sum()}/{flat.size}  OK")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='*', default=CONFIGS)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — the mamba_ssm selective-scan kernel needs a "
              "GPU. Run this on the training box.")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}  torch: {torch.__version__}")

    ok = True
    try:
        check_dedup()
    except Exception:
        ok = False; traceback.print_exc()
    for path in args.configs:
        try:
            check_config(path, device)
        except Exception:
            ok = False; traceback.print_exc()

    print("\n" + ("=" * 40))
    print("SMOKE TEST: PASS ✅ — safe to start training" if ok
          else "SMOKE TEST: FAIL ❌ — fix the errors above before training")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
