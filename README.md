# Occlusion-Aware 3D Human Pose Estimation with Mamba (AnatomyProj-Mamba)

Efficient (**< 1M parameter**) monocular 2D→3D human-pose lifter on a bidirectional
spatio-temporal **Mamba (SSM)** backbone, targeting **occlusion robustness** on
Human3.6M (VideoPose3D / CPN protocol, 243-frame, seq2seq).

Two novelties:
- **(A) Differentiable Anatomical Projection (DAP)** decoder — regressed joints refined
  onto the constant-bone-length manifold; also completes occluded joints.
- **(B) Confidence-gated selective scan** — the SSM time-step Δ is gated by 2D-keypoint
  confidence, so occluded joints coast on temporal memory.
Plus a local-joint **GCN branch** fused with the spatial SSM, and **structured
(spatio-temporally correlated) occlusion augmentation** with masked+noise MPM pretraining.

## Quickstart
See **[GUIDE_GPU.md](GUIDE_GPU.md)** for environment setup, data layout, and the exact
train / evaluate / occlusion-study commands. Architecture details are in
[architect.md](architect.md).

```bash
# clean-accuracy config with the GCN branch (chase the <1M SOTA cluster)
PYTHONPATH=$PWD python pretrain.py --config configs/anatproj_gcn.yaml
PYTHONPATH=$PWD python train.py    --config configs/anatproj_gcn.yaml \
    --pretrained checkpoints/pretrained_anatomyproj_mamba_gcn.pth --tag anatproj_gcn
PYTHONPATH=$PWD python evaluate.py --config configs/anatproj_gcn.yaml \
    --checkpoint checkpoints/best_anatproj_gcn.pth

# full occlusion-aware model (headline)
PYTHONPATH=$PWD python train.py    --config configs/anatproj_occ.yaml --tag anatproj_occ
PYTHONPATH=$PWD python evaluate.py --config configs/anatproj_occ.yaml \
    --checkpoint checkpoints/best_anatproj_occ.pth --occlusion
```

## Status
Current baseline (0.968M, `best_anatproj_sota.pth`): **48.2 / 37.9 mm** MPJPE/P-MPJPE
(H36M CPN 243f). Target: match the <1M cluster (SasMamba 41.5, PoseMamba-S ~38) and win
on occlusion benchmarks. See GUIDE_GPU.md §5 for the roadmap and ablation matrix.

## Configs
- `anatproj_clean.yaml` — clean-ceiling run (occlusion aug off)
- `anatproj_gcn.yaml` — clean + local-joint GCN branch
- `anatproj_occ.yaml` — full occlusion-aware model (GCN + spatial conf gate + structured aug + noisy MPM)
