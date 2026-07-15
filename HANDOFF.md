# KinFK-Mamba — implementation handoff

A real selective-SSM monocular 3D pose lifter (replaces the misnamed BiGRU
"BoneStateMamba"). Two novel mechanisms, verified unclaimed vs PoseMamba /
SasMamba / SAMA / DF-Mamba / P-STMO:

- **(A) Kinematic-tree FK-coupled SSM** — the spatial scan traverses the skeleton
  tree root↔leaf (`common/skeleton.py:KIN_SCAN_ORDER`); decoded by the FK head
  with per-clip-shared bone lengths (`model/bsmamba.py:FKDecoder`).
- **(B) Confidence-gated selective scan** — `model/ssm.py:ConfMamba`: per-frame 2D
  confidence gates the SSM Δ and integrated input, so occluded joints coast on
  temporal memory. Verified: conf=0 cuts sensitivity to occluded-region noise ~8×.

## Environment
- Python: `/opt/dlami/nvme/envs/bsmamba/bin/python` (torch 2.1.2+cu118, A10G).
- Always run with `PYTHONPATH=/opt/dlami/nvme/timymamba`.
- `mamba-ssm` 2.2.2 + `causal-conv1d` 1.4.0 (prebuilt wheels), `transformers==4.38.2` pinned.
- Disk: keep everything on `/opt/dlami/nvme` (root `/` is full). See memory `bsmamba-env`.

## Data
- CPN (VideoPose3D convention): `data/motion3d/cpn_vp3d/` (extracted from the
  RAR-in-`.pkl`). Loader: `common/dataset_vp3d.py` → camera-space mm, root-relative,
  S1/5/6/7/8 train, S9/11 test. Cache `_cache_*.pkl` built on first use.
- GT-2D: same loader with `keypoints_file=data_2d_h36m_gt.npz` (metric sanity check;
  expect ~21–30 mm).

## Architecture (`model/`)
`bsmamba.py` (encode → FK decode) · `st_block.py` (spatial then temporal) ·
`spatial_block.py` (kin-tree fast BiSSM + parent-gather prior, novelty A) ·
`temporal_block.py` (conf-gated BiSSM, novelty B) · `ssm.py` (`BiSSM`/`ConfMamba`).
Configs: `configs/cpn.yaml` (D=104, 1.32M) · `configs/cpn_tiny.yaml` (D=64, 0.446M).

## Run commands
```bash
cd /opt/dlami/nvme/timymamba
export PYTHONPATH=$PWD; PY=/opt/dlami/nvme/envs/bsmamba/bin/python

# (optional) Stage-I MPM pretrain → encoder weights
$PY pretrain.py --config configs/cpn_tiny.yaml

# Train (seq2seq, val-selected, AMP+EMA). Add --pretrained checkpoints/pretrained_*.pth
$PY train.py --config configs/cpn_tiny.yaml --tag kinfk_cpn_tiny

# Evaluate on test (mm, root-relative, all frames, flip-TTA, per-action)
$PY evaluate.py --config configs/cpn_tiny.yaml --checkpoint checkpoints/best_kinfk_cpn_tiny.pth

# Occlusion-robustness study (the headline experiment for novelty B)
$PY scripts/occlusion_eval.py --config configs/cpn_tiny.yaml --checkpoint checkpoints/best_kinfk_cpn_tiny.pth
```

## Status (2026-06-12)
- DONE + tested: env/kernels, SSM seam (coasting verified), full model (fwd/bwd/occlusion),
  VP3D loader (scale validated, 438mm body extent), seq2seq train (val split, AMP, EMA,
  exp-LR), eval (P-MPJPE/flip-TTA/per-action), MPM pretrain, occlusion script.
- RUNNING: `configs/cpn_tiny.yaml` 150-epoch CPN training (tag `kinfk_cpn_tiny`,
  ~7 min/epoch; log `/opt/dlami/nvme/train_cpn_tiny.log`). Best ckpt by VAL MPJPE.
- TODO (post-convergence): run evaluate.py + occlusion_eval.py; ablation matrix
  (GRU→SSM, +A, +FK-couple, +B, +MPM, +flip-TTA); GT-2D sanity gate; optionally the
  larger `cpn.yaml`. The current run is the no-MPM baseline; re-run with `--pretrained`
  for the two-stage number.

## Verification gates (per plan)
1. GT-2D MPJPE should land ~21–30 mm — confirms the camera-space mm metric is correct.
2. Occlusion curves: confidence-gated model should degrade more gracefully than a
   no-conf-gate ablation (set `conf_gate=False` in the blocks, retrain) — substantiates B.
