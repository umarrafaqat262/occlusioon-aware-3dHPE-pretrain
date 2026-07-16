# Results — Occlusion-Aware 3D HPE with AnatomyProj-Mamba

> **Repository**: `umarrafaqat262/occlusioon-aware-3dHPE-pretrain`
> **Config**: `anatproj_clean.yaml` — clean-ceiling run (occlusion augmentation OFF)
> **Model**: AnatomyProj-Mamba (~0.97M params, **< 1M**)
> **Dataset**: Human3.6M, CPN fine-tuned 2D keypoints, 243-frame seq2seq (VideoPose3D protocol)
> **Train split**: S1, S5, S6, S7, S8 | **Test split**: S9, S11 (standard H36M best-on-test convention, `select_on_test: true`)
> **Hardware**: NVIDIA L4, CUDA 13.2, torch 2.14.0.dev20260626
> **Status**: Fine-tuning in progress (epoch 22/120, estimated ~16h remaining)

---

## 1. Objective

Measure the **true clean-CPN accuracy ceiling** of the AnatomyProj-Mamba architecture (DAP decoder + Mamba backbone) **without** occlusion augmentation contamination. This establishes the baseline before adding GCN, spatial confidence gating, structured occlusion augmentation, and noisy MPM pretraining.

**Target**: Compete in the **< 1M parameter** cluster on H36M CPN 243f:
| Model | Params | MPJPE ↓ | P-MPJPE ↓ |
|---|---|---|---|
| SasMamba | < 1M | **41.5** | 34.8 |
| PoseMamba-S | < 1M | **~38–39** | — |
| **AnatomyProj-Mamba (this run)** | **0.97M** | **TBD** (test eval pending) | — |

> ⚠️ **Metric caveat — read before comparing.** The ~42 mm figure below is the
> **training-batch MPJPE** (`train.py` progress-bar, last iteration of the epoch on
> augmented train data). It is **not** the test result and is systematically
> **optimistic** — this repo's own diagnostics showed a **~16 mm train→test gap**
> (33.3 mm train vs 49.6 mm test). The comparable test number is the
> **`VAL (EMA) MPJPE`** line logged every 5 epochs and, definitively, the
> `evaluate.py` run after 120 epochs (flip-TTA + frame-dedup). Do **not** claim we
> beat the 48.2 mm baseline until that test number is in — 48.2 was a *test* number.

---

## 2. Setup

### 2.1 Environment

```
conda create -n posemamba python=3.10 -y
# CUDA 13.2 + torch 2.14.0.dev20260626
pip install causal-conv1d>=1.4.0
# mamba-ssm 2.3.2.post1 compiled from source with CUDA extensions
pip install numpy pyyaml tqdm tensorboard einops
```

### 2.2 Data

Two files placed under `data/motion3d/cpn_vp3d/`:
- `data_3d_h36m.npz` (3D world positions, 407 MB)
- `data_2d_h36m_cpn_ft_h36m_dbb.npz` (CPN fine-tuned 2D detections)

First run builds a `_cache_v2_*.pkl` (~few minutes).

### 2.3 Smoke test

All three configs pass:
```
PYTHONPATH=$PWD python smoke_test.py
```

### 2.4 Config (`anatproj_clean.yaml`)

| Key | Value | Notes |
|---|---|---|
| `num_joints` | 17 | H36M skeleton |
| `num_bones` | 16 | Bone kinematics |
| `num_frames` | 243 | Long temporal context |
| `joint_embed_dim` | 48 | Per-joint token |
| `bone_embed_dim` | 48 | Per-bone token |
| `state_dim` | 96 | Mamba state dimension |
| `num_heads` / `num_blocks` | 4 / 4 | Architecture depth |
| `ssm_expand` | 1 | No SSM expansion |
| `d_state` | 24 | SSM state width |
| `decoder` | DAP | Differentiable Anatomical Projection |
| `dap_iter` / `dap_rho` / `dap_step` | 8 / 5.0 / 0.05 | Projection parameters |
| `dropout` | 0.15 | Regularisation |
| `batch_size` | 32 | — |
| `lr` | 0.0005 | Cosine schedule, 5-epoch warmup |
| `epochs` | 120 | Full schedule |
| `aug_joint_mask_min/max` | 0.0 | **Occlusion OFF** (clean run) |
| `flip_prob` / `rotation_max_deg` / `jitter_scale` | 0.5 / 20 / 0.05 | Mild spatial aug only |

### 2.5 Commands run

```bash
# Stage 1 — MPM Pretrain (25 epochs)
PYTHONPATH=$PWD python pretrain.py --config configs/anatproj_clean.yaml

# Stage 2 — Supervised Fine-tune (120 epochs, currently epoch 22/120)
PYTHONPATH=$PWD python train.py --config configs/anatproj_clean.yaml \
    --pretrained checkpoints/pretrained_anatomyproj_mamba_clean.pth \
    --tag anatproj_clean

# Stage 3 — Evaluate (pending fine-tune completion)
PYTHONPATH=$PWD python evaluate.py --config configs/anatproj_clean.yaml \
    --checkpoint checkpoints/best_anatproj_clean.pth
```

---

## 3. Results — MPM Pretrain

| Epochs | Final Avg Loss | Learning Rate | Checkpoint |
|---|---|---|---|
| 25 / 25 | **0.00049** | 0 (cosine decayed to 0) | `pretrained_anatomyproj_mamba_clean.pth` (3.8 MB) |

---

## 4. Results — Supervised Fine-tune (in progress)

Training started from MPM-pretrained checkpoint. **21 epochs completed** as of last capture (epoch 22 in progress).

### 4.1 Per-epoch TRAINING metrics (not test — see caveat above)

The MPJPE column is the training-batch metric from the progress bar, not the test
set. Test numbers (`VAL (EMA) MPJPE`, logged every 5 epochs) must be pasted here
from the training log — they are the only figures comparable to published/CPN numbers.

| Epoch | Loss (epoch avg) | Loss (end of epoch) | train MPJPE (last iter) | Learning Rate |
|---|---|---|---|---|
| 1 | 0.2371 | 0.1510 | 124.7mm | 1.0e-4 (warmup) |
| 2 | 0.1170 | 0.0975 | 81.5mm | 2.0e-4 (warmup) |
| 3 | 0.0918 | 0.0903 | 75.9mm | 3.0e-4 (warmup) |
| 4 | 0.0816 | 0.0816 | 68.7mm | 4.0e-4 (warmup) |
| 5 | 0.0755 | 0.0662 | 54.7mm | 5.0e-4 (peak) |
| 6 | 0.0707 | 0.0717 | 60.8mm | 5.0e-4 |
| 7 | 0.0676 | 0.0681 | 57.1mm | 5.0e-4 |
| 8 | 0.0653 | 0.0630 | 52.3mm | 5.0e-4 |
| 9 | 0.0634 | 0.0668 | 56.2mm | 4.99e-4 |
| 10 | 0.0616 | 0.0704 | 59.8mm | 4.99e-4 |
| 11 | 0.0603 | 0.0599 | 50.2mm | 4.98e-4 |
| 12 | 0.0590 | 0.0594 | 49.1mm | 4.97e-4 |
| 13 | 0.0577 | 0.0570 | 47.3mm | 4.95e-4 |
| 14 | 0.0569 | 0.0542 | 44.6mm | 4.94e-4 |
| 15 | 0.0559 | 0.0538 | 44.3mm | 4.93e-4 |
| 16 | 0.0553 | 0.0505 | 41.2mm | 4.91e-4 |
| 17 | 0.0543 | 0.0535 | 44.3mm | 4.89e-4 |
| 18 | 0.0537 | 0.0549 | 45.6mm | 4.87e-4 |
| 19 | 0.0532 | 0.0529 | 43.5mm | 4.85e-4 |
| 20 | 0.0527 | 0.0533 | 43.9mm | 4.82e-4 |
| 21 | 0.0521 | 0.0512 | **41.9mm** | 4.80e-4 |

*Epoch 22 in progress* — loss ~0.05, MPJPE ~40-45mm.

### 4.2 Training trajectory

- **Rapid initial drop**: MPJPE fell from ~125mm → ~55mm in the first 5 epochs (warmup phase)
- **Stabilisation**: Epochs 5-21 show gradual improvement from ~55mm → ~42mm
- **Current plateau**: Loss hovers around 0.05, MPJPE around 40-45mm with per-epoch fluctuation
- **Best checkpoint so far**: `checkpoints/best_anatproj_clean.pth` (15.9 MB, saved at lowest validation/test loss)
- **Training speed**: ~1.05s/iter × 554 iters/epoch ≈ 9.7 min/epoch on NVIDIA L4
- **Throughput**: ~10h for 60 epochs, ~19-20h for full 120 epochs

### 4.3 Key observations

1. The **cosine LR schedule** peaks at epoch 5 (5e-4) and has decayed ~4% by epoch 21. The loss follows a steady downward trend.
2. The **EMA (exponential moving average)** with decay 0.999 tracks the online model; the best checkpoint reflects EMA weights.
3. Per-epoch MPJPE fluctuates ±3mm around the trend — typical for H36M training with mild augmentation. The **test-time evaluation** (after fine-tune completes) will average over multi-frame predictions and should show lower, more stable MPJPE.
4. The **48.2 mm baseline is a TEST number**; the ~42 mm here is a **training** number, so they are **not directly comparable** (training MPJPE is expected to sit well below test given the ~16 mm gap). The healthy, steadily-decreasing trajectory is a good sign the clean config + pretraining + fixes are working, but the actual clean-CPN result is **unknown until `evaluate.py`** runs at epoch 120. Report VAL (EMA) at ep 5/10/15/20 to track the real test curve.

---

## 5. SOTA Context

All models evaluated on **Human3.6M CPN 243f** (Protocol #1, mm):

Numbers below are from the primary papers (cross-checked against the MotionAGFormer /
KTPFormer benchmark tables). Diffusion methods (D3DP) are cited as **J-Agg** (aggregated,
deployable), NOT the oracle J-Best.

| Model | Params | MPJPE ↓ | P-MPJPE ↓ | Venue |
|---|---|---|---|---|
| MixSTE | 33.6M | 40.9 | 32.6 | CVPR'22 |
| MotionBERT (ft) | ~42M | 39.2 | 32.9 | ICCV'23 |
| MotionAGFormer-B | 11.7M | 38.4 | 32.6 | WACV'24 |
| D3DP (J-Agg) | ~34M | 39.5 | 31.6 | ICCV'23 |
| KTPFormer | 33.7M | 37.3 | 30.1 | CVPR'24 |
| _— <1M efficient cluster (our target) —_ | | | | |
| SasMamba | 0.64M | 41.5 | 34.8 | 2024 |
| PoseMamba-S | 0.90M | 41.8 | 35.0 | AAAI'25 |
| PoseMamba-L (ref, >1M) | 6.7M | 38.1 | 32.5 | AAAI'25 |
| **AnatomyProj-Mamba (prev, TEST)** | 0.97M | **48.2** | 37.9 | repo baseline |
| **AnatomyProj-Mamba CLEAN (this run)** | 0.97M | **TBD — eval pending** | — | in progress |

Goal: bring the clean-CPN **test** MPJPE into the <1M cluster (SasMamba 41.5 / PoseMamba-S
41.8). Whether this run gets there is unknown until `evaluate.py` reports the test number;
the training curve only indicates the optimization is healthy.

---

## 6. Checkpoints produced

| File | Size | Description |
|---|---|---|
| `pretrained_anatomyproj_mamba_clean.pth` | 3.8 MB | MPM-pretrained backbone (25 epochs, loss 0.00049) |
| `best_anatproj_clean.pth` | 15.9 MB | Best supervised fine-tune checkpoint so far (21 epochs, ~42mm MPJPE) |
| `best_anatproj_sota.pth` | 15.9 MB | Previous repo baseline (48.2mm) |

---

## 7. What's next (pending fine-tune)

1. **Complete 120 epochs** (~16h remaining)
2. **Evaluate on test set**: `PYTHONPATH=$PWD python evaluate.py --config configs/anatproj_clean.yaml --checkpoint checkpoints/best_anatproj_clean.pth`
3. **Repeat for GCN config** (`anatproj_gcn.yaml`) to chase PoseMamba-S territory (~38-39mm)
4. **Full occlusion-aware model** (`anatproj_occ.yaml`) with structured occlusion aug + spatial conf gate
5. **Occlusion ablation study**: per-limb occlusion sweep, noise robustness, confidence-off ablation

---

*Last updated: 2026-07-16 10:00 UTC*
