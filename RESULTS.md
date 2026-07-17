# Results — Occlusion-Aware 3D HPE

> **Repository**: `umarrafaqat262/occlusioon-aware-3dHPE-pretrain`
> **Dataset**: Human3.6M, CPN fine-tuned 2D keypoints, 243-frame seq2seq (VideoPose3D protocol)
> **Train split**: S1, S5, S6, S7, S8 | **Test split**: S9, S11 (standard H36M best-on-test convention, `select_on_test: true`)
> **Hardware**: NVIDIA L4, CUDA 13.2, torch 2.14.0.dev20260626

---

## 1. Objective

Measure the accuracy ceiling of two architectures in the **< 1M parameter** cluster on H36M CPN 243f:

| Model | Params | MPJPE ↓ | P-MPJPE ↓ |
|---|---|---|---|
| SasMamba | 0.64M | **41.5** | 34.8 |
| PoseMamba-S | 0.90M | **41.8** | 35.0 |
| PoseMamba-L (ref, >1M) | 6.7M | 38.1 | 32.5 |
| **AnatomyProj-Mamba (prev)** | **0.97M** | **48.2** | 37.9 |
| **AnatomyProj-Mamba CLEAN** | **0.97M** | **51.06** (best) | — |
| **CSM-Pose_S** | **0.908M** | **TBD** | — |

---

## 2. Run A: AnatomyProj-Mamba Clean (`anatproj_clean.yaml`)

**Config**: clean-ceiling run (occlusion augmentation OFF), DAP decoder + Mamba backbone.

### 2.1 MPM Pretrain

| Epochs | Final Avg Loss | LR | Checkpoint |
|---|---|---|---|
| 25 / 25 | **0.00049** | 0 (cosine decayed) | `pretrained_anatomyproj_mamba_clean.pth` (3.8 MB) |

### 2.2 Supervised Fine-tune (killed at epoch 62/120)

| Epoch | Train Loss | Test MPJPE (EMA) | Best |
|---|---|---|---|
| 5 | 0.0755 | 80.67mm | ✓ |
| 10 | 0.0616 | 56.12mm | ✓ |
| 15 | 0.0559 | 53.69mm | ✓ |
| 20 | 0.0527 | 52.42mm | ✓ |
| 25 | 0.0503 | 51.63mm | ✓ |
| 30 | 0.0487 | 51.25mm | ✓ |
| **35** | 0.0472 | **51.06mm** | ✓ **BEST** |
| 40 | 0.0457 | 51.07mm | — |
| 45 | 0.0444 | 51.07mm | — |
| 50 | 0.0437 | 51.30mm | — |
| 55 | 0.0429 | 51.21mm | — |
| 60 | 0.0423 | 51.29mm | — |

**Summary**: Best test MPJPE **51.06mm** at epoch 35. Plateaued ~51mm. Training killed at epoch 62.

---

## 3. Run B: CSM-Pose_S (`csm_s.yaml`)

**Config**: redesigned CSM-Pose_S architecture (0.908M params) with 8 new modules: DCT, KPA, Laplacian PE, limb-reorder scan, SSI, spatial conf gate, MSM, 3D in-fill head.

### 3.1 MPM Pretrain

| Epoch | Avg Loss | LR |
|---|---|---|
| 1 | 0.00500 | 0.00100 |
| 2 | 0.00161 | 0.00098 |
| 3 | 0.00131 | 0.00097 |
| 4 | 0.00116 | 0.00094 |
| 5 | 0.00103 | 0.00091 |
| 6 | 0.00101 | 0.00086 |
| 7 | 0.00093 | 0.00082 |
| 8 | 0.00088 | 0.00077 |
| 9 | 0.00083 | 0.00071 |
| 10 | 0.00078 | 0.00066 |
| 11 | 0.00077 | 0.00059 |
| 12 | 0.00074 | 0.00053 |
| 13 | 0.00070 | 0.00047 |
| 14 | 0.00066 | 0.00041 |
| 15 | 0.00064 | 0.00035 |
| 16 | 0.00061 | 0.00029 |
| 17 | 0.00059 | 0.00023 |
| 18 | 0.00059 | 0.00018 |
| 19 | 0.00056 | 0.00014 |
| 20 | 0.00055 | 0.00010 |
| 21 | 0.00054 | 0.00006 |
| 22 | 0.00053 | 0.00004 |
| 23 | 0.00051 | 0.00002 |
| 24 | 0.00051 | 0.000004 |
| **25** | **0.00050** | 0.000000 |

**Checkpoint**: `pretrained_csm_pose_s.pth` (4.07 MB)

### 3.2 Supervised Fine-tune (in progress — DIVERGED at epoch 40/120)

Training started from MPM-pretrained checkpoint. Loss exploded at epoch 16 onward.

| Epoch | Train Loss | VAL(EMA) MPJPE | Best |
|---|---|---|---|
| **5** | 0.2359 | **63.36mm** | ✓ **BEST** |
| 10 | 0.3434 | 65.04mm | — |
| 15 | 0.5675 | 295.24mm | — (exploding) |
| 20 | 8.0650 | 1456.89mm | — |
| 25 | 21.9045 | 3773.50mm | — |
| 30 | 35.7866 | 5669.28mm | — |
| 35 | 32.6480 | 2590.63mm | — |
| 40 | 32.0617 | 2590.63mm | — |

**Best VAL(EMA) MPJPE: 63.36mm** at epoch 5 (checkpoint: `best_csm_s.pth`).

**Training instability**: Loss collapsed at epoch 16 (0.23 → 0.99 → 2.35 → ... → 117 at epoch 32). The model achieved its best validation at epoch 5 and then diverged. Possible causes:
- Learning rate (0.0005) too aggressive for the new CSM-Pose modules
- Numerical instability in SSI, MSM, or spatial conf gate
- MPM pretrain mismatch with fine-tune objective

Training is still running (epoch 41/120) but unlikely to recover.

---

## 4. Checkpoints produced

| File | Size | Description |
|---|---|---|
| `pretrained_anatomyproj_mamba_clean.pth` | 3.8 MB | MPM-pretrained AnatomyProj-Mamba (25 epochs, loss 0.00049) |
| `best_anatproj_clean.pth` | 15.9 MB | Best AnatomyProj-Mamba (epoch 35, 51.06mm test MPJPE) |
| `best_anatproj_sota.pth` | 15.9 MB | Previous repo baseline (48.2mm) |
| `best_bonestatemamba_tiny.pth` | 19.9 MB | BoneStateMamba-Tiny checkpoint |
| `best_kinfk_cpn_sota.pth` | 7.4 MB | Kinematics-FK SOTA checkpoint |
| `best_kinfk_cpn_tiny.pth` | 7.4 MB | Kinematics-FK Tiny checkpoint |
| `pretrained_anatomyproj_mamba_cpn_sota.pth` | 3.97 MB | MPM-pretrained SOTA backbone |
| `pretrained_anatomyproj_mamba_v2.pth` | 4.57 MB | MPM-pretrained v2 backbone |
| `pretrained_kinfk_mamba_cpn_tiny_mpm.pth` | 1.86 MB | MPM-pretrained KinFK Tiny |
| `pretrained_csm_pose_s.pth` | 4.07 MB | MPM-pretrained CSM-Pose_S (25 epochs, loss 0.00050) |
| `best_csm_s.pth` | 15.0 MB | Best CSM-Pose_S (epoch 5, 63.36mm VAL MPJPE) |

---

## 5. SOTA Context

All models evaluated on **Human3.6M CPN 243f** (Protocol #1, mm):

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
| **AnatomyProj-Mamba CLEAN (this run)** | 0.97M | **51.06** (best) | — | this run |
| **CSM-Pose_S (this run)** | 0.908M | **63.36** (VAL, diverged) | — | this run |

---

*Last updated: 2026-07-17 11:00 UTC*
