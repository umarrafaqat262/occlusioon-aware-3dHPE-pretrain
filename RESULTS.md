# Results — Occlusion-Aware 3D HPE with AnatomyProj-Mamba

> **Repository**: `umarrafaqat262/occlusioon-aware-3dHPE-pretrain`
> **Config**: `anatproj_clean.yaml` — clean-ceiling run (occlusion augmentation OFF)
> **Model**: AnatomyProj-Mamba (~0.97M params, **< 1M**)
> **Dataset**: Human3.6M, CPN fine-tuned 2D keypoints, 243-frame seq2seq (VideoPose3D protocol)
> **Train split**: S1, S5, S6, S7, S8 | **Test split**: S9, S11 (standard H36M best-on-test convention, `select_on_test: true`)
> **Hardware**: NVIDIA L4, CUDA 13.2, torch 2.14.0.dev20260626
> **Status**: Fine-tuning in progress (epoch 62/120, estimated ~10h remaining)

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

# Stage 2 — Supervised Fine-tune (120 epochs, currently epoch 62/120)
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

Training started from MPM-pretrained checkpoint. **61 epochs completed** as of last capture (epoch 62 in progress).

### 4.1 Per-epoch training loss

| Epoch | Train Loss (avg) | LR |
|---|---|---|
| 1 | 0.2371 | 1.0e-4 |
| 2 | 0.1170 | 2.0e-4 |
| 3 | 0.0918 | 3.0e-4 |
| 4 | 0.0816 | 4.0e-4 |
| 5 | 0.0755 | 5.0e-4 |
| 6 | 0.0707 | 5.0e-4 |
| 7 | 0.0676 | 5.0e-4 |
| 8 | 0.0653 | 5.0e-4 |
| 9 | 0.0634 | 4.99e-4 |
| 10 | 0.0616 | 4.99e-4 |
| 11 | 0.0603 | 4.98e-4 |
| 12 | 0.0590 | 4.97e-4 |
| 13 | 0.0577 | 4.95e-4 |
| 14 | 0.0569 | 4.94e-4 |
| 15 | 0.0559 | 4.93e-4 |
| 16 | 0.0553 | 4.91e-4 |
| 17 | 0.0543 | 4.89e-4 |
| 18 | 0.0537 | 4.87e-4 |
| 19 | 0.0532 | 4.85e-4 |
| 20 | 0.0527 | 4.82e-4 |
| 21 | 0.0521 | 4.80e-4 |
| 22 | 0.0515 | 4.77e-4 |
| 23 | 0.0511 | 4.74e-4 |
| 24 | 0.0505 | 4.71e-4 |
| 25 | 0.0503 | 4.67e-4 |
| 26 | 0.0498 | 4.64e-4 |
| 27 | 0.0495 | 4.60e-4 |
| 28 | 0.0493 | 4.57e-4 |
| 29 | 0.0489 | 4.53e-4 |
| 30 | 0.0487 | 4.49e-4 |
| 31 | 0.0484 | 4.45e-4 |
| 32 | 0.0481 | 4.41e-4 |
| 33 | 0.0478 | 4.37e-4 |
| 34 | 0.0475 | 4.33e-4 |
| 35 | 0.0472 | 4.28e-4 |
| 36 | 0.0469 | 4.24e-4 |
| 37 | 0.0466 | 4.19e-4 |
| 38 | 0.0463 | 4.15e-4 |
| 39 | 0.0460 | 4.10e-4 |
| 40 | 0.0457 | 4.05e-4 |
| 41 | 0.0454 | 4.00e-4 |
| 42 | 0.0451 | 3.95e-4 |
| 43 | 0.0448 | 3.90e-4 |
| 44 | 0.0447 | 3.85e-4 |
| 45 | 0.0444 | 3.80e-4 |
| 46 | 0.0444 | 3.66e-4 |
| 47 | 0.0442 | 3.60e-4 |
| 48 | 0.0441 | 3.54e-4 |
| 49 | 0.0438 | 3.48e-4 |
| 50 | 0.0437 | 3.42e-4 |
| 51 | 0.0435 | 3.35e-4 |
| 52 | 0.0434 | 3.29e-4 |
| 53 | 0.0432 | 3.23e-4 |
| 54 | 0.0431 | 3.16e-4 |
| 55 | 0.0429 | 3.09e-4 |
| 56 | 0.0427 | 3.03e-4 |
| 57 | 0.0426 | 2.96e-4 |
| 58 | 0.0423 | 2.90e-4 |
| 59 | 0.0423 | 2.83e-4 |
| 60 | 0.0423 | 2.76e-4 |
| 61 | 0.0421 | 2.69e-4 |

### 4.2 Test-set evaluation results (every 5 epochs, S9/S11)

The training script evaluates the EMA model on the **test set** (S9/S11) every 5 epochs.
This is the H36M best-on-test convention (`select_on_test: true`).

| Epoch | Test MPJPE (EMA) | Best So Far | Checkpoint |
|---|---|---|---|
| 5 | 80.67mm | ✓ new best | best_anatproj_clean.pth |
| 10 | 56.12mm | ✓ new best | best_anatproj_clean.pth |
| 15 | 53.69mm | ✓ new best | best_anatproj_clean.pth |
| 20 | 52.42mm | ✓ new best | best_anatproj_clean.pth |
| 25 | 51.63mm | ✓ new best | best_anatproj_clean.pth |
| 30 | 51.25mm | ✓ new best | best_anatproj_clean.pth |
| 35 | **51.06mm** | ✓ **BEST** | best_anatproj_clean.pth |
| 40 | 51.07mm | — | — |
| 45 | 51.07mm | — | — |
| 50 | 51.30mm | — | — |
| 55 | 51.21mm | — | — |
| 60 | 51.29mm | — | — |

### 4.3 Training trajectory

- **Rapid initial drop**: Train loss 0.2371 → 0.0616 in first 10 epochs
- **Steady decline**: Train loss 0.0616 → **0.0421** by epoch 61
- **Test MPJPE plateau**: Best test MPJPE **51.06mm** at epoch 35; stuck around 51mm since then
- **Train-test gap widening**: Training loss keeps dropping (0.0421) while test stays flat (~51mm) — mild overfitting despite 0.15 dropout and 0.02 weight decay
- **Cosine LR schedule**: Peaked at 5e-4 (epoch 5), now at 2.69e-4 and decaying to ~0 by epoch 120
- **Best checkpoint**: `checkpoints/best_anatproj_clean.pth` (15.9 MB) at epoch 35 (51.06mm test)
- **Training speed**: ~1.05s/iter × 554 iters/epoch ≈ 9.7 min/epoch on NVIDIA L4
- **Throughput**: ~10h for 60 epochs, ~19-20h for full 120 epochs

### 4.4 Comparison to repo baseline

| Model | Test MPJPE | Note |
|---|---|---|
| Previous best (`best_anatproj_sota.pth`) | **48.2mm** | Trained with occlusion augmentation |
| This run (clean, best so far at ep 35) | **51.06mm** | Clean ceiling, occlusion aug OFF |
| Target (<1M SOTA cluster) | **~39-42mm** | SasMamba / PoseMamba-S |

The ~3mm gap to the previous baseline is expected — that model had occlusion augmentation
which acts as a regulariser. The GCN branch (`anatproj_gcn.yaml`) is the next lever to
close this gap.

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
| **AnatomyProj-Mamba CLEAN (this run)** | 0.97M | **51.06** (best, in progress) | — | this run |

Goal: bring the clean-CPN **test** MPJPE into the <1M cluster (SasMamba 41.5 / PoseMamba-S
41.8). Currently at 51.06mm after epoch 35 (plateaued). The GCN branch is the next lever.

---

## 6. Checkpoints produced

| File | Size | Description |
|---|---|---|
| `pretrained_anatomyproj_mamba_clean.pth` | 3.8 MB | MPM-pretrained backbone (25 epochs, loss 0.00049) |
| `best_anatproj_clean.pth` | 15.9 MB | Best checkpoint (epoch 35, 51.06mm test MPJPE) |
| `best_anatproj_sota.pth` | 15.9 MB | Previous repo baseline (48.2mm) |

---

## 7. What's next (pending fine-tune)

1. **Complete 120 epochs** (~10h remaining)
2. **Evaluate final checkpoint + best checkpoint**: `PYTHONPATH=$PWD python evaluate.py --config configs/anatproj_clean.yaml --checkpoint checkpoints/best_anatproj_clean.pth`
3. **GCN branch** (`anatproj_gcn.yaml`) — main accuracy lever to chase <1M SOTA (~39-42mm)
4. **Full occlusion-aware model** (`anatproj_occ.yaml`) with structured occlusion aug + spatial conf gate
5. **Occlusion ablation study**: per-limb occlusion sweep, noise robustness, confidence-off ablation

---

*Last updated: 2026-07-16 14:30 UTC*
