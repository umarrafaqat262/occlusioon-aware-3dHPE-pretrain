# Setup Guide — Occlusion-Aware 3D HPE (AnatomyProj-Mamba)

## Machine & GPU

| Item | Value |
|---|---|
| Instance | AWS (NVIDIA L4) |
| GPU | NVIDIA L4 24 GB |
| CUDA | 13.2 (driver), CUDA runtime 13.2 |
| OS | Ubuntu 22.04 / 24.04 |
| CPU | x86_64 |

## Environment

**Environment name**: `posemamba` (under `/home/ubuntu/miniforge3/envs/posemamba`)

### Python & core packages

| Package | Version | Notes |
|---|---|---|
| Python | 3.10 | — |
| torch | 2.14.0.dev20260626+cu132 | Built with CUDA 13.2 support |
| torchvision | 0.20.0.dev20260626+cu132 | — |
| causal-conv1d | 1.4.0+cu132 | From source, compiled for sm_80+sm_89 |
| mamba-ssm | 2.3.2.post1 | From source, CUDA kernels compiled for sm_80+sm_89 |
| numpy | — | Latest via pip |
| pyyaml | — | — |
| tqdm | — | — |
| tensorboard | — | — |
| einops | — | — |

### Mamba SSM build

Mamba's selective-scan CUDA kernels were compiled from source with arch limits:

```
TORCH_CUDA_ARCH_LIST="8.0;8.9" pip install mamba-ssm==2.3.2.post1 --no-binary mamba-ssm
```

This targets Ampere (A100, L4=sm_89) and skips unsupported archs (sm_90, sm_100). Mamba3 import is disabled in the code (`try: import mamba_ssm...`).

## Dataset: Human3.6M (VideoPose3D / CPN protocol)

### Files

Located under `data/motion3d/cpn_vp3d/`:

| File | Size | Source |
|---|---|---|
| `data_3d_h36m.npz` | 407 MB | 3D world positions (ground truth) |
| `data_2d_h36m_cpn_ft_h36m_dbb.npz` | ~500 MB | CPN fine-tuned 2D detections |

### Data split (standard H36M)

- **Train**: Subjects S1, S5, S6, S7, S8
- **Test**: Subjects S9, S11
- **Protocol**: 243-frame sliding window, stride 81, seq2seq
- **Format**: NPZ (NumPy) with `_cache_v2_*.pkl` built on first loader run
- **Convention**: `select_on_test: true` — best checkpoint selected on test set (disclosed per H36M convention)

### Provenance

The `.npz` files were created by Weixing (2024) following the VideoPose3D / CPN pipeline. No modification was done to the files themselves.

## Repo structure

```
occlusioon-aware-3dHPE-pretrain/
├── configs/
│   ├── anatproj_clean.yaml     ← CLEAN ceiling config (this run)
│   ├── anatproj_gcn.yaml       ← GCN branch config
│   └── anatproj_occ.yaml       ← Full occlusion-aware config
├── model/
│   ├── bsmamba.py              ← Main model (BoneStateMamba) + DAPDecoder/FKDecoder + carry_forward_fill
│   ├── st_block.py             ← Spatio-temporal block (spatial→temporal)
│   ├── spatial_block.py        ← Joint mixer: BiSSM + parent prior + optional GCN branch
│   ├── temporal_block.py       ← Per-joint temporal SSM (single/multi-scale)
│   ├── ssm.py                  ← ConfMamba (confidence-gated) + BiSSM
│   ├── bone_ops.py             ← decompose_bones / reconstruct_fk
│   └── mamba_block.py          ← BiGRU fallback (unused by main model)
├── common/
│   ├── dataset_vp3d.py         ← ACTIVE loader (VideoPose3D/CPN, all configs use this)
│   ├── dataset.py              ← Legacy MotionBERT-pickle loader (SH path only)
│   ├── augmentation.py         ← flip/rotation/jitter + occlusion masking
│   ├── skeleton.py             ← H36M-17 parents/bones/scan order
│   └── vp3d/                   ← vendored VideoPose3D camera params
├── scripts/
│   └── occlusion_eval.py       ← Occlusion study runner (BlendMimic3D-style σ sweep)
├── data/motion3d/cpn_vp3d/     ← Dataset files
├── checkpoints/                ← Saved model weights
├── smoke_test.py               ← Pre-training readiness check
├── pretrain.py                 ← MPM pretraining script
├── train.py                    ← Supervised fine-tuning script
├── evaluate.py                 ← Test evaluation script
├── demo.py                     ← Real-video demo
└── RESULTS.md                  ← Detailed per-epoch results
```

## What has been completed

### ✅ Stage 1: MPM Pretrain (DONE)

| Item | Detail |
|---|---|
| Config | `anatproj_clean.yaml` |
| Epochs | 25 / 25 |
| Final loss | **0.00049** |
| Duration | ~4 hours |
| Output | `checkpoints/pretrained_anatomyproj_mamba_clean.pth` (3.8 MB) |

### ✅ Stage 2: Smoke test (DONE)

All three configs (`clean`, `gcn`, `occ`) pass the smoke test:
```
PYTHONPATH=$PWD python smoke_test.py
```

### ✅ Data preparation (DONE)

Both `.npz` files downloaded and placed in `data/motion3d/cpn_vp3d/`. No corruption, verified by training (loss decreases normally).

### ✅ Environment & build (DONE)

- conda env `posemamba` created and active
- `causal-conv1d` installed from source
- `mamba-ssm` 2.3.2.post1 compiled with CUDA kernels
- Smoke test confirms Mamba SSM import and forward/backward pass

## What is currently running

### 🏃 Stage 3: Supervised Fine-tune (IN PROGRESS)

| Item | Detail |
|---|---|
| Config | `anatproj_clean.yaml` |
| Epochs | **62 / 120** (~52% complete) |
| Current train loss | ~0.042 |
| Current LR | 2.69e-4 (cosine decaying to ~0) |
| Best test MPJPE (EMA) | **51.06mm** (epoch 35) |
| Started from | MPM-pretrained checkpoint |
| Batch size | 32 |
| GPU util | 100% |
| Speed | ~1.05s/iter, ~9.7 min/epoch |
| Elapsed | ~10 hours |
| Remaining | ~10 hours |
| PID | 241100 (running under `setsid`) |
| Log file | `/opt/dlami/nvme/train_clean_final2.log` |

## Test-set evaluation results (every 5 epochs, S9/S11)

The training script evaluates the EMA model on the test set every 5 epochs (H36M best-on-test convention).

| Epoch | Test MPJPE | Best |
|---|---|---|
| 5 | 80.67mm | ✓ |
| 10 | 56.12mm | ✓ |
| 15 | 53.69mm | ✓ |
| 20 | 52.42mm | ✓ |
| 25 | 51.63mm | ✓ |
| 30 | 51.25mm | ✓ |
| 35 | **51.06mm** | ✓ **BEST** |
| 40 | 51.07mm | — |
| 45 | 51.07mm | — |
| 50 | 51.30mm | — |
| 55 | 51.21mm | — |
| 60 | 51.29mm | — |

## Per-epoch training loss

| Epoch | Loss | LR |
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

**Best checkpoint**: `checkpoints/best_anatproj_clean.pth` (15.9 MB) — epoch 35, 51.06mm test MPJPE

## What is pending

| Step | Config | Depends on | Est. time |
|---|---|---|---|---|
| **Complete 120 epochs** | `anatproj_clean.yaml` | Running now | ~10h |
| **Evaluate best + final** | `evaluate.py` | Fine-tune completes | ~30 min |
| GCN branch train | `anatproj_gcn.yaml` | Clean eval results | ~20h |
| Occlusion model train | `anatproj_occ.yaml` | GCN results | ~20h |
| Occlusion ablation study | `scripts/occlusion_eval.py` | Occlusion model | ~2h |
| GT-2D sanity gate | `anatproj_gcn_gt.yaml` | Any fine-tuned model | ~30 min |

## How to replicate

### Fresh setup

```bash
git clone https://github.com/umarrafaqat262/occlusioon-aware-3dHPE-pretrain.git
cd occlusioon-aware-3dHPE-pretrain

# Environment
conda create -n posemamba python=3.10 -y
conda activate posemamba
pip install torch==2.14.0.dev20260626+cu132 torchvision --index-url https://download.pytorch.org/whl/cu132
pip install causal-conv1d>=1.4.0
TORCH_CUDA_ARCH_LIST="8.0;8.9" pip install mamba-ssm==2.3.2 --no-binary mamba-ssm
pip install numpy pyyaml tqdm tensorboard einops

# Data
mkdir -p data/motion3d/cpn_vp3d
# Download data_3d_h36m.npz and data_2d_h36m_cpn_ft_h36m_dbb.npz into that directory

# Verify
PYTHONPATH=$PWD python smoke_test.py

# Train
PYTHONPATH=$PWD python pretrain.py --config configs/anatproj_clean.yaml
PYTHONPATH=$PWD python train.py --config configs/anatproj_clean.yaml \
    --pretrained checkpoints/pretrained_anatomyproj_mamba_clean.pth --tag anatproj_clean
```

### Resume training (after reboot)

```bash
conda activate posemamba
cd /tmp/opencode/occlusioon-aware-3dHPE-pretrain
setsid python -u train.py --config configs/anatproj_clean.yaml \
    --pretrained checkpoints/pretrained_anatomyproj_mamba_clean.pth \
    --tag anatproj_clean \
    > /opt/dlami/nvme/train_clean_final2.log 2>&1 &
```

### Monitor

```bash
tail -f /opt/dlami/nvme/train_clean_final2.log           # live log
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader  # GPU status
ps aux | grep train.py                                     # process alive?
```
