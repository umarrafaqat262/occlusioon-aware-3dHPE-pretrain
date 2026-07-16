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
│   ├── anatomyproj_mamba.py    ← Main model
│   ├── mpm.py                  ← MPM pretraining head
│   ├── ssm.py                  ← Mamba SSM blocks
│   └── dap.py                  ← DAP decoder
├── common/
│   └── dataset.py              ← Legacy dataset loader
├── scripts/
│   └── occlusion_eval.py       ← Occlusion study runner
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
| Epochs | **22 / 120** (~18% complete) |
| Current loss | ~0.05 |
| Current MPJPE | ~40–45mm (training, last-iteration) |
| Started from | MPM-pretrained checkpoint |
| Batch size | 32 |
| Learning rate | 4.8e-4 (cosine schedule, peaked at 5e-4) |
| GPU util | 100% |
| Speed | ~1.05s/iter, ~9.7 min/epoch |
| Elapsed | ~3.5 hours |
| Remaining | ~16 hours |
| PID | 241100 (running under `setsid`) |
| Log file | `/opt/dlami/nvme/train_clean_final2.log` |

## Results so far (per-epoch)

| Epoch | Loss | MPJPE | LR |
|---|---|---|---|
| 1 | 0.2371 | 124.7mm | 1.0e-4 |
| 2 | 0.1170 | 81.5mm | 2.0e-4 |
| 3 | 0.0918 | 75.9mm | 3.0e-4 |
| 4 | 0.0816 | 68.7mm | 4.0e-4 |
| 5 | 0.0755 | 54.7mm | 5.0e-4 |
| 6 | 0.0707 | 60.8mm | 5.0e-4 |
| 7 | 0.0676 | 57.1mm | 5.0e-4 |
| 8 | 0.0653 | 52.3mm | 5.0e-4 |
| 9 | 0.0634 | 56.2mm | 4.99e-4 |
| 10 | 0.0616 | 59.8mm | 4.99e-4 |
| 11 | 0.0603 | 50.2mm | 4.98e-4 |
| 12 | 0.0590 | 49.1mm | 4.97e-4 |
| 13 | 0.0577 | 47.3mm | 4.95e-4 |
| 14 | 0.0569 | 44.6mm | 4.94e-4 |
| 15 | 0.0559 | 44.3mm | 4.93e-4 |
| 16 | 0.0553 | 41.2mm | 4.91e-4 |
| 17 | 0.0543 | 44.3mm | 4.89e-4 |
| 18 | 0.0537 | 45.6mm | 4.87e-4 |
| 19 | 0.0532 | 43.5mm | 4.85e-4 |
| 20 | 0.0527 | 43.9mm | 4.82e-4 |
| 21 | 0.0521 | **41.9mm** | 4.80e-4 |

**Checkpoint saved**: `checkpoints/best_anatproj_clean.pth` (15.9 MB)

## What is pending

| Step | Config | Depends on | Est. time |
|---|---|---|---|
| **Evaluate** | `anatproj_clean.yaml` | Fine-tune completes | ~30 min |
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
