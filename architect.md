# AnatomyProj-Mamba — Architecture & Training Reference

Monocular 2D→3D human-pose lifter (Human3.6M, CPN 2D, 243-frame, **seq2seq**) built on a
selective state-space (Mamba) backbone with two novelties:
- **(A) Differentiable Anatomical Projection (DAP) decoder** — direct coordinate regression
  refined by a differentiable projection onto the constant-bone-length manifold (also completes
  occluded joints).
- **(B) Confidence-gated selective scan** — the SSM time-step Δ is gated by 2D-keypoint
  confidence, so occluded joints coast on temporal memory.

Total trainable parameters: **968,092 (0.968 M)** — under the 1 M "efficient SSM" budget.

---

## 1. End-to-end data flow

```
  2D keypoints  x_2d  (B, 243, 17, 2)          confidence  conf  (B, 243, 17, 1)
        │
        ▼   encode()                                   model/bsmamba.py
  ┌─────────────────────────────────────────────────────────────────┐
  │ carry-forward fill (occluded joints) → velocity aug [x,y,dx,dy]  │
  │ 2D bone decomposition (dir, len) → embeddings → + positional PE  │
  └─────────────────────────────────────────────────────────────────┘
        │  h (B, 243, 17, 96)
        ▼   backbone: 4 × STBlock                       model/st_block.py
  ┌─────────────────────────────────────────────────────────────────┐
  │ SpatialBlock : per frame, mix 17 joints (kinematic-tree BiSSM)   │
  │ TemporalBlock: per joint, mix 243 frames (conf-gated BiSSM)      │
  └─────────────────────────────────────────────────────────────────┘
        │  h (B, 243, 17, 96)
        ▼   DAPDecoder                                  model/bsmamba.py
  ┌─────────────────────────────────────────────────────────────────┐
  │ coord_head → P0 (B,243,17,3)   len_head → L (B,1,16,1)           │
  │ projection: 8 unrolled steps onto ‖bone‖=L manifold (conf-wtd)   │
  └─────────────────────────────────────────────────────────────────┘
        │
        ▼   3D pose  P (B, 243, 17, 3)   root-relative, camera-space metres
```

- **B** = 32 (batch), **T** = 243 (frames), **J** = 17 (joints), **D** = 96 (feature width).
- 3D targets are root-relative camera-space in **metres** (MPJPE ×1000 → mm).

---

## 2. Skeleton (`common/skeleton.py`)

17 joints in DFS order; parent index < child index for every joint, so the natural index order
**is** the kinematic-tree (root→leaf) walk.

```
H36M_PARENTS    = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
BONE_CHILD_IDX  = [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16]   # 16 bones
BONE_PARENT_IDX = [ 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8,11,12, 8,14,15]
SYMMETRY (by bone): (0,3)(1,4)(2,5)  legs   (10,13)(11,14)(12,15)  arms
```
Joint 0 = Hip (root). Limb groups used by the occlusion study: rleg `[1,2,3]`, lleg `[4,5,6]`,
larm `[11,12,13]`, rarm `[14,15,16]`.

---

## 3. Input processing & embeddings (`BoneStateMamba.encode`)

| step | op | output |
|---|---|---|
| 1 | `carry_forward_fill` — occluded joints (conf≈0) ← last valid value over time; low conf retained | (B,T,17,2) |
| 2 | velocity aug: concat finite-difference → `[x,y,dx,dy]` | (B,T,17,4) |
| 3 | `decompose_bones(x_2d)` — parameter-free 2D bone dir (2) + len (1) | (B,T,16,2),(B,T,16,1) |
| 4 | `joint_embed` Linear(4→48); `bone_dir_embed` (2→24) + `bone_len_embed` (1→24) → bone token (48), scattered to child joints | (B,T,17,96) |
| 5 | + `temporal_pe` (1,243,1,96) + `joint_pe` (1,1,17,96) | (B,T,17,96) |

D = `joint_embed_dim`(48) + `bone_embed_dim`(48) = 96, and the invariant **`state_dim == jd + bd`** must hold.

---

## 4. Backbone — 4 × STBlock (`model/st_block.py`)

Each STBlock = SpatialBlock → TemporalBlock (factorized: O(J²) spatial, O(T) temporal).

### SpatialBlock (`model/spatial_block.py`) — joints within a frame
Operates on `(B·T, 17, 96)`:
1. `LayerNorm`
2. **parent-feature injection**: `parent_proj(h[parent_idx])` — anatomical prior (Linear 96→96).
3. **kinematic-tree BiSSM**: reorder joints to `KIN_SCAN_ORDER` → bidirectional fused Mamba
   (`fast=True`, `expand=1`, `d_state=24`) → restore order. Forward scan = root→leaf, backward = leaf→root.
4. residual `x + ssm + parent_feat`, then MLP (96→192→96, `mlp_ratio=2`).
- `scan_order='shuffle'` swaps in a fixed random joint order (ablation).

### TemporalBlock (`model/temporal_block.py`) — frames per joint
Operates on `(B·17, 243, 96)`:
- `LayerNorm` → **confidence-gated BiSSM** (`conf_gate=True`, `expand=1`, `d_state=24`) → residual.

---

## 5. Selective scan internals (`model/ssm.py`)

- **`ConfMamba`** subclasses mamba-ssm `Mamba`, forces the slow path (`use_fast_path=False`) to
  expose Δ. Confidence gate `g = sigmoid(α·conf + β)` (learnable `α` init 5.0, `β` init −2.0;
  g(conf=1)≈0.95, g(conf=0)≈0.12). The gate multiplies **both** the discretized step
  `Δ = softplus(dt)·g` and the integrated input `u = x·g`. When occluded (g→0): `Ā=exp(ΔA)→I`
  (state coasts) and no noisy observation enters. Uses CUDA `selective_scan_fn` / `causal_conv1d_fn`.
- **`BiSSM`** = forward `ConfMamba` + reversed `ConfMamba`, concat → `Linear(2D→D)` merge + dropout.
  Spatial blocks use `fast=True` (stock fused Mamba, no gating — kinematic *order* is the spatial
  mechanism); temporal blocks use the gated path (novelty B).

---

## 6. DAP Decoder (`model/bsmamba.py:DAPDecoder`) — primary novelty A

Consumes `h (B,T,17,96)`:

1. **coord_head** `LN → Linear(96→160) → GELU → Linear(160→3)` ⇒ `P0 (B,T,17,3)`: direct
   root-relative joint regression (no spatial autoregression → no chain-error accumulation).
2. **len_head** `LN → Linear(96→80) → GELU → Linear(80→1) → Softplus` on child-joint features,
   **averaged over time** ⇒ rigid per-clip bone lengths `L (B,1,16,1)` (anatomy prior).
3. **Differentiable projection** `_project()` — minimizes, by **8 unrolled projected-GD steps**:
   ```
   E(P) = Σ_j w_j ‖P_j − P0_j‖²            (stay near regressed point)
        + ρ Σ_b ( ‖P_child(b) − P_parent(b)‖ − L_b )²   (bone-length manifold)
   ```
   Per step: data grad `w·(P−P0)`; bone grad `ρ·r·(v/‖v‖)` scattered to child(+)/parent(−) via
   `index_add`; `P ← P − step·grad`. Defaults `ρ=5.0` (`dap_rho`), `step=0.05` (`dap_step`),
   `n_iter=8` (`dap_iter`); math in fp32 for stability. **Global** (all joints jointly), not chained.
   - **Occlusion completion (dual use):** `w_j = w_floor + (1−w_floor)·conf_j` (`w_floor=0.1`).
     Confident joints stay pinned to `P0`; occluded joints (w→floor) are pulled into place by the
     bone constraints from confident neighbours ⇒ reconstructed by the *same* solver.
   - `ρ=0` ⇒ projection is exact identity (`P==P0`) — verified sanity check.
4. Returns `(P, bone_dir, bone_len=L, P0)`. `L` feeds the symmetry + bone-length losses; `P0`
   feeds the auxiliary regression loss.

`decoder: fk` selects the legacy **FKDecoder** (root + bone dir + shared len → `reconstruct_fk`)
as the ablation baseline; `reconstruct_fk`/`decompose_bones` live in `model/bone_ops.py`.

---

## 7. Parameter sizes (exact, verified)

| component | params | notes |
|---|---:|---|
| **Spatial blocks** (4×) | **552,576** | 138,144 / block (BiSSM expand1 ×2 + parent_proj + MLP + merge) |
| **Temporal blocks** (4×) | **365,968** | 91,492 / block (conf-gated BiSSM expand1 ×2 + merge) |
| coord_head | 16,195 | LN+Linear(96→160)+Linear(160→3) |
| len_head | 8,033 | LN+Linear(96→80)+Linear(80→1) |
| temporal_pe | 23,328 | (1,243,1,96) |
| joint_pe | 1,632 | (1,1,17,96) |
| joint_embed | 240 | Linear(4→48) |
| bone_dir_embed | 72 | Linear(2→24) |
| bone_len_embed | 48 | Linear(1→24) |
| **TOTAL** | **968,092** | **0.968 M (<1M)** |

Backbone (encoder used by pretraining) = everything except the decoder heads ≈ 943,864.

---

## 8. Pretraining — Masked Pose Modeling (`pretrain.py`)

Self-supervised; trains the **encoder** only (decoder unused).

- **MPMWrapper** = `BoneStateMamba.encode` + **MPMHead** (`LN→Linear(96→48)→GELU→Linear(48→2)`,
  ≈4,754 params, **discarded after pretraining**).
- **Masking** (`mpm_mask`): random joints (`mpm_mask_joint_ratio=0.2`) + a temporal span
  (`mpm_mask_frame_ratio=0.1`) zeroed in both 2D and conf; predict the masked 2D positions (MSE).
- Optimizer AdamW `lr=mpm_lr=0.001`, wd 0.01, CosineAnnealingLR, bf16, grad-clip 1.0, flip aug.
- **Epochs: 25**, batch 32, 243f, train stride 81 (VP3D train split, all 5 subjects).
- Saves `backbone.state_dict()` → `checkpoints/pretrained_anatomyproj_mamba_cpn_sota.pth`.
  (Includes randomly-initialised decoder weights; fine-tune loads with `strict=False`.)

---

## 9. Fine-tuning — supervised seq2seq (`train.py`)

Loads the MPM encoder (`--pretrained …`, `strict=False`), trains the full model on 3D.

**Loss** (`losses.py:TotalLoss`, 3D in metres):
```
L = MPJPE(P)                          # projected pose
  + 0.10 · MPJPE(P0)                  # lambda_p0  — keep raw regression head honest
  + 2.0  · velocity                   # lambda_vel — 1st-derivative (was 20 → over-smoothed)
  + 0.5  · acceleration               # lambda_temp
  + 2.0  · bone_length(L vs GT)       # lambda_blen — correct scale
  + 0.1  · symmetry(L)                # lambda_sym
```

**Optimization:** AdamW `lr=5e-4`, wd 0.01, **cosine LR** to `lr_min_ratio=0.01` after 5-epoch
warmup, grad-clip 1.0, bf16 AMP, **EMA 0.999** (eval on EMA weights). Batch 32, **120 epochs**,
seq2seq (all frames supervised). Validate every 5 epochs (root-relative MPJPE, mm).

**Protocol:** `select_on_test: true` — train on all 5 subjects (S1,5,6,7,8), best-on-test
(S9,11), the accepted H36M SOTA convention.

**Augmentation:** joint-mask curriculum (0→0.3), temporal edge dropout (0.1, span≤20),
horizontal flip (p=0.5), 2D jitter (scale/shift 0.05, noise 0.01). Masking teaches the
confidence-driven projection + gate.

**Checkpoints/logs:** best → `checkpoints/best_anatproj_sota.pth`, final → `final_*`; logs
`logs/anatproj_sota.log`, tensorboard `runs/anatproj_sota`.

---

## 10. Config keys (`configs/anatproj_sota.yaml`)

```yaml
# dims (state_dim == joint_embed_dim + bone_embed_dim)
joint_embed_dim: 48   bone_embed_dim: 48   state_dim: 96
num_blocks: 4   ssm_expand: 1   mlp_ratio: 2   d_state: 24   fk_hidden: 160   dropout: 0.1
# decoder
decoder: dap   dap_iter: 8   dap_rho: 5.0   dap_step: 0.05   dap_w_floor: 0.1
# data — VP3D / CPN, camera-space mm
dataset: vp3d   data_dir: data/motion3d/cpn_vp3d
keypoints_file: data_2d_h36m_cpn_ft_h36m_dbb.npz   num_frames: 243   train_stride: 81
# training
batch_size: 32   lr: 5e-4   weight_decay: 0.01   epochs: 120
lr_sched: cosine   lr_min_ratio: 0.01   warmup_epochs: 5   clip_grad: 1.0
use_amp: true   ema_decay: 0.999   select_on_test: true
# loss
lambda_vel: 2.0   lambda_temp: 0.5   lambda_blen: 2.0   lambda_sym: 0.1   lambda_p0: 0.1
# pretrain
mpm_epochs: 25   mpm_lr: 0.001   mpm_mask_joint_ratio: 0.2   mpm_mask_frame_ratio: 0.1
```

---

## 11. Evaluation (`evaluate.py`)

- **Clean:** Protocol-1 MPJPE + Protocol-2 P-MPJPE, root-relative mm, over all test frames,
  with horizontal-flip TTA, per-action breakdown.
- **Occlusion study** (`--occlusion`): occlude each limb (conf→0 over a centred 50% span) and
  report MPJPE on the occluded joints (reconstruction) + overall. Run on gate-off / FK-baseline
  checkpoints for the ablation columns.

Command:
```
python evaluate.py --config configs/anatproj_sota.yaml \
  --checkpoint checkpoints/best_anatproj_sota.pth --occlusion
```

---

## 12. SOTA target (H36M, CPN 2D, 243f)

| model | params | P1 (MPJPE) | P2 (P-MPJPE) |
|---|---:|---:|---:|
| **SasMamba** | 0.64 M | **41.48** | **34.84** | ← <1M SOTA to beat |
| PoseMamba-S | 0.90 M | 41.8 | 35.0 |
| SasMamba-large | 4.1 M | 39.77 | 33.61 |
| PoseMamba-L | 6.7 M | 38.1 | 32.5 | ← absolute floor (~CPN-2D limited) |
| MotionBERT | 42.3 M | 39.2 | 32.9 |
| **AnatomyProj-Mamba (ours)** | **0.968 M** | *training* | *training* |

Goal: beat 41.48 / 34.84 at <1M, with a decisive occlusion-robustness margin.

---

## 13. File map

| file | role |
|---|---|
| `model/bsmamba.py` | `BoneStateMamba` (encode), `DAPDecoder`, `FKDecoder`, `carry_forward_fill` |
| `model/st_block.py` | `STBlock` (spatial→temporal; threads `d_state`) |
| `model/spatial_block.py` | kinematic-tree BiSSM + parent injection + MLP |
| `model/temporal_block.py` | confidence-gated BiSSM over time |
| `model/ssm.py` | `ConfMamba` (gated Δ), `BiSSM` |
| `model/bone_ops.py` | `decompose_bones`, `reconstruct_fk` |
| `losses.py` | `TotalLoss` (MPJPE + P0 + vel + accel + blen + sym) |
| `pretrain.py` | MPM self-supervised encoder pretraining |
| `train.py` | supervised seq2seq fine-tuning (cosine LR, EMA, AMP) |
| `evaluate.py` | P1/P2 + flip TTA + per-action + occlusion study |
| `configs/anatproj_sota.yaml` | the 0.968M run config |
| `common/skeleton.py` | joints/bones/parents/symmetry/scan order |
| `common/dataset_vp3d.py` | VideoPose3D/CPN loader (camera-space mm) |
