# CSM-Pose Redesign — Build & Run Guide (for the GPU agent)

**Branch:** `csm-pose-redesign`. Pull this branch on the GPU box to build/train the new model.

## ⚠️ READ FIRST — the all-modules run diverged; do this instead
The first attempt (`csm_s.yaml` with lr 5e-4 and all 8 modules on) **diverged** — loss exploded
at epoch 16, VAL MPJPE hit 1.7–5.7 METERS, best was only 63.4 mm at epoch 5. Root cause:
lr too high for the deeper/module-rich net + several unbounded modules, all enabled at once.

**ROOT CAUSE (confirmed by two independent audits):** the divergence was caused by
**`lambda_vel: 20`**, not the architecture. Proof: the identical backbone trained perfectly
stably during MPM pretrain at lr=1e-3 (5× higher) — MPM just doesn't use the velocity loss.
Our `velocity_loss` is an *unsquared* L2 norm (non-vanishing gradient, blind to absolute
position), so at 20× it makes AdamW limit-cycle and the position drift → gradual runaway.
**Now fixed: `lambda_vel: 2.0`** in both configs (the proven-stable value; csm_base also uses
`dap_iter: 8`). Also kept as defensive measures: lr 2e-4 / warmup 10 / clip 0.5, softmax-bounded
SSI, LayerNorm KPA, averaged limb-reorder, and a `train.py` **divergence guard** (per-step +
VAL-based) that now aborts automatically — including gradual climbs.

**Mandatory procedure — do NOT enable all modules at once again:**
1. Run **`configs/csm_base.yaml` first** (backbone only, all new flags OFF). It MUST train stably —
   target ~44–48 mm test. If even the base diverges, lower lr further (1e-4) and report.
2. Then enable **ONE module at a time** (copy csm_base.yaml, flip one flag), ~25-epoch runs, in the
   order in Step 3 below. Keep a module only if it is stable AND lowers TEST MPJPE (or helps occlusion).
3. Assemble the final config from the survivors. `csm_s.yaml` (all modules) is the *end goal*, not
   the starting point — only run it once the ablation shows the modules are individually stable.

## Why this redesign
The old `AnatomyProj-Mamba` (0.97M) plateaued at **~51 mm test** and overfit — an architecture
problem. The two efficient winners hit ~41.5 mm at a *smaller* budget by spending it well:
**SasMamba 0.64M → 41.48**, **PoseMamba-S 0.86M → 41.8**. This redesign adopts their efficient
recipe **and** adds an occlusion novelty.

**Honest targets** (see the plan for full reasoning):
- Clean CPN, <1M → **~40 mm = best-in-class sub-1M** (beat SasMamba 41.48). Clean CPN is
  detection-noise-bound, so the absolute record (37.1 mm @ 26.5M) is **not** a <1M target.
- Occlusion → **SOTA by construction** (no occlusion-aware Mamba lifter exists).

## What changed vs the old model
Deep-narrow backbone (D=64, 6 blocks, **expand=2** — was 96/4/expand-1) + these modules, each
**flag-gated and ablatable**, each near-free (<a few K params):

| Flag (config) | Module | Source | Role |
|---|---|---|---|
| `use_dct` / `dct_keep_ratio` | DCT low-pass denoise front-end (parameter-free) | PoseFormerV2/SCT | attacks CPN 2D noise |
| `spatial_kpa` | KPA ModulatedGCN prior on joint tokens (~1.6K) | KTPFormer | local skeletal structure |
| `spatial_lap_pe` (k) | Laplacian eigenvector PE (parameter-free vecs) | PerturbPE | where each joint sits on the skeleton |
| `spatial_limb_reorder` | additive global + limb-chain scan | PoseMamba | anatomically-ordered recurrence |
| `spatial_ssi` | **confidence-weighted state fusion across joints (~0.3K)** | SAMA + ours | **novel** — occluded joint inferred from confident neighbours |
| `temporal_motion` | motion-adaptive Δ (MSM, ~2 params/block) | SAMA | fast joints update state more |
| `spatial_conf_gate` | spatial occlusion coasting | ours (novelty B) | occluded joints coast spatially |
| `use_infill` | 3D occlusion in-fill head | LInKs | complete occluded joints in 3D |

The novelty for the paper is the **confidence-conditioned SSI + MSM + conf-gate** (no Mamba
method makes the recurrence uncertainty-aware) plus the DCT denoiser — all at <1M.

## Files
New: `model/dct.py`, `model/graph_priors.py` (KPA + Laplacian PE), `configs/csm_s.yaml`.
Changed: `model/ssm.py` (MSM), `model/spatial_block.py` (KPA/PE/limb-reorder/SSI),
`model/st_block.py`, `model/temporal_block.py`, `model/bsmamba.py` (DCT + in-fill + flag wiring),
`common/skeleton.py` (edges, limb index), `smoke_test.py`.

## Step 1 — verify build + budget FIRST (mandatory)
```bash
git checkout csm-pose-redesign && git pull
PYTHONPATH=$PWD python smoke_test.py
```
This builds `csm_s` on synthetic data and prints its param count + which modules are ON.
**The model MUST be < 1M params.** If `csm_s` prints ≥ 1,000,000, edit `configs/csm_s.yaml`:
lower `num_blocks` (6→5) or `state_dim` (64→56) until it fits, then re-run smoke_test. (The
config is sized to target ~0.85–0.95M but the exact count depends on the mamba-ssm build — verify.)

## Step 2 — train (clean-first) + evaluate
```bash
PYTHONPATH=$PWD python pretrain.py --config configs/csm_s.yaml
PYTHONPATH=$PWD python train.py    --config configs/csm_s.yaml \
    --pretrained checkpoints/pretrained_csm_pose_s.pth --tag csm_s
PYTHONPATH=$PWD python evaluate.py --config configs/csm_s.yaml \
    --checkpoint checkpoints/best_csm_s.pth            # flip-TTA + dedup TEST MPJPE
```
**Report the TEST metric** (`VAL (EMA) MPJPE` every 5 epochs + the final `evaluate.py` number) —
NOT the training-batch MPJPE. Compare against SasMamba 41.48 (the number to beat).

## Step 3 — ablation (this is the paper's evidence, and de-risks the design)
Build the modules up one at a time — keep each only if it lowers TEST MPJPE. Copy `csm_s.yaml`
and toggle ONE flag per run, in this order (cheapest/highest-confidence first):
1. base (all new flags false, `ssm_expand:2`, 6 blocks) — the efficient backbone alone.
2. + `spatial_limb_reorder` → + `spatial_kpa` → + `spatial_lap_pe:6` → + `temporal_motion`
   → + `use_dct` → + `spatial_ssi` → + `spatial_conf_gate` → + `use_infill`.
Record each run's TEST MPJPE delta in RESULTS.md. If a module doesn't help clean, keep it only
if it helps the occlusion metric (Step 4).

## Step 4 — occlusion (the headline)
```bash
# synthetic per-limb + noise sweep on H36M
PYTHONPATH=$PWD python evaluate.py --config configs/csm_s.yaml \
    --checkpoint checkpoints/best_csm_s.pth --occlusion
PYTHONPATH=$PWD python scripts/occlusion_eval.py --config configs/csm_s.yaml \
    --checkpoint checkpoints/best_csm_s.pth
```
For the real headline, prepare **3DPW-OCC / 3DOH / BlendMimic3D** (a loader mirroring
`common/dataset_vp3d.py` still needs to be added — flag when you get there) and report
occluded-vs-visible MPJPE vs D3DP / MotionBERT / PoseFormerV2.

## Rules (same as GUIDE_GPU.md)
- Don't modify model code silently — the CPU box reviews commits. Flag issues in RESULTS.md.
- Report TEST metrics, not training MPJPE. Don't fabricate file names/numbers.
- Pin exact versions in SETUP.md.
- All new flags default OFF, so `anatproj_*` configs are byte-for-byte unchanged (clean A/B baseline).

## Notes / risks
- Module gains may not fully stack — the Step-3 ablation is the safeguard; drop any module that
  neither helps clean nor occlusion.
- Confidence modules need real per-joint confidence; on H36M-CPN (conf≈1) they're near-identity
  and get exercised by the occlusion aug / real benchmarks.
- Not yet implemented (future): DropPath, the real-occlusion dataset loaders, an optional -B
  (D=96) variant for an absolute-accuracy row.
