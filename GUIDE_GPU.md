# GPU Run Guide — Occlusion-Aware 3D HPE (AnatomyProj-Mamba)

This guide covers environment setup, data, the bug fixes that just landed, and the
exact commands to reproduce and push the results on a CUDA GPU (the `mamba_ssm`
CUDA kernels require an NVIDIA GPU; the model cannot run on CPU).

---

## FOR THE GPU-SIDE AGENT (opencode) — READ THIS FIRST

You are running on the GPU box. The editing/audit machine (CPU-only, no torch/GPU/data)
fixed the code and wrote this guide; **you pull the code and run it here.** Here is the
full picture so you act correctly.

### Why this work exists (the problem)
The project is an **occlusion-aware, <1M-param, Mamba** 2D→3D pose lifter (Human3.6M, CPN,
243f), for a paper. The prior best was **48.2 mm test MPJPE**, ~7–10 mm behind the <1M
SOTA cluster (SasMamba 41.5, PoseMamba-S 41.8) at the **same** param budget — so the gap
was architecture + training recipe + **bugs**, not capacity. An audit found real bugs and
missing pieces; they have now been fixed on the CPU box and pushed.

### What was wrong and what got fixed (already in the code you pull)
- **occlusion_eval.py / demo.py crashed** (`pred,_,_` vs a 4-tuple) → fixed. The occlusion
  study and demo now run.
- **Eval double-counted tail frames** (`cover_tail` overlap) → fixed with a frame-dedup mask.
- **Training loss wasn't root-aligned** while eval was → fixed (matches the metric now).
- **Confidence-gate init** was 0.95/0.12 → now ≈1 at conf=1, ≈0 at conf=0.
- **Spatial confidence gate was dead code** (occluded joints leaked into neighbours) → now
  toggled by `spatial_conf_gate`.
- **Selection is on the TEST set** under `select_on_test` (H36M best-on-test convention) —
  now honestly documented + logged; disclose it in the paper.
- **New capabilities** added, all config-gated: local-joint **GCN branch** (`spatial_gcn`),
  **structured limb occlusion** aug (`aug_structured_occ`), **masked+noise MPM**
  (`mpm_noise_std`), real-confidence loader support. Three new configs: `anatproj_clean`,
  `anatproj_gcn`, `anatproj_occ`.

### YOUR RULES (do this, avoid that)
1. **`git pull` before each run** and **do not modify the model/training code** — run it as
   pushed. If you believe something needs changing, write the reason in RESULTS.md and flag
   it; don't silently diverge (the CPU box reviews your commits).
2. **Report the RIGHT metric.** The progress-bar `mpjpe=...` is a **TRAINING-batch** number
   and is optimistic (~16 mm train/test gap here). The real, comparable numbers are:
   - the **`VAL (EMA) MPJPE`** line the trainer logs **every 5 epochs**, and
   - the final **`evaluate.py`** output (flip-TTA + frame-dedup) after training completes.
   Put THOSE in RESULTS.md. Never present a training MPJPE as if it were the test result,
   and never claim the 48.2 mm baseline (a test number) is beaten from a training number.
3. **Don't fabricate.** Only write file names, numbers, and results you actually observe.
   (A previous SETUP.md invented model files that don't exist — corrected. Don't repeat.)
4. **Pin versions** you actually used in SETUP.md (exact torch / mamba-ssm / causal-conv1d);
   avoid nightly builds if a stable one works — the paper needs reproducibility.
5. **Push what's verifiable:** commit the concise epoch log (the `VAL (EMA)` lines) and, if
   size allows, the best checkpoint — prose alone can't be reproduced.
6. **Run `smoke_test.py` first** (§2b) and only start a long run if it prints PASS.

### YOUR TASK QUEUE (in order)
1. Finish the `anatproj_clean` run (§3a) → run `evaluate.py` → record VAL(EMA)+eval test MPJPE.
2. `anatproj_gcn` (§3b) — chase the <1M cluster (~38–42 mm test).
3. `anatproj_occ` (§3c) — the occlusion-aware model; then the occlusion studies
   (`evaluate.py --occlusion` and `scripts/occlusion_eval.py`) and the conf-off ablation.
4. GT-2D sanity gate (§3d) — a healthy lifter hits ~15–17 mm on GT 2D; if not, tell the
   CPU box (backbone bottleneck).
5. Real occlusion benchmarks (§6) — flag if datasets aren't available.
Report results per stage in RESULTS.md, then `git push`.

---

## 0. What changed (audit fixes applied)

**Correctness bugs fixed**
- **occlusion_eval.py / demo.py crash** — `pred, _, _ = model(...)` unpacked 3 from a
  4-tuple → `ValueError`. Now `pred, *_`. The occlusion σ-sweep study and the demo run.
- **Eval tail double-counting** — `cover_tail` overlaps duplicate frames; `evaluate.py`
  and `train.py` now use `VP3DDataset.dedup_mask()` to score each unique test frame once.
- **Train/eval mismatch** — `losses.py` now root-aligns pred & GT (matches the eval metric).
- **Confidence gate init** — `ssm.py` gate now ≈1 at conf=1 and ≈0 at conf=0 (was 0.95/0.12).
- **Honest selection** — `train.py` docstring + a startup log now state clearly whether the
  checkpoint is selected on TEST (best-on-test convention, disclose it) or a leakage-free
  train-holdout.

**New capabilities (config-gated, default OFF → old configs reproduce exactly)**
- `spatial_conf_gate: true` — occluded joints are down-weighted in the **spatial** scan too
  (previously dead code); fixes occluded joints leaking into bone-chain neighbours.
- `spatial_gcn: true` (+`gcn_hidden`) — local-joint **GCN branch** fused with the spatial SSM
  (Pose Magic / HGMamba / MDTF recipe). Bottleneck keeps the model **< 1M params**.
- `aug_structured_occ: true` — **spatio-temporally correlated** limb occlusion augmentation.
- `mpm_noise_std` — MotionBERT-style **masked + noise** MPM pretraining.
- `VP3DDataset` now reads a **real confidence** channel from the 2D npz if present (else ones).

**New configs**
- `configs/anatproj_clean.yaml` — clean-ceiling run (occlusion aug OFF).
- `configs/anatproj_gcn.yaml`   — clean + GCN branch (chase the <1M clean SOTA).
- `configs/anatproj_occ.yaml`   — full occlusion-aware model (GCN + spatial gate + structured
  aug + noisy MPM) — the paper's headline.

---

## 1. Environment

```bash
conda create -n bsmamba python=3.10 -y
conda activate bsmamba
# CUDA 11.8 build used originally (match your GPU/driver)
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install mamba-ssm==2.2.2 causal-conv1d>=1.2.0
pip install numpy pyyaml tqdm tensorboard einops
# for the demo only:
pip install opencv-python imageio imageio-ffmpeg matplotlib rtmlib onnxruntime-gpu
```

Verify the CUDA kernels import:
```bash
python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba ok')"
```

---

## 2. Data (Human3.6M, VideoPose3D / CPN protocol)

Place these under `data/motion3d/cpn_vp3d/` (the `data_dir` in the configs):
- `data_3d_h36m.npz`                          (3D world positions)
- `data_2d_h36m_cpn_ft_h36m_dbb.npz`          (CPN fine-tuned 2D — the main protocol)
- `data_2d_h36m_gt.npz`                        (GT 2D — for the sanity gate below), optional

The loader builds a `_cache_v2_*.pkl` next to the npz on first run (a few minutes).
> The old `_cache_*.pkl` (v1) is stale after this update — the v2 cache adds the
> confidence channel and is created automatically; you can delete v1 caches.

Standard split: train = S1,5,6,7,8 ; test = S9,S11. MPJPE is real millimetres.

---

## 2b. Smoke test (run FIRST — ~30s, no dataset needed)

Verifies each config builds, stays < 1M params, runs a forward + backward with a
finite decreasing loss, that augmentation works, and that the new occlusion
features are active. Gate every long run behind it:

```bash
PYTHONPATH=$PWD python smoke_test.py                       # all 3 configs
PYTHONPATH=$PWD python smoke_test.py && echo "ready to train"
```
Exits non-zero on any failure, so `python smoke_test.py && python train.py ...` is safe.

## 3. Reproduce & improve — command sequence

All commands run from the repo root with `PYTHONPATH=$PWD`. Each is one GPU job
(~8 min/epoch on an A10G at 243f, batch 32 → ~16h for 120 epochs).

### 3a. Clean ceiling (measure true clean-CPN capability)
```bash
PYTHONPATH=$PWD python pretrain.py --config configs/anatproj_clean.yaml
PYTHONPATH=$PWD python train.py    --config configs/anatproj_clean.yaml \
    --pretrained checkpoints/pretrained_anatomyproj_mamba_clean.pth --tag anatproj_clean
PYTHONPATH=$PWD python evaluate.py --config configs/anatproj_clean.yaml \
    --checkpoint checkpoints/best_anatproj_clean.pth
```

### 3b. GCN branch (chase the <1M clean SOTA — main accuracy lever)
```bash
PYTHONPATH=$PWD python pretrain.py --config configs/anatproj_gcn.yaml
PYTHONPATH=$PWD python train.py    --config configs/anatproj_gcn.yaml \
    --pretrained checkpoints/pretrained_anatomyproj_mamba_gcn.pth --tag anatproj_gcn
PYTHONPATH=$PWD python evaluate.py --config configs/anatproj_gcn.yaml \
    --checkpoint checkpoints/best_anatproj_gcn.pth
```

### 3c. Full occlusion-aware model (headline)
```bash
PYTHONPATH=$PWD python pretrain.py --config configs/anatproj_occ.yaml
PYTHONPATH=$PWD python train.py    --config configs/anatproj_occ.yaml \
    --pretrained checkpoints/pretrained_anatomyproj_mamba_occ.pth --tag anatproj_occ
# clean metrics
PYTHONPATH=$PWD python evaluate.py --config configs/anatproj_occ.yaml \
    --checkpoint checkpoints/best_anatproj_occ.pth
# structured per-limb occlusion study
PYTHONPATH=$PWD python evaluate.py --config configs/anatproj_occ.yaml \
    --checkpoint checkpoints/best_anatproj_occ.pth --occlusion --occ-frac 0.5
# BlendMimic3D-style noise sweep + conf-aware vs conf-off ablation (now runs)
PYTHONPATH=$PWD python scripts/occlusion_eval.py --config configs/anatproj_occ.yaml \
    --checkpoint checkpoints/best_anatproj_occ.pth
```

### 3d. GT-2D sanity gate (health check of the lifter)
Point `keypoints_file` at `data_2d_h36m_gt.npz` (copy `anatproj_gcn.yaml` → `*_gt.yaml`)
and eval. A healthy 243f lifter reaches ~15–17 mm on GT 2D (PoseMamba-L = 15.6). If it
doesn't, the backbone — not occlusion — is the bottleneck.

### 3e. Real-video demo
```bash
PYTHONPATH=$PWD python demo.py --video clip.mp4 --config configs/anatproj_occ.yaml \
    --checkpoint checkpoints/best_anatproj_occ.pth --occlude rarm     # A/B with --conf-off
```

---

## 4. Ablation matrix (for the paper)

Run each by toggling one config key; compare clean MPJPE + occluded-joint MPJPE.

| Run | key change | measures |
|---|---|---|
| base | anatproj_clean | current architecture, clean ceiling |
| +GCN | anatproj_gcn (`spatial_gcn`) | local-joint modeling gain |
| +spatial gate | `spatial_conf_gate: true` | spatial occlusion coasting |
| +structured aug | `aug_structured_occ: true` | correlated occlusion training |
| +noisy MPM | `mpm_noise_std: 0.01` | denoising pretext |
| full | anatproj_occ | everything together (headline) |
| conf-off | eval with `--conf-off` (demo) / conf=1 | isolates novelty B |

Report **clean** and **occlusion** tables separately. Diffusion baselines (D3DP): cite
the **J-Agg** number, not the oracle **J-Best**. Disclose `select_on_test`.

---

## 5. Targets

- Clean CPN 243f: current 48.2 / 37.9 mm → aim for the <1M cluster: **SasMamba 41.5 / 34.8**,
  **PoseMamba-S ~38–39**. GCN + finished cosine schedule + reduced overfit gap are the levers.
- Occlusion: current occluded-joint MPJPE mean ~154 mm (arms 174–205). Spatial gate +
  structured aug + noisy MPM should cut this sharply; arms are the key metric.

## 6. Real occlusion benchmarks (dependency to obtain)

For the paper's headline claim, evaluate on real occlusion data — **not** in this repo yet:
3DPW-OCC, 3DOH, BlendMimic3D (has per-keypoint occlusion labels; matches
`scripts/occlusion_eval.py`), VOccl3D. Compare vs D3DP (most occlusion-robust per
arXiv 2504.10350), MotionBERT, PoseFormerV2. Add a loader mirroring `VP3DDataset`.

## 7. Known limitations left as-is
- Legacy `common/dataset.py` (Stacked-Hourglass pickle path, `tiny.yaml`) still scales 3D by
  `res_w` while `train.py` ×1000 — its "mm" is meaningless. Use the VP3D loader (all current
  configs do). Retire or fix the legacy path if you revive SH data.
- `temporal_edge_dropout` drops a span anywhere (name is historical); `reproj_loss`/`p_mpjpe`
  in `losses.py` are dead (weight 0 / uncalled).
