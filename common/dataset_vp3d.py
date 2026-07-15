"""VideoPose3D-convention Human3.6M loader (CPN / GT 2D).

Produces the standard monocular-lifting protocol used by PoseMamba / SasMamba /
MixSTE / VideoPose3D so MPJPE is reported in real millimetres, comparable to the
published CPN numbers:

  * 3D target  : world positions → camera space (per camera extrinsics) →
                 root-relative, in metres (×1000 = mm at eval).
  * 2D input   : detector keypoints normalised with normalize_screen_coordinates
                 (X/w*2 - [1, h/w]).  Same convention as the rest of the repo.
  * split      : train = S1,S5,S6,S7,S8 ; test = S9,S11.
  * 17 joints  : VP3D remove_static_joints subset, whose order already equals the
                 repo's H36M_PARENTS / left-right convention (verified).

Camera parameters are vendored from VideoPose3D (common/vp3d, CC BY-NC).
A processed cache is written next to the npz so this only runs once.
"""

import os, pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from common.vp3d.h36m_dataset import (
    h36m_cameras_intrinsic_params as _INTR,
    h36m_cameras_extrinsic_params as _EXTR,
)

TRAIN_SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8']
TEST_SUBJECTS = ['S9', 'S11']
# 32 → 17 joint subset (VP3D remove_static_joints); order matches repo convention
KEPT_JOINTS = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]


# ── quaternion rotation (numpy port of VP3D qrot/qinverse) ───────────────────
def _qinverse(q):
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def _qrot(q, v):
    """Rotate v (..,3) by quaternion q (..,4)."""
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (q[..., :1] * uv + uuv)


def _world_to_camera(X, R, t):
    """X (N,J,3) world metres, R quaternion (4,), t (3,) metres → camera (N,J,3)."""
    Rt = _qinverse(R)
    Rt = np.tile(Rt, (*X.shape[:-1], 1))
    return _qrot(Rt, X - t)


def _normalize_2d(X, w, h):
    return X / w * 2 - np.array([1.0, h / w], dtype=np.float32)


# ── cache builder ─────────────────────────────────────────────────────────────
def _build_cache(data_dir, keypoints_file):
    p3d = np.load(os.path.join(data_dir, 'data_3d_h36m.npz'), allow_pickle=True)['positions_3d'].item()
    kp = np.load(os.path.join(data_dir, keypoints_file), allow_pickle=True)['positions_2d'].item()

    out = {'train': _empty(), 'test': _empty()}
    for subj in TRAIN_SUBJECTS + TEST_SUBJECTS:
        split = 'train' if subj in TRAIN_SUBJECTS else 'test'
        for action, pos_w in p3d[subj].items():
            pos_w = pos_w[:, KEPT_JOINTS].astype(np.float32)        # (N,17,3) world m
            cams = _EXTR[subj]
            for c in range(len(cams)):
                R = np.array(cams[c]['orientation'], dtype=np.float32)
                t = np.array(cams[c]['translation'], dtype=np.float32) / 1000.0  # mm→m
                res_w, res_h = _INTR[c]['res_w'], _INTR[c]['res_h']
                cam_id = _INTR[c]['id']

                p3 = _world_to_camera(pos_w, R, t)                  # (N,17,3) cam m
                p3 = p3 - p3[:, :1]                                  # root-relative

                p2_full = np.asarray(kp[subj][action][c], dtype=np.float32)  # (M,17,2|3)
                # Use a real detector-confidence channel if present (some CPN/GT
                # npz store (x,y,score)); else fall back to all-ones.
                if p2_full.shape[-1] >= 3:
                    p2 = p2_full[..., :2]
                    cscore = p2_full[..., 2:3]
                else:
                    p2 = p2_full[..., :2]
                    cscore = np.ones((*p2.shape[:-1], 1), dtype=np.float32)
                n = min(len(p2), len(p3))
                p2, p3, cscore = p2[:n], p3[:n], cscore[:n]
                p2 = _normalize_2d(p2, res_w, res_h)

                src = f"{subj}_{action}_{cam_id}"
                out[split]['joint_2d'].append(p2)
                out[split]['pose_3d'].append(p3)
                out[split]['conf'].append(cscore.astype(np.float32))
                out[split]['source'].append(np.array([src] * n))
    for sp in out:
        for k in out[sp]:
            out[sp][k] = np.concatenate(out[sp][k], axis=0)
    return out


def _empty():
    return {'joint_2d': [], 'pose_3d': [], 'conf': [], 'source': []}


def _split_clips(source, n_frames, stride, cover_tail=False):
    db = {}
    for i, s in enumerate(source):
        db.setdefault(s, []).append(i)
    clips = []
    for frames in db.values():
        frames = np.array(frames)
        if len(frames) < n_frames:
            continue
        last = 0
        for start in range(0, len(frames) - n_frames + 1, stride):
            clips.append(frames[start:start + n_frames]); last = start
        # cover_tail: add an end-anchored window so no tail frames are dropped
        # at eval (full frame-set parity with SOTA, which pads the remainder).
        if cover_tail and last + n_frames < len(frames):
            clips.append(frames[len(frames) - n_frames:])
    return np.array(clips, dtype=np.int64)


class VP3DDataset(Dataset):
    """Human3.6M (VideoPose3D convention). conf is all-ones (CPN file has no
    confidence); occlusion robustness is studied via synthetic masking."""

    def __init__(self, data_dir, split, num_frames, stride,
                 keypoints_file='data_2d_h36m_cpn_ft_h36m_dbb.npz',
                 subset='all', val_fraction=0.0):
        """subset: 'all' | 'train' | 'val'. When split=='train' and val_fraction>0,
        whole source videos are held out for val (leakage-free, deterministic)."""
        # cache version bumped to v2 (now stores a real confidence channel)
        cache = os.path.join(data_dir, f"_cache_v2_{keypoints_file.replace('.npz','')}.pkl")
        if not os.path.isfile(cache):
            dt = _build_cache(data_dir, keypoints_file)
            with open(cache, 'wb') as f:
                pickle.dump(dt, f, protocol=4)
        else:
            with open(cache, 'rb') as f:
                dt = pickle.load(f)

        raw = dt[split]
        source = raw['source']
        # cover the tail at eval so every test frame is scored (frame-set parity)
        clips = _split_clips(source, num_frames, stride, cover_tail=(split == 'test'))

        if split == 'train' and val_fraction > 0 and subset in ('train', 'val'):
            uniq = sorted(set(source.tolist()))
            step = max(int(round(1.0 / val_fraction)), 2)   # every step-th source → val
            val_sources = set(uniq[::step])
            clip_src = source[clips[:, 0]]                   # source of each clip
            in_val = np.array([s in val_sources for s in clip_src])
            clips = clips[in_val] if subset == 'val' else clips[~in_val]

        self.clips = clips                                 # (C,T) global frame idx
        self.split = split
        self.pose_2d = raw['joint_2d'][clips]              # (C,T,17,2)
        self.pose_3d = raw['pose_3d'][clips]               # (C,T,17,3) metres
        if 'conf' in raw and raw['conf'].shape[0] == raw['joint_2d'].shape[0]:
            self.conf = raw['conf'][clips].astype(np.float32)   # (C,T,17,1) real/ones
        else:
            self.conf = np.ones((*self.pose_2d.shape[:3], 1), dtype=np.float32)
        self.sources = source[clips[:, 0]]                 # (C,) source id per clip

    def dedup_mask(self):
        """(C,T) bool: True the FIRST time each global frame is seen across clips.

        `cover_tail` adds an overlapping end-anchored window at eval, so some tail
        frames appear in two clips. Averaging MPJPE over all clip×frame positions
        would double-count them. Multiply per-frame errors by this mask (and divide
        by its sum) to score every unique test frame exactly once."""
        flat = self.clips.reshape(-1)
        _, first = np.unique(flat, return_index=True)
        keep = np.zeros(flat.shape[0], dtype=bool)
        keep[first] = True
        return keep.reshape(self.clips.shape)

    def __len__(self):
        return len(self.pose_2d)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.pose_2d[idx]),
                torch.from_numpy(self.pose_3d[idx]),
                torch.from_numpy(self.conf[idx]))
