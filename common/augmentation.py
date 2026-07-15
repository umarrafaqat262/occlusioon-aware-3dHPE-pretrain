"""Data augmentation for 2D→3D pose lifting."""

import random
import math
import torch


# H36M left↔right joint swap indices for horizontal flip
_LEFT  = [4, 5, 6, 11, 12, 13]
_RIGHT = [1, 2, 3, 14, 15, 16]


def random_joint_mask(pose_2d, conf, ratio):
    """Zero out random joints. pose_2d: (B,T,J,2), conf: (B,T,J,1)."""
    B, T, J, _ = pose_2d.shape
    n = max(1, int(J * ratio))
    for b in range(B):
        idx = random.sample(range(J), n)
        pose_2d[b, :, idx] = 0.0
        conf[b, :, idx]    = 0.0
    return pose_2d, conf


def temporal_edge_dropout(pose_2d, conf, prob, max_span):
    """Drop contiguous temporal spans with probability prob."""
    if prob <= 0.0:
        return pose_2d, conf
    B, T, J, _ = pose_2d.shape
    for b in range(B):
        if random.random() < prob:
            span = random.randint(1, max(1, max_span))
            start = random.randint(0, max(0, T - span))
            pose_2d[b, start:start + span] = 0.0
            conf[b,    start:start + span] = 0.0
    return pose_2d, conf


# H36M limb joint groups for structured (spatio-temporally correlated) occlusion.
_LIMB_GROUPS = {
    'rleg': [1, 2, 3], 'lleg': [4, 5, 6],
    'larm': [11, 12, 13], 'rarm': [14, 15, 16], 'head': [9, 10],
}


def structured_limb_occlusion(pose_2d, conf, prob, min_span, max_span):
    """Occlude a WHOLE limb over a contiguous temporal window (realistic occlusion:
    a body part hidden for a while), independently per sample. Zeros both the 2D
    and the confidence so the conf-gated SSM must coast + the decoder must complete.
    Unlike `random_joint_mask` (a joint gone for ALL frames) and
    `temporal_edge_dropout` (ALL joints gone for a span), this is correlated on BOTH
    axes, matching real person-object occlusion and the eval protocols.
    pose_2d:(B,T,J,2), conf:(B,T,J,1)."""
    if prob <= 0.0:
        return pose_2d, conf
    B, T, J, _ = pose_2d.shape
    groups = list(_LIMB_GROUPS.values())
    for b in range(B):
        if random.random() < prob:
            joints = groups[random.randrange(len(groups))]
            span = random.randint(max(1, min_span), max(1, max_span))
            start = random.randint(0, max(0, T - span))
            sl = slice(start, start + span)
            pose_2d[b, sl][:, joints] = 0.0
            conf[b, sl][:, joints] = 0.0
    return pose_2d, conf


def severity_curriculum(epoch, total_epochs, min_ratio, max_ratio):
    """Linearly anneal masking severity from min→max over training."""
    t = min(epoch / max(total_epochs - 1, 1), 1.0)
    return min_ratio + t * (max_ratio - min_ratio)


def horizontal_flip(pose_2d, conf, pose_3d=None):
    """Flip left↔right. pose_2d: (B,T,J,2), conf: (B,T,J,1)."""
    pose_2d = pose_2d.clone()
    pose_2d[..., 0] = -pose_2d[..., 0]          # negate x
    # swap left/right joints
    tmp2 = pose_2d[:, :, _LEFT].clone()
    pose_2d[:, :, _LEFT]  = pose_2d[:, :, _RIGHT]
    pose_2d[:, :, _RIGHT] = tmp2
    tmp_c = conf[:, :, _LEFT].clone()
    conf[:, :, _LEFT]  = conf[:, :, _RIGHT]
    conf[:, :, _RIGHT] = tmp_c

    if pose_3d is not None:
        pose_3d = pose_3d.clone()
        pose_3d[..., 0] = -pose_3d[..., 0]
        tmp3 = pose_3d[:, :, _LEFT].clone()
        pose_3d[:, :, _LEFT]  = pose_3d[:, :, _RIGHT]
        pose_3d[:, :, _RIGHT] = tmp3
        return pose_2d, conf, pose_3d

    return pose_2d, conf


def random_2d_jitter(pose_2d, scale=0.05, shift=0.05, noise=0.01):
    """Per-sample 2D scale + shift + pixel-noise jitter (normalized coords).

    Regularizes against the train-subject overfit / camera-scale specificity by
    perturbing the 2D input distribution. pose_2d: (B,T,J,2)."""
    pose_2d = pose_2d.clone()
    B = pose_2d.shape[0]
    if scale > 0:
        s = 1.0 + (torch.rand(B, 1, 1, 1, device=pose_2d.device) * 2 - 1) * scale
        pose_2d = pose_2d * s
    if shift > 0:
        t = (torch.rand(B, 1, 1, 2, device=pose_2d.device) * 2 - 1) * shift
        pose_2d = pose_2d + t
    if noise > 0:
        pose_2d = pose_2d + torch.randn_like(pose_2d) * noise
    return pose_2d


def random_rotation_2d(pose_2d, max_deg=30):
    """DEPRECATED — rotates only 2D, leaving the 3D target unrotated, which breaks
    the 2D<->3D correspondence. Kept for backward-compat; use `random_rotation`."""
    pose_2d = pose_2d.clone()
    B = pose_2d.shape[0]
    for b in range(B):
        angle = random.uniform(-max_deg, max_deg) * math.pi / 180.0
        c, s = math.cos(angle), math.sin(angle)
        R = pose_2d.new_tensor([[c, -s], [s, c]])
        pose_2d[b] = pose_2d[b] @ R.T
    return pose_2d


def random_rotation(pose_2d, pose_3d, max_deg=20):
    """Geometrically-consistent in-plane rotation = a camera roll about the optical
    axis. Rotates the 2D keypoints AND the 3D target's (X,Y) by the SAME per-sample
    angle; depth Z is unchanged. Since the projection is x=fX/Z, y=fY/Z, rotating
    (X,Y) by R rotates (x,y) by the same R, so the 2D<->3D mapping stays valid
    (unlike `random_rotation_2d`). Coords are origin-centred (cf. `horizontal_flip`
    negating x), so rotation is about 0.
    pose_2d:(B,T,J,2), pose_3d:(B,T,J,3) -> rotated (pose_2d, pose_3d)."""
    pose_2d = pose_2d.clone()
    pose_3d = pose_3d.clone()
    B = pose_2d.shape[0]
    for b in range(B):
        angle = random.uniform(-max_deg, max_deg) * math.pi / 180.0
        c, s = math.cos(angle), math.sin(angle)
        R = pose_2d.new_tensor([[c, -s], [s, c]])
        pose_2d[b] = pose_2d[b] @ R.T
        pose_3d[b, ..., :2] = pose_3d[b, ..., :2] @ R.T
    return pose_2d, pose_3d
