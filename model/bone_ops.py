"""Bone decomposition and FK reconstruction."""

import torch
from common.skeleton import BONE_CHILD_IDX, BONE_PARENT_IDX, H36M_PARENTS


def decompose_bones(joints, child_idx=BONE_CHILD_IDX, parent_idx=BONE_PARENT_IDX):
    """joints: (B, T, J, C) → bone_dir: (B,T,16,C), bone_len: (B,T,16,1)"""
    child = joints[:, :, child_idx]
    parent = joints[:, :, parent_idx]
    vec = child - parent
    length = vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    direction = vec / length
    return direction, length


def reconstruct_fk(root_pos, bone_dir, bone_len, parents=H36M_PARENTS):
    """FK: root_pos (B,T,3), bone_dir (B,T,16,3), bone_len (B,T,16,1) → (B,T,17,3)"""
    B, T = root_pos.shape[:2]
    J = len(parents)
    joints = torch.zeros(B, T, J, 3, device=root_pos.device, dtype=root_pos.dtype)
    joints[:, :, 0] = root_pos
    bone_pairs = [(i, parents[i]) for i in range(J) if parents[i] >= 0]
    for bone_idx, (child, parent) in enumerate(bone_pairs):
        joints[:, :, child] = joints[:, :, parent] + bone_dir[:, :, bone_idx] * bone_len[:, :, bone_idx]
    return joints
