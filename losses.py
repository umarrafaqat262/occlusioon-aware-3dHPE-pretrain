"""Losses: MPJPE + bone-length consistency + symmetry + reprojection."""

import torch
import torch.nn as nn
from common.skeleton import BONE_SYMMETRY_PAIRS
from model.bone_ops import decompose_bones


def mpjpe(pred, gt):
    return (pred - gt).norm(dim=-1).mean()


def p_mpjpe(pred, gt):
    pred_c = pred - pred.mean(dim=-2, keepdim=True)
    gt_c = gt - gt.mean(dim=-2, keepdim=True)
    s_pred = pred_c.norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True).clamp(min=1e-6)
    s_gt = gt_c.norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True).clamp(min=1e-6)
    pred_c = pred_c / s_pred
    gt_c = gt_c / s_gt
    H = pred_c.transpose(-1, -2) @ gt_c
    U, S, Vh = torch.linalg.svd(H)
    d = torch.sign(torch.linalg.det(Vh.transpose(-1, -2) @ U.transpose(-1, -2)))
    D = torch.eye(3, device=pred.device).unsqueeze(0).unsqueeze(0).expand_as(H).clone()
    D[..., -1, -1] = d
    R = Vh.transpose(-1, -2) @ D @ U.transpose(-1, -2)
    pred_aligned = (pred_c @ R) * s_gt + gt.mean(dim=-2, keepdim=True)
    return (pred_aligned - gt).norm(dim=-1).mean()


def bone_length_loss(bone_len):
    mean_len = bone_len.mean(dim=1, keepdim=True)
    return ((bone_len - mean_len) ** 2).mean()


def symmetry_loss(bone_len, pairs=BONE_SYMMETRY_PAIRS):
    loss = 0.0
    for li, ri in pairs:
        loss = loss + ((bone_len[:, :, li] - bone_len[:, :, ri]) ** 2).mean()
    return loss / max(len(pairs), 1)


def reproj_loss(pred_3d, input_2d):
    return (pred_3d[..., :2] - input_2d).norm(dim=-1).mean()


def velocity_loss(pred_3d, gt_3d):
    """Motion (1st-derivative) consistency over time: matches per-frame velocity."""
    if pred_3d.shape[1] < 2:
        return pred_3d.new_zeros(())
    pv = pred_3d[:, 1:] - pred_3d[:, :-1]
    gv = gt_3d[:, 1:] - gt_3d[:, :-1]
    return (pv - gv).norm(dim=-1).mean()


def accel_loss(pred_3d, gt_3d):
    """Acceleration (2nd-derivative) consistency — temporal smoothness."""
    if pred_3d.shape[1] < 3:
        return pred_3d.new_zeros(())
    pa = pred_3d[:, 2:] - 2 * pred_3d[:, 1:-1] + pred_3d[:, :-2]
    ga = gt_3d[:, 2:] - 2 * gt_3d[:, 1:-1] + gt_3d[:, :-2]
    return (pa - ga).norm(dim=-1).mean()


class TotalLoss(nn.Module):
    """Seq2seq lifting loss: MPJPE over all frames + anatomy (symmetry) +
    motion/temporal consistency. 3D is in metres; component values are reported
    as-is (multiply mpjpe by 1000 for mm)."""

    def __init__(self, cfg):
        super().__init__()
        self.lb = getattr(cfg, 'lambda_bone', 0.0)
        self.ls = getattr(cfg, 'lambda_sym', 0.1)
        self.lr = getattr(cfg, 'lambda_reproj', 0.0)
        self.lv = getattr(cfg, 'lambda_vel', 0.0)
        self.lt = getattr(cfg, 'lambda_temp', 0.0)
        self.lblen = getattr(cfg, 'lambda_blen', 0.0)
        self.lp0 = getattr(cfg, 'lambda_p0', 0.1)

    def forward(self, pred_3d, gt_3d, bone_len, input_2d, pred_p0=None):
        # Root-align to match the evaluation metric (eval subtracts joint 0 before
        # MPJPE). Without this the DAP projection can drift the root and the train
        # objective disagrees with eval on the global-translation component.
        pred_3d = pred_3d - pred_3d[..., :1, :]
        gt_3d = gt_3d - gt_3d[..., :1, :]
        if pred_p0 is not None:
            pred_p0 = pred_p0 - pred_p0[..., :1, :]
        l1 = mpjpe(pred_3d, gt_3d)
        l3 = symmetry_loss(bone_len)
        lv = velocity_loss(pred_3d, gt_3d)
        lt = accel_loss(pred_3d, gt_3d)
        total = l1 + self.ls * l3 + self.lv * lv + self.lt * lt
        metrics = {'total': total.item(), 'mpjpe': l1.item(),
                   'sym': l3.item(), 'vel': lv.item(), 'temp': lt.item()}
        if pred_p0 is not None and self.lp0 > 0:
            # Keep the raw regression head (pre-projection P0) honest so the
            # anatomical projection refines a good estimate, not a degenerate one.
            lp0 = mpjpe(pred_p0, gt_3d)
            total = total + self.lp0 * lp0
            metrics['p0'] = lp0.item()
        if self.lblen > 0:
            # Directly supervise predicted bone lengths against GT bone lengths
            # (from the 3D target) — teaches correct per-subject scale, the main
            # fix for the test P1−P2 (global-scale) gap.
            _, gt_len = decompose_bones(gt_3d)               # (B,T,16,1)
            lbl = (bone_len - gt_len).abs().mean()
            total = total + self.lblen * lbl
            metrics['blen'] = lbl.item()
        if self.lb > 0:
            l2 = bone_length_loss(bone_len)
            total = total + self.lb * l2
            metrics['bone'] = l2.item()
        if self.lr > 0:
            l4 = reproj_loss(pred_3d, input_2d)
            total = total + self.lr * l4
            metrics['reproj'] = l4.item()
        return total, metrics
