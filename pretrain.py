"""Masked Pose Modeling (MPM) pretraining.

Self-supervised: mask random joints/frames in 2D input, train the encoder
to reconstruct the masked 2D positions. No 3D labels needed.
Saves encoder weights for use with: train.py --pretrained checkpoints/pretrained_*.pth
"""

import os, argparse, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.utils import load_config, set_seed, count_parameters
from common.dataset import H36MDataset
from common.augmentation import horizontal_flip, random_rotation_2d
from model.bsmamba import BoneStateMamba
from model.bone_ops import decompose_bones
from common.skeleton import BONE_CHILD_IDX


# ─────────────────────────────────────────────────────────────────────────────
class MPMHead(nn.Module):
    """Lightweight reconstruction head: predicts masked 2D joint positions."""

    def __init__(self, d_model, num_joints=17):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),   # predict (x, y)
        )

    def forward(self, h):
        """h: (B, T, J, D) -> (B, T, J, 2)"""
        return self.head(h)


class MPMWrapper(nn.Module):
    """BoneStateMamba encoder + MPM reconstruction head."""

    def __init__(self, backbone: BoneStateMamba):
        super().__init__()
        self.backbone = backbone
        D = backbone.cfg.state_dim
        J = backbone.cfg.num_joints
        self.mpm_head = MPMHead(D, J)

    def forward(self, x_2d, conf):
        """Encode with the shared backbone, then reconstruct masked 2D."""
        h = self.backbone.encode(x_2d, conf)
        return self.mpm_head(h)     # (B, T, J, 2)


# ─────────────────────────────────────────────────────────────────────────────
def mpm_mask(pose_2d, conf, joint_ratio=0.2, frame_ratio=0.1, noise_std=0.0):
    """Mask random joints and temporal spans, and (MotionBERT recipe) inject
    Gaussian noise on the VISIBLE joints so the encoder learns to denoise as well
    as in-fill. Reconstruction target stays the clean pose_2d. Returns masked
    (+noised) input, masked conf, and the boolean mask of reconstructed positions."""
    pose_masked = pose_2d.clone()
    conf_masked = conf.clone()
    B, T, J, _ = pose_2d.shape
    mask = torch.zeros(B, T, J, dtype=torch.bool, device=pose_2d.device)

    # Joint masking
    n_joint = max(1, int(J * joint_ratio))
    for b in range(B):
        joints = random.sample(range(J), n_joint)
        pose_masked[b, :, joints] = 0.0
        conf_masked[b, :, joints] = 0.0
        mask[b, :, joints] = True

    # Frame masking
    n_frames = max(1, int(T * frame_ratio))
    for b in range(B):
        start = random.randint(0, T - n_frames)
        pose_masked[b, start:start+n_frames] = 0.0
        conf_masked[b, start:start+n_frames] = 0.0
        mask[b, start:start+n_frames] = True

    # Denoising: perturb the still-visible joints (not the masked/zeroed ones).
    if noise_std > 0.0:
        visible = (~mask).unsqueeze(-1)
        pose_masked = pose_masked + torch.randn_like(pose_masked) * noise_std * visible

    return pose_masked, conf_masked, mask


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/small.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.mpm_epochs = args.epochs

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = getattr(cfg, 'model_name', 'BoneStateMamba')
    print(f"\nMPM Pretraining: {model_name}  |  {cfg.mpm_epochs} epochs  |  {device}")

    # ── Data (train split only — self-supervised) ─────────────────
    if getattr(cfg, 'dataset', None) == 'vp3d':
        from common.dataset_vp3d import VP3DDataset
        train_set = VP3DDataset(cfg.data_dir, 'train', cfg.num_frames,
                                getattr(cfg, 'train_stride', 81), cfg.keypoints_file)
    else:
        train_set = H36MDataset(cfg.data_dir, 'train', cfg.num_frames, 81, cfg.dt_file)
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        shuffle=True, num_workers=2, pin_memory=True, drop_last=True,
    )

    # ── Model ─────────────────────────────────────────────────────
    backbone = BoneStateMamba(cfg).to(device)
    model    = MPMWrapper(backbone).to(device)
    print(f"Backbone params: {count_parameters(backbone):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.mpm_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.mpm_epochs)
    criterion = nn.MSELoss()

    os.makedirs('checkpoints', exist_ok=True)
    run_tag = model_name.lower().replace(' ', '_').replace('-', '_')
    writer  = SummaryWriter(f'runs/pretrain_{run_tag}')

    # ── Pretraining loop ──────────────────────────────────────────
    for epoch in range(cfg.mpm_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader,
                    desc=f"[MPM {model_name}] Epoch {epoch+1}/{cfg.mpm_epochs}")

        for pose_2d, pose_3d, conf in pbar:
            pose_2d = pose_2d.to(device)
            conf    = conf.to(device)

            # Optional flip augmentation
            if random.random() < getattr(cfg, 'flip_prob', 0.5):
                pose_2d, conf = horizontal_flip(pose_2d, conf)

            # Mask
            pose_masked, conf_masked, mask = mpm_mask(
                pose_2d, conf,
                cfg.mpm_mask_joint_ratio,
                cfg.mpm_mask_frame_ratio,
                getattr(cfg, 'mpm_noise_std', 0.0),
            )

            # Forward (AMP bf16)
            with torch.autocast('cuda', dtype=torch.bfloat16,
                                enabled=getattr(cfg, 'use_amp', True)):
                pred_2d = model(pose_masked, conf_masked)        # (B, T, J, 2)
                loss = criterion(pred_2d[mask].float(), pose_2d[mask].float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        scheduler.step()
        avg = epoch_loss / len(train_loader)
        writer.add_scalar('pretrain/loss', avg, epoch)
        print(f"\n[MPM] Epoch {epoch+1}  avg_loss={avg:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

    # Save backbone weights only
    ckpt_path = f'checkpoints/pretrained_{run_tag}.pth'
    torch.save(backbone.state_dict(), ckpt_path)
    print(f"\nPretraining done. Backbone saved to {ckpt_path}")
    print(f"Use: python train.py --config configs/... --pretrained {ckpt_path}")
    writer.close()


if __name__ == '__main__':
    main()
