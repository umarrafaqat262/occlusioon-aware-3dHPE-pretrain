"""KinFK-Mamba training — seq2seq monocular 3D-HPE (VideoPose3D / CPN protocol).

- supervises ALL frames (seq2seq), not just the centre frame
- checkpoint selection depends on the config:
    * select_on_test: true  -> best-on-TEST (S9/S11) every 5 epochs. This is the
      common H36M "best-on-test" reporting convention, NOT a leakage-free protocol;
      it is an oracle early-stop and must be disclosed as such in any paper. The
      final-epoch checkpoint is always saved too (final_*.pth) for honest reporting.
    * select_on_test: false -> best on a leakage-free val split held out from the
      TRAIN subjects (S1/5/6/7/8); the test set is never touched during training.
- AdamW + warmup + cosine (or exponential) LR decay, AMP (bf16), EMA
- MPJPE reported in millimetres (3D targets are in metres)
"""

import os, argparse, random, logging, sys, copy
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None

from common.utils import load_config, set_seed, count_parameters
from common.augmentation import (
    random_joint_mask, temporal_edge_dropout, severity_curriculum,
    horizontal_flip, random_rotation, random_2d_jitter,
    structured_limb_occlusion,
)
from model.bsmamba import BoneStateMamba
from losses import TotalLoss


def setup_logger(log_path):
    logger = logging.getLogger('bsm')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path, encoding='utf-8'); fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); logger.addHandler(ch)
    return logger


class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)


def build_loaders(cfg):
    if getattr(cfg, 'dataset', None) == 'vp3d':
        from common.dataset_vp3d import VP3DDataset
        kp = cfg.keypoints_file
        if getattr(cfg, 'select_on_test', False):
            # SOTA protocol: train on ALL 5 train subjects, select best-on-test
            # (the accepted H36M convention used by SasMamba/PoseMamba).
            train_set = VP3DDataset(cfg.data_dir, 'train', cfg.num_frames,
                                    getattr(cfg, 'train_stride', 81), kp,
                                    subset='all', val_fraction=0.0)
            val_set = VP3DDataset(cfg.data_dir, 'test', cfg.num_frames,
                                  cfg.num_frames, kp)
        else:
            vf = getattr(cfg, 'val_fraction', 0.05)
            train_set = VP3DDataset(cfg.data_dir, 'train', cfg.num_frames,
                                    getattr(cfg, 'train_stride', 81), kp,
                                    subset='train', val_fraction=vf)
            val_set = VP3DDataset(cfg.data_dir, 'train', cfg.num_frames,
                                  cfg.num_frames, kp, subset='val', val_fraction=vf)
    else:
        from common.dataset import H36MDataset
        train_set = H36MDataset(cfg.data_dir, 'train', cfg.num_frames, 81, cfg.dt_file)
        val_set = H36MDataset(cfg.data_dir, 'test', cfg.num_frames, cfg.num_frames, cfg.dt_file)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    return train_loader, val_loader


def augment(pose_2d, pose_3d, conf, cfg, epoch):
    mask_ratio = severity_curriculum(epoch, cfg.epochs,
                                     cfg.aug_joint_mask_min, cfg.aug_joint_mask_max)
    if mask_ratio > 0:
        pose_2d, conf = random_joint_mask(pose_2d.clone(), conf.clone(), mask_ratio)
    pose_2d, conf = temporal_edge_dropout(pose_2d, conf,
                                          cfg.aug_edge_dropout, cfg.aug_edge_max_span)
    if getattr(cfg, 'aug_structured_occ', False):
        pose_2d, conf = structured_limb_occlusion(
            pose_2d, conf,
            getattr(cfg, 'aug_structured_prob', 0.5),
            getattr(cfg, 'aug_structured_min_span', 10),
            getattr(cfg, 'aug_structured_max_span', 60))
    if random.random() < getattr(cfg, 'flip_prob', 0.5):
        pose_2d, conf, pose_3d = horizontal_flip(pose_2d, conf, pose_3d)
    if getattr(cfg, 'rotation_aug', False):
        pose_2d, pose_3d = random_rotation(pose_2d, pose_3d,
                                           getattr(cfg, 'rotation_max_deg', 20))
    if getattr(cfg, 'jitter_aug', False):
        pose_2d = random_2d_jitter(pose_2d,
                                   getattr(cfg, 'jitter_scale', 0.05),
                                   getattr(cfg, 'jitter_shift', 0.05),
                                   getattr(cfg, 'jitter_noise', 0.01))
    return pose_2d, pose_3d, conf


@torch.no_grad()
def evaluate(model, loader, device):
    """Seq2seq MPJPE over all frames, root-relative, in millimetres.

    Uses the dataset's dedup mask so overlapping cover_tail frames are scored once
    (otherwise the tail is double-counted and the val number is biased)."""
    model.eval()
    ds = loader.dataset
    keep2d = ds.dedup_mask() if hasattr(ds, 'dedup_mask') else None
    tot, n = 0.0, 0
    cptr = 0
    for pose_2d, pose_3d, conf in loader:
        b = len(pose_2d)
        pose_2d, pose_3d, conf = pose_2d.to(device), pose_3d.to(device), conf.to(device)
        pred, *_ = model(pose_2d, conf)
        pred = pred - pred[:, :, :1]            # root-relative
        gt = pose_3d - pose_3d[:, :, :1]
        err = (pred - gt).norm(dim=-1).mean(dim=-1)     # (B,T)
        if keep2d is not None:
            km = torch.from_numpy(keep2d[cptr:cptr + b]).to(err.device)   # (b,T)
            tot += (err * km).sum().item(); n += int(km.sum().item())
        else:
            tot += err.sum().item(); n += err.numel()
        cptr += b
    return tot / n * 1000.0                       # mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/cpn.yaml')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--pretrained', default=None, help='MPM pretrained encoder')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--tag', default=None, help='checkpoint/run tag override')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.epochs = args.epochs

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # quality-neutral speed knobs (no effect on the bf16 math / results)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model_name = getattr(cfg, 'model_name', 'KinFK-Mamba')
    os.makedirs('checkpoints', exist_ok=True); os.makedirs('logs', exist_ok=True)
    run_tag = args.tag or model_name.lower().replace(' ', '_').replace('-', '_')
    log = setup_logger(f'logs/{run_tag}.log')
    log.info('=' * 60); log.info(f"  {model_name}  |  {device}  |  tag={run_tag}")

    train_loader, val_loader = build_loaders(cfg)
    log.info(f"  train batches {len(train_loader)}  val batches {len(val_loader)}")
    if getattr(cfg, 'select_on_test', False):
        log.info("  [SELECTION] select_on_test=TRUE -> best checkpoint chosen on the "
                 "TEST set (S9/S11). This is the H36M best-on-test convention (oracle "
                 "early-stop), disclose it; final_*.pth is also saved.")
    else:
        log.info("  [SELECTION] leakage-free: best checkpoint chosen on a train-subject "
                 "val holdout; the test set is untouched during training.")

    model = BoneStateMamba(cfg).to(device)
    n_params = count_parameters(model)
    log.info(f"  Parameters: {n_params:,}  ({n_params/1e6:.3f}M)"); log.info('=' * 60)

    if args.pretrained and os.path.isfile(args.pretrained):
        state = torch.load(args.pretrained, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info(f"Loaded pretrained | missing={len(missing)} unexpected={len(unexpected)}")

    criterion = TotalLoss(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = EMA(model, getattr(cfg, 'ema_decay', 0.999))
    use_amp = getattr(cfg, 'use_amp', True)
    lr_decay = getattr(cfg, 'lr_decay', 0.99)
    warmup = getattr(cfg, 'warmup_epochs', 5)

    writer = SummaryWriter(f'runs/{run_tag}')
    use_wandb = wandb is not None and os.environ.get('WANDB_DISABLED') != '1'
    if use_wandb:
        wandb.init(project=os.environ.get('WANDB_PROJECT', 'kinfk-mamba'),
                   name=run_tag, config=vars(cfg),
                   mode=os.environ.get('WANDB_MODE', 'online'))
    best_val = float('inf'); start_epoch = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1; best_val = ckpt.get('best_mpjpe', float('inf'))
        if 'ema' in ckpt: ema.shadow = ckpt['ema']
        log.info(f"Resumed from epoch {start_epoch}")

    import math
    lr_sched = getattr(cfg, 'lr_sched', 'cosine')
    eta = getattr(cfg, 'lr_min_ratio', 0.01)
    # divergence guard: skip non-finite / exploding steps; abort if it persists
    run_mean_loss = None
    bad_steps = 0
    val_bad = 0
    diverged = False
    for epoch in range(start_epoch, cfg.epochs):
        # LR: linear warmup, then cosine-to-floor (default) or exponential decay.
        if epoch < warmup:
            lr = cfg.lr * (epoch + 1) / warmup
        elif lr_sched == 'cosine':
            progress = (epoch - warmup) / max(1, cfg.epochs - warmup)
            lr = cfg.lr * (eta + (1 - eta) * 0.5 * (1 + math.cos(math.pi * progress)))
        else:
            lr = cfg.lr * (lr_decay ** (epoch - warmup))
        for g in optimizer.param_groups:
            g['lr'] = lr

        model.train(); epoch_loss = 0.0; metrics = {}
        pbar = tqdm(train_loader, desc=f"Ep{epoch+1}/{cfg.epochs}", ncols=90, file=sys.stdout)
        for pose_2d, pose_3d, conf in pbar:
            pose_2d, pose_3d, conf = pose_2d.to(device), pose_3d.to(device), conf.to(device)
            pose_2d, pose_3d, conf = augment(pose_2d, pose_3d, conf, cfg, epoch)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                pred_3d, bone_dir, bone_len, pred_p0 = model(pose_2d, conf)
                loss, metrics = criterion(pred_3d, pose_3d, bone_len, pose_2d, pred_p0)
            # divergence guard — skip non-finite or exploding (>20x running mean) steps
            lv = loss.item()
            exploding = (run_mean_loss is not None) and (lv > 20.0 * run_mean_loss)
            if (not math.isfinite(lv)) or exploding:
                bad_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if bad_steps >= 50:
                    log.info(f"[DIVERGENCE] loss={lv:.4g} (run_mean={run_mean_loss}) for "
                             f"{bad_steps} steps at epoch {epoch+1} — aborting training. "
                             f"Lower lr / check module stability.")
                    diverged = True
                    break
                continue
            bad_steps = 0
            run_mean_loss = lv if run_mean_loss is None else 0.99 * run_mean_loss + 0.01 * lv
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            optimizer.step(); ema.update(model)
            epoch_loss += metrics['total']
            pbar.set_postfix(loss=f"{metrics['total']:.4f}", mpjpe=f"{metrics['mpjpe']*1000:.1f}")

        if diverged:
            break
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('train/loss', avg_loss, epoch); writer.add_scalar('train/lr', lr, epoch)
        for k, v in metrics.items(): writer.add_scalar(f'train/{k}', v, epoch)
        if use_wandb:
            wandb.log({'epoch': epoch, 'train/loss': avg_loss, 'lr': lr,
                       **{f'train/{k}': v for k, v in metrics.items()}}, step=epoch)
        log.info(f"Epoch {epoch+1:3d}/{cfg.epochs}  loss={avg_loss:.4f}  lr={lr:.6f}")

        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            ema_model = copy.deepcopy(model); ema.copy_to(ema_model)
            val_mm = evaluate(ema_model, val_loader, device)
            writer.add_scalar('val/mpjpe_mm', val_mm, epoch)
            log.info(f"  VAL (EMA)  MPJPE = {val_mm:.2f} mm   (best {best_val:.2f})")
            # VAL-based divergence abort (catches gradual climbs the per-step guard misses)
            if best_val != float('inf') and val_mm > 2.0 * best_val:
                val_bad += 1
                log.info(f"  [DIVERGENCE] VAL {val_mm:.1f}mm > 2x best {best_val:.1f}mm "
                         f"({val_bad} consecutive check(s))")
                if val_bad >= 2:
                    log.info("  [DIVERGENCE] validation diverged — aborting training.")
                    diverged = True
            else:
                val_bad = 0
            if use_wandb:
                wandb.log({'val/mpjpe_mm': val_mm, 'val/best_mm': min(val_mm, best_val)}, step=epoch)
            if val_mm < best_val:
                best_val = val_mm
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'ema': ema.shadow, 'optimizer': optimizer.state_dict(),
                            'best_mpjpe': best_val, 'config': vars(cfg)},
                           f'checkpoints/best_{run_tag}.pth')
                log.info(f"  NEW BEST -> checkpoints/best_{run_tag}.pth")
            del ema_model
            if diverged:
                break

    # final-epoch checkpoint (transparency: report final + best, not only best-on-test)
    torch.save({'epoch': cfg.epochs - 1, 'model': model.state_dict(),
                'ema': ema.shadow, 'optimizer': optimizer.state_dict(),
                'best_mpjpe': best_val, 'config': vars(cfg)},
               f'checkpoints/final_{run_tag}.pth')
    log.info(f"DONE  best {best_val:.2f} mm  | final ckpt -> checkpoints/final_{run_tag}.pth")
    writer.close()
    if use_wandb:
        wandb.summary['best_mpjpe_mm'] = best_val; wandb.finish()


if __name__ == '__main__':
    main()
