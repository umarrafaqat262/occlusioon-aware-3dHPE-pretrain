"""BoneStateMamba V2 — Spatial+Temporal architecture with velocity input."""

import torch
import torch.nn as nn
from model.bone_ops import decompose_bones, reconstruct_fk
from model.st_block import STBlock
from model.ssm import BiSSM
from common.skeleton import BONE_CHILD_IDX, BONE_PARENT_IDX, H36M_PARENTS


class BodyToken(nn.Module):
    """Per-frame global body context (V2, novelty A2): pool joints → MLP →
    broadcast-add back, so every joint sees whole-body orientation/depth context.
    Targets the global depth/orientation error (large P1−P2 gap). Shared across
    blocks. x:(B,T,J,D) → (B,T,J,D)."""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        g = x.mean(dim=2, keepdim=True)                 # (B,T,1,D) pool over joints
        g = self.fc2(self.act(self.fc1(self.norm(g))))  # (B,T,1,D)
        return x + self.drop(g)                          # broadcast over joints


def carry_forward_fill(pose, conf, eps=1e-6):
    """Replace occluded joints (conf≈0) with their last valid value over time.

    Occluded inputs would otherwise feed zeros/noise into the embeddings. We fill
    them with the last valid observation (forward fill; leading gaps use the next
    valid one). The original low `conf` is kept so the confidence-gated SSM still
    coasts on these positions — fill + gate are complementary (novelty B + SasMamba
    "invalid→last valid", here along time).

    pose: (B, T, J, C), conf: (B, T, J, 1) → filled pose (B, T, J, C).
    """
    B, T, J, C = pose.shape
    valid = conf.squeeze(-1) > eps                              # (B, T, J)
    ar = torch.arange(T, device=pose.device).view(1, T, 1).expand(B, T, J)
    fwd = torch.where(valid, ar, torch.full_like(ar, -1))
    fwd = torch.cummax(fwd, dim=1).values                       # last valid t (or -1)
    bwd = torch.where(valid, ar, torch.full_like(ar, T))
    bwd = torch.flip(torch.cummin(torch.flip(bwd, [1]), dim=1).values, [1])  # next valid t
    src = torch.where(fwd >= 0, fwd, bwd).clamp(0, T - 1)        # (B, T, J)
    src = src.unsqueeze(-1).expand(B, T, J, C)
    return torch.gather(pose, 1, src)


class FKDecoder(nn.Module):
    """Predict bone directions + lengths → reconstruct via FK."""

    def __init__(self, d_model, num_joints=17, num_bones=16, hidden=256):
        super().__init__()
        # Root position from global average pooling of joint features
        self.root_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.dir_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.len_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Softplus(),
        )

    def forward(self, h, conf=None):
        """h: (B, T, J, D) → joints (B,T,J,3), bone_dir, bone_len, P0(=None)"""
        B, T = h.shape[:2]
        # Root: average over all joint features
        root = self.root_head(h.mean(dim=2))    # (B, T, 3)

        # Bone features from child-joint tokens
        h_bones = h[:, :, BONE_CHILD_IDX]       # (B, T, 16, D)

        bone_dir = self.dir_head(h_bones)        # (B, T, 16, 3)
        bone_dir = bone_dir / bone_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Per-clip-shared bone lengths: bones are rigid, so predict one length
        # per bone for the whole clip and broadcast over time. This makes
        # bone-length constancy an architectural prior (not just a soft loss)
        # and is the FK-coupling half of novelty (A).
        bone_len = self.len_head(h_bones).mean(dim=1, keepdim=True)   # (B,1,16,1)
        bone_len = bone_len.expand(B, T, -1, -1)                       # (B,T,16,1)

        joints_3d = reconstruct_fk(root, bone_dir, bone_len)
        return joints_3d, bone_dir, bone_len, None


class DAPDecoder(nn.Module):
    """Differentiable Anatomical Projection decoder (primary novelty A).

    Two heads on the backbone features:
      * coord_head → P0: direct root-relative joint regression (the accuracy
        workhorse FK lacked — no spatial autoregression, so no chain error).
      * len_head   → L : per-clip-shared rigid bone lengths (anatomy prior).

    A differentiable projection then snaps P0 onto the constant-bone-length
    manifold by K unrolled steps of (projected) gradient descent on

        min_P  Σ_j w_j‖P_j − P0_j‖²  +  ρ Σ_b (‖P_c(b) − P_p(b)‖ − L_b)²

    Every joint is adjusted jointly (global, not FK-chained). The data weight
    w_j is driven by per-joint confidence: occluded joints (w→0) detach from the
    noisy regressed point and are reconstructed from bone-length constraints to
    confident neighbours — the *same* solver does anatomy enforcement AND
    occlusion completion. Returns P0 too, so the loss can keep the raw head honest.
    """

    def __init__(self, d_model, num_joints=17, num_bones=16, hidden=256,
                 n_iter=8, rho=5.0, step=0.05, w_floor=0.1):
        super().__init__()
        self.J = num_joints
        self.n_iter = n_iter
        self.rho = rho
        self.step = step
        self.w_floor = w_floor
        self.register_buffer('child_idx', torch.tensor(BONE_CHILD_IDX, dtype=torch.long))
        self.register_buffer('parent_idx', torch.tensor(BONE_PARENT_IDX, dtype=torch.long))
        self.coord_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.len_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Softplus(),
        )

    def _project(self, P0, L, w):
        """Unrolled projected-GD onto the bone-length manifold (all in fp32).
        P0:(N,J,3)  L:(N,16,1)  w:(N,J,1)  →  P:(N,J,3)."""
        ci, pi = self.child_idx, self.parent_idx
        P = P0
        for _ in range(self.n_iter):
            g = w * (P - P0)                                    # data-term grad (N,J,3)
            v = P[:, ci] - P[:, pi]                             # bone vectors (N,16,3)
            length = v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            r = length - L                                      # length residual (N,16,1)
            gb = self.rho * r * (v / length)                    # bone-term grad (N,16,3)
            g = g.index_add(1, ci, gb)
            g = g.index_add(1, pi, -gb)
            P = P - self.step * g
        return P

    def forward(self, h, conf=None):
        """h:(B,T,J,D), conf:(B,T,J,1) or None → P, bone_dir, bone_len, P0."""
        B, T, J = h.shape[:3]
        P0 = self.coord_head(h)                                 # (B,T,J,3)
        h_bones = h[:, :, self.child_idx]                       # (B,T,16,D)
        bone_len = self.len_head(h_bones).mean(dim=1, keepdim=True)   # (B,1,16,1)
        bone_len = bone_len.expand(B, T, -1, -1)                # (B,T,16,1)

        if conf is not None:
            w = self.w_floor + (1.0 - self.w_floor) * conf      # (B,T,J,1) in [floor,1]
        else:
            w = torch.ones(B, T, J, 1, device=h.device, dtype=P0.dtype)

        N = B * T
        P = self._project(P0.reshape(N, J, 3).float(),
                          bone_len.reshape(N, -1, 1).float(),
                          w.reshape(N, J, 1).float())
        P = P.reshape(B, T, J, 3).to(P0.dtype)

        bone_dir, _ = decompose_bones(P)
        return P, bone_dir, bone_len, P0


class BoneStateMamba(nn.Module):
    """
    BoneStateMamba V2

    Key upgrades over V1:
      - Velocity-augmented input: [x, y, dx, dy] per joint
      - Separate spatial attention (over 17 joints) per frame
      - Separate temporal BiGRU (over 243 frames) per joint
      - Global-average-pool root head (no heavy J*D linear)

    Input:  (B, T, 17, 2)  2D keypoints
            (B, T, 17, 1)  confidence
    Output: (B, T, 17, 3)  3D joints
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        J   = cfg.num_joints        # 17
        jd  = cfg.joint_embed_dim
        bd  = cfg.bone_embed_dim
        D   = cfg.state_dim         # = jd + bd

        # ---- Input embeddings ----
        # Joints get velocity-augmented 4-channel input [x,y,dx,dy]
        self.joint_embed    = nn.Linear(4, jd)
        self.bone_dir_embed = nn.Linear(2, bd // 2)
        self.bone_len_embed = nn.Linear(1, bd // 2)

        # ---- Positional embeddings ----
        self.temporal_pe = nn.Parameter(
            torch.randn(1, cfg.num_frames, 1, D) * 0.02)
        self.joint_pe    = nn.Parameter(
            torch.randn(1, 1, J, D) * 0.02)

        # ---- V2 temporal/body-token config ----
        scales       = tuple(getattr(cfg, 'temporal_scales', [1]))
        has_coarse   = any(s > 1 for s in scales)
        share_coarse = getattr(cfg, 'share_coarse', True)
        d_state      = getattr(cfg, 'd_state', 16)
        d_state_c    = getattr(cfg, 'd_state_coarse', 16)
        # one coarse BiSSM shared across all blocks (regularizer + budget) — stored
        # here so it lives in this module's params once; blocks hold an unregistered ref.
        self.coarse_ssm = (BiSSM(D, d_state=d_state_c, d_conv=4, expand=1,
                                 conf_gate=True, dropout=cfg.dropout)
                           if (has_coarse and share_coarse) else None)
        self.body_token = (BodyToken(D, cfg.dropout)
                           if getattr(cfg, 'body_token', False) else None)

        # ---- Backbone: N × STBlock ----
        self.blocks = nn.ModuleList([
            STBlock(
                d_model         = D,
                num_heads       = cfg.num_heads,
                expand          = cfg.ssm_expand,
                mlp_ratio       = cfg.mlp_ratio,
                dropout         = cfg.dropout,
                d_state         = d_state,
                scan_order      = getattr(cfg, 'scan_order', 'kin'),
                temporal_scales = scales,
                temporal_fuse   = getattr(cfg, 'temporal_fuse', 'gate'),
                d_state_coarse  = d_state_c,
                coarse_conf_pool= getattr(cfg, 'coarse_conf_pool', 'avg'),
                coarse_upsample = getattr(cfg, 'coarse_upsample', 'linear'),
                coarse_ssm      = self.coarse_ssm,
                body_token      = self.body_token,
                spatial_conf_gate = getattr(cfg, 'spatial_conf_gate', False),
                spatial_gcn     = getattr(cfg, 'spatial_gcn', False),
                gcn_hidden      = getattr(cfg, 'gcn_hidden', None),
            )
            for _ in range(cfg.num_blocks)
        ])

        # ---- Decoder: 'dap' (default, novelty A) or 'fk' (ablation baseline) ----
        if getattr(cfg, 'decoder', 'dap') == 'fk':
            self.decoder = FKDecoder(D, J, cfg.num_bones, cfg.fk_hidden)
        else:
            self.decoder = DAPDecoder(
                D, J, cfg.num_bones, cfg.fk_hidden,
                n_iter=getattr(cfg, 'dap_iter', 8),
                rho=getattr(cfg, 'dap_rho', 5.0),
                step=getattr(cfg, 'dap_step', 0.05),
                w_floor=getattr(cfg, 'dap_w_floor', 0.1),
            )

    # ------------------------------------------------------------------
    def encode(self, x_2d, conf=None):
        """Shared encoder (used by forward and by MPM pretraining).
        x_2d: (B,T,17,2), conf: (B,T,17,1) or None → state h: (B,T,J,D)."""
        B, T, J, _ = x_2d.shape

        # 0. Occlusion-graceful input: fill occluded joints with last valid value
        #    (conf is kept low so the SSM still coasts on them).
        if conf is not None:
            x_2d = carry_forward_fill(x_2d, conf)

        # 1. Velocity (finite difference; zero at t=0)
        vel = torch.zeros_like(x_2d)
        vel[:, 1:] = x_2d[:, 1:] - x_2d[:, :-1]
        x_vel = torch.cat([x_2d, vel], dim=-1)   # (B, T, J, 4)

        # 2. Bone decomposition (parameter-free, 2D)
        bone_dir_in, bone_len_in = decompose_bones(x_2d)

        # 3. Embed joints (velocity-aware)
        h_joint = self.joint_embed(x_vel)          # (B, T, J, jd)

        # 4. Embed bones → scatter to child joint positions
        h_bd = self.bone_dir_embed(bone_dir_in)    # (B, T, 16, bd/2)
        h_bl = self.bone_len_embed(bone_len_in)    # (B, T, 16, bd/2)
        h_bone_child = torch.cat([h_bd, h_bl], dim=-1)   # (B, T, 16, bd)

        h_bone_full = torch.zeros(B, T, J, self.cfg.bone_embed_dim,
                                  device=x_2d.device, dtype=h_joint.dtype)
        h_bone_full[:, :, BONE_CHILD_IDX] = h_bone_child

        # 5. Bone-augmented state + 6. positional embeddings
        h = torch.cat([h_joint, h_bone_full], dim=-1)   # (B, T, J, D)
        h = h + self.temporal_pe[:, :T] + self.joint_pe[:, :, :J]

        # 7. Backbone
        for block in self.blocks:
            h = block(h, conf)
        return h

    def forward(self, x_2d, conf=None):
        """x_2d:(B,T,17,2), conf:(B,T,17,1) or None → joints, bone_dir, bone_len, P0."""
        h = self.encode(x_2d, conf)
        return self.decoder(h, conf)
