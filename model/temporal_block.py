"""Temporal block(s): bidirectional selective SSM (Mamba) over T frames per joint.

Each joint is processed independently over time — O(J) parallel streams of
length T. The sequence model is a real selective SSM (mamba-ssm) whose Δ is
modulated by per-frame 2D confidence, so occluded frames coast on temporal
memory (novelty B).

`TemporalBlock` is the original single-resolution block (used when
`temporal_scales == [1]`, and it preserves the V1 parameter names so an old
checkpoint loads). `MultiScaleTemporalBlock` (V2) adds coarse global branches:
the same sequence is confidence-pooled to lower temporal resolutions, scanned by
a (optionally shared) BiSSM, upsampled back, and fused with the full-res local
scan. The local branch captures fine per-frame dynamics; the coarse branches
integrate slow global depth/orientation trends over the whole window with far
fewer recurrence steps — the measured weakness (large P1−P2 gap, static actions).

Input shape: (B*J, T, D).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ssm import BiSSM


class TemporalBlock(nn.Module):
    """V1 single-resolution temporal block (exact param-name compatibility)."""
    def __init__(self, d_model, expand=2, dropout=0.1, d_state=16, d_conv=4,
                 motion_adaptive=False):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = BiSSM(d_model, d_state=d_state, d_conv=d_conv,
                         expand=expand, conf_gate=True, dropout=dropout,
                         motion_adaptive=motion_adaptive)

    def forward(self, x, conf=None):
        """x: (B*J, T, D), conf: (B*J, T, 1) or None → (B*J, T, D)."""
        return x + self.ssm(self.norm(x), conf)


def _pool_time(x, conf, s, conf_pool='avg'):
    """Confidence-weighted average-pool over time by factor s (left-padded so no
    tail is dropped). x:(N,T,D), conf:(N,T,1) or None.
    Returns (x_pooled:(N,T//s_ceil,D), conf_pooled, T_padded, T_orig)."""
    N, T, D = x.shape
    pad = (s - T % s) % s
    if pad:
        x = torch.cat([x[:, :1].expand(N, pad, D), x], dim=1)
        if conf is not None:
            conf = torch.cat([conf[:, :1].expand(N, pad, 1), conf], dim=1)
    Tp = x.shape[1]
    xw = x.reshape(N, Tp // s, s, D)
    if conf is not None:
        cw = conf.reshape(N, Tp // s, s, 1)
        xp = (xw * cw).sum(2) / (cw.sum(2) + 1e-6)             # conf-weighted mean
        confp = cw.min(2).values if conf_pool == 'min' else cw.mean(2)
    else:
        xp = xw.mean(2)
        confp = None
    return xp, confp, Tp, T


def _upsample_time(y, Tp, T, mode='linear'):
    """Upsample (N,Tc,D) back to length T (the last T of the Tp-length grid,
    matching the left-pad in `_pool_time`)."""
    yt = y.transpose(1, 2)                                     # (N,D,Tc)
    if mode == 'nearest':
        up = F.interpolate(yt, size=Tp, mode='nearest')
    else:
        up = F.interpolate(yt, size=Tp, mode='linear', align_corners=False)
    up = up.transpose(1, 2)                                    # (N,Tp,D)
    return up[:, Tp - T:, :]


class MultiScaleTemporalBlock(nn.Module):
    """Multi-scale global-local temporal block (V2)."""
    def __init__(self, d_model, expand=1, dropout=0.1, d_state=24, d_conv=4,
                 scales=(1, 3), d_state_coarse=16, fuse='gate',
                 conf_pool='avg', upsample='linear', coarse_ssm=None):
        super().__init__()
        self.scales = tuple(scales)
        self.coarse_scales = [s for s in self.scales if s > 1]
        self.conf_pool = conf_pool
        self.upsample = upsample
        self.fuse = fuse

        self.norm = nn.LayerNorm(d_model)
        # full-resolution local branch
        self.local_ssm = BiSSM(d_model, d_state=d_state, d_conv=d_conv,
                               expand=expand, conf_gate=True, dropout=dropout)

        # coarse branch: shared module (passed in, stored unregistered to avoid
        # duplicate state_dict keys) or this block's own.
        if self.coarse_scales:
            if coarse_ssm is not None:
                self._shared_coarse = (coarse_ssm,)            # tuple → not a submodule
            else:
                self._shared_coarse = None
                self.coarse_ssm = BiSSM(d_model, d_state=d_state_coarse,
                                        d_conv=d_conv, expand=1,
                                        conf_gate=True, dropout=dropout)

        K = len(self.scales)
        if K > 1:
            if fuse == 'gate':
                self.fuse_gate = nn.Linear(d_model, K)
            self.fuse_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _coarse(self):
        return self._shared_coarse[0] if self._shared_coarse is not None else self.coarse_ssm

    def forward(self, x, conf=None):
        """x: (B*J, T, D), conf: (B*J, T, 1) or None → (B*J, T, D)."""
        T = x.shape[1]
        h = self.norm(x)
        branches = [self.local_ssm(h, conf)]
        for s in self.coarse_scales:
            xp, cp, Tp, T_orig = _pool_time(h, conf, s, self.conf_pool)
            yc = self._coarse()(xp, cp)
            branches.append(_upsample_time(yc, Tp, T_orig, self.upsample))

        if len(branches) == 1:
            out = branches[0]
        elif self.fuse == 'gate':
            g = torch.softmax(self.fuse_gate(h), dim=-1)       # (N,T,K)
            out = sum(g[..., k:k + 1] * branches[k] for k in range(len(branches)))
            out = self.fuse_proj(out)
        else:                                                  # 'sum'
            out = self.fuse_proj(sum(branches))
        return x + self.drop(out)
