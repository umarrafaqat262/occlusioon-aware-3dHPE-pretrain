"""Selective-SSM seam for KinFK-Mamba.

Two pieces:

  ConfMamba  — a Mamba-1 block (state-spaces/mamba) whose selective time-step Δ
               is modulated by per-token 2D-keypoint confidence.  When a joint /
               frame is occluded (conf → 0) the gate drives Δ → 0, so the
               discretised transition  Ā = exp(Δ·A) → I  and the state *coasts*
               on its own memory instead of ingesting the noisy observation.
               This is novelty (B): occlusion robustness as learned selectivity,
               not preprocessing.  conf=None ⇒ identical to a vanilla Mamba block.

  BiSSM      — bidirectional wrapper (forward + reversed scan, merged) exposing a
               single interface the temporal/spatial blocks depend on.

Both rely on the official CUDA kernels (`selective_scan_fn`, `causal_conv1d_fn`).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:  # pragma: no cover
    causal_conv1d_fn = None


class ConfMamba(Mamba):
    """Mamba-1 block with optional confidence-gated selective scan.

    Inherits all parameters/initialisation from the stock Mamba block and
    overrides only the forward (non-fused slow path, which exposes Δ).
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, conf_gate=True, **kw):
        # force slow path — the fused kernel hides Δ and cannot be gated
        super().__init__(d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                         use_fast_path=False, **kw)
        self.conf_gate = conf_gate
        if conf_gate:
            # gate = sigmoid(alpha * conf + beta), alpha>0 → high conf keeps Δ,
            # low conf shrinks Δ toward 0 (coast). Init: gate(conf=1)≈0.98 (≈vanilla
            # Mamba when fully visible), gate(conf=0)≈0.02 (strong coasting). Learnable
            # so the model tunes how aggressively to trust the detector.
            self.conf_alpha = nn.Parameter(torch.tensor(8.0))
            self.conf_beta = nn.Parameter(torch.tensor(-4.0))

    def _gate(self, conf, seqlen):
        """conf: (B, L, 1) in [0,1] → gate (B, 1, L) in (0,1)."""
        g = torch.sigmoid(self.conf_alpha * conf + self.conf_beta)   # (B, L, 1)
        return rearrange(g, "b l 1 -> b 1 l")

    def forward(self, hidden_states, conf=None):
        """hidden_states: (B, L, D); conf: (B, L, 1) or None → (B, L, D)."""
        batch, seqlen, dim = hidden_states.shape

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())                  # (d_inner, d_state)
        x, z = xz.chunk(2, dim=1)

        # short causal conv + SiLU
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()                    # (d_inner, b*l)
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) n -> b n l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) n -> b n l", l=seqlen).contiguous()

        if conf is not None and self.conf_gate:
            # Δ = softplus(dt + bias) computed here, then gated by confidence so
            # the kernel receives the final Δ directly (delta_softplus=False).
            # The selective params (dt, B, C) are still derived from the observed
            # (conv) signal, but the *integrated* input u and the D-skip are gated
            # too — so when occluded (gate→0): Δ→0 (Ā=exp(ΔA)→I, state coasts) and
            # u→0 (no noisy observation enters the state or the skip). The block
            # then predicts from carried temporal memory.  Novelty (B).
            g = self._gate(conf, seqlen).to(dt.dtype)         # (B, 1, L)
            delta = (F.softplus(dt.float() + self.dt_proj.bias.float()[None, :, None])
                     * g.float()).to(x.dtype)
            u = (x * g).to(x.dtype)
            y = selective_scan_fn(
                u, delta, A, B, C, self.D.float(), z=z,
                delta_bias=None, delta_softplus=False, return_last_state=False,
            )
        else:
            y = selective_scan_fn(
                x, dt, A, B, C, self.D.float(), z=z,
                delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
                return_last_state=False,
            )

        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)


class BiSSM(nn.Module):
    """Bidirectional selective SSM over a (B, L, D) sequence.

    Forward + reversed scans, concatenated and projected back to d_model.
    `conf` (B, L, 1) is optional; when given it drives the confidence-gated Δ
    in both directions.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 conf_gate=True, dropout=0.0, fast=False):
        super().__init__()
        self.conf_gate = conf_gate and not fast
        if fast:
            # stock fused Mamba kernel (no confidence gating) — much faster; used
            # for the spatial kinematic scan where coasting is not the mechanism.
            self.fwd = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.bwd = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.fwd = ConfMamba(d_model, d_state, d_conv, expand, conf_gate=conf_gate)
            self.bwd = ConfMamba(d_model, d_state, d_conv, expand, conf_gate=conf_gate)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, conf=None):
        """x: (B, L, D); conf: (B, L, 1) or None → (B, L, D)."""
        if self.conf_gate:
            yf = self.fwd(x, conf)
            cb = conf.flip(1) if conf is not None else None
            yb = self.bwd(x.flip(1), cb).flip(1)
        else:
            yf = self.fwd(x)
            yb = self.bwd(x.flip(1)).flip(1)
        return self.drop(self.merge(torch.cat([yf, yb], dim=-1)))
