"""Spatial block over the 17 joints (per frame).

The joint mixer is a bidirectional selective SSM that scans joints in kinematic-tree
order. On top of that, several parameter-cheap, evidence-backed spatial modules can
be switched on (all default OFF → the block reduces to the original SSM + parent
prior, so old configs reproduce exactly):

  * conf_gate      — gated (slow-path) spatial SSM so occluded joints coast spatially.
  * kpa            — KTPFormer KPA ModulatedGCN prior on the joint tokens (arXiv 2404.00658).
  * lap_pe (k>0)   — parameter-free Laplacian-eigenvector positional encoding (arXiv 2405.17397).
  * limb_reorder   — PoseMamba additive global + limb-chain-reordered view before the scan.
  * ssi            — SAMA-style Structure-aware State Integrator: fuse per-joint SSM
                     states across the skeleton, weighted by per-joint CONFIDENCE so an
                     occluded joint's state is inferred from confident neighbours (novel).
  * gcn            — a bottleneck skeleton GCN branch (kept for ablation).
"""

import torch
import torch.nn as nn

from model.ssm import BiSSM
from model.graph_priors import KPAGraphConv, laplacian_pe, _row_normalized_adjacency
from common.skeleton import H36M_PARENTS, KIN_SCAN_ORDER, KIN_SCAN_INV, LIMB_REORDER_INDEX


def _skeleton_adjacency(parents):
    """Symmetric, self-looped, symmetrically-normalized adjacency  Â = D^-½(A+I)D^-½."""
    J = len(parents)
    A = torch.eye(J)
    for j, p in enumerate(parents):
        if p >= 0:
            A[j, p] = 1.0
            A[p, j] = 1.0
    deg = A.sum(1)
    dinv = torch.diag(deg.clamp(min=1.0).pow(-0.5))
    return dinv @ A @ dinv


class SpatialBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, mlp_ratio=2, dropout=0.1,
                 d_state=16, d_conv=4, scan_order='kin', conf_gate=False,
                 gcn=False, gcn_hidden=None, kpa=False, lap_pe=0,
                 limb_reorder=False, ssi=False):
        super().__init__()
        if scan_order == 'shuffle':
            import numpy as _np
            order = _np.random.RandomState(0).permutation(len(KIN_SCAN_ORDER)).tolist()
        else:
            order = list(KIN_SCAN_ORDER)
        inv = [order.index(j) for j in range(len(order))]
        J = len(order)
        self.register_buffer('order', torch.tensor(order, dtype=torch.long))
        self.register_buffer('inv', torch.tensor(inv, dtype=torch.long))
        parents = [p if p >= 0 else j for j, p in enumerate(H36M_PARENTS)]
        self.register_buffer('parent_idx', torch.tensor(parents, dtype=torch.long))

        self.norm1 = nn.LayerNorm(d_model)
        self.conf_gate = conf_gate
        self.ssm = BiSSM(d_model, d_state=d_state, d_conv=min(d_conv, 4),
                         expand=1, fast=not conf_gate, conf_gate=conf_gate,
                         dropout=dropout)
        self.parent_proj = nn.Linear(d_model, d_model)

        # KPA graph prior on the joint tokens (residual)
        self.kpa = KPAGraphConv(d_model, J) if kpa else None

        # Laplacian-eigenvector PE (parameter-free buffer + a tiny projection)
        self.lap_pe = lap_pe
        if lap_pe:
            self.register_buffer('lap_vec', laplacian_pe(J, lap_pe))     # (J, k)
            self.lap_proj = nn.Linear(lap_pe, d_model)

        # limb-chain additive reorder
        self.limb_reorder = limb_reorder
        if limb_reorder:
            self.register_buffer('limb_idx', torch.tensor(LIMB_REORDER_INDEX, dtype=torch.long))

        # SAMA-style confidence-weighted state integrator (init to skeleton adjacency,
        # gate init 0 → starts as a no-op)
        self.ssi = ssi
        if ssi:
            self.ssi_adj = nn.Parameter(_row_normalized_adjacency(J))
            self.ssi_gate = nn.Parameter(torch.zeros(1))

        # optional bottleneck GCN branch (ablation)
        self.gcn = gcn
        if gcn:
            gh = gcn_hidden or max(8, d_model // 3)
            self.register_buffer('A_norm', _skeleton_adjacency(H36M_PARENTS))
            self.gcn1 = nn.Linear(d_model, gh)
            self.gcn_act = nn.GELU()
            self.gcn2 = nn.Linear(gh, d_model)
            self.gcn_gate = nn.Parameter(torch.zeros(1))

        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, conf=None):
        """x: (N, J, D) with N = B*T; conf: (N, J, 1) or None → (N, J, D)."""
        h = self.norm1(x)
        if self.lap_pe:
            h = h + self.lap_proj(self.lap_vec.to(h.dtype))          # (1,J,D) broadcast
        if self.kpa is not None:
            h = h + self.kpa(h)                                      # residual graph prior

        parent_feat = self.parent_proj(h[:, self.parent_idx])

        # limb-chain additive view (global + local) before the scan
        h_scan = h + h[:, self.limb_idx] if self.limb_reorder else h
        hs = h_scan[:, self.order]
        cs = conf[:, self.order] if conf is not None else None
        hs = self.ssm(hs, cs)
        hs = hs[:, self.inv]

        # confidence-weighted state fusion across joints (SSI, novel)
        if self.ssi:
            cw = conf if conf is not None else hs.new_ones(hs.shape[0], hs.shape[1], 1)
            msg = torch.einsum('ak,nkd->nad', self.ssi_adj.to(hs.dtype), cw.to(hs.dtype) * hs)
            hs = hs + torch.tanh(self.ssi_gate) * msg

        x = x + hs + parent_feat
        if self.gcn:
            g = torch.einsum('ij,njd->nid', self.A_norm.to(h.dtype), h)
            g = self.gcn_act(self.gcn1(g))
            g = torch.einsum('ij,njd->nid', self.A_norm.to(h.dtype), g)
            g = self.gcn2(g)
            x = x + torch.tanh(self.gcn_gate) * g
        x = x + self.mlp(self.norm2(x))
        return x
