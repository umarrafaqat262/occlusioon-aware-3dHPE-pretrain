"""Spatial block over the 17 joints (per frame) — novelty (A).

The joint mixer is a bidirectional selective SSM that scans joints in
kinematic-tree order (root→leaf forward, leaf→root backward). Combined with the
FK decoder, the spatial recurrence propagates state along the skeleton so it
behaves as a learned forward-kinematics integrator — unlike prior SSM lifters
that use stride / global-local / vanilla scans over a flattened joint axis.

A lightweight parent-gather term injects each joint's parent feature as an
explicit anatomical prior (replaces the previous learnable-adjacency attention
bias).

Spatial confidence gating: if `conf_gate=True`, the joint SSM uses the gated
(slow-path) ConfMamba so an occluded joint's carried/filled value is down-weighted
in the spatial scan (Δ→0) instead of leaking into its bone-chain neighbours. When
`conf_gate=False` (default, checkpoint-compatible) the fused kernel is used and the
`conf` argument is ignored — spatial coasting is then disabled and only the
temporal axis coasts.
"""

import torch
import torch.nn as nn

from model.ssm import BiSSM
from common.skeleton import H36M_PARENTS, KIN_SCAN_ORDER, KIN_SCAN_INV


def _skeleton_adjacency(parents):
    """Symmetric, self-looped, symmetrically-normalized adjacency  Â = D^-½(A+I)D^-½
    over the H36M joints (child<->parent edges). Returns (J,J) float tensor."""
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
                 gcn=False, gcn_hidden=None):
        super().__init__()
        # scan permutation. 'kin' = kinematic-tree (root→leaf) order (novelty A);
        # 'shuffle' = a fixed random order (ablation to prove kin-order matters).
        if scan_order == 'shuffle':
            import numpy as _np
            order = _np.random.RandomState(0).permutation(len(KIN_SCAN_ORDER)).tolist()
        else:
            order = list(KIN_SCAN_ORDER)
        inv = [order.index(j) for j in range(len(order))]
        self.register_buffer('order', torch.tensor(order, dtype=torch.long))
        self.register_buffer('inv', torch.tensor(inv, dtype=torch.long))
        parents = [p if p >= 0 else j for j, p in enumerate(H36M_PARENTS)]
        self.register_buffer('parent_idx', torch.tensor(parents, dtype=torch.long))

        self.norm1 = nn.LayerNorm(d_model)
        # spatial sequence length is J (=17). fast=True uses the fused kernel (no
        # gating); conf_gate=True uses the slow-path ConfMamba so occluded joints
        # are down-weighted spatially too (helps occlusion, esp. distal joints).
        self.conf_gate = conf_gate
        self.ssm = BiSSM(d_model, d_state=d_state, d_conv=min(d_conv, 4),
                         expand=1, fast=not conf_gate, conf_gate=conf_gate,
                         dropout=dropout)
        # parent-feature injection (anatomical prior)
        self.parent_proj = nn.Linear(d_model, d_model)

        # Local-joint GCN branch (Pose Magic / HGMamba / MDTF recipe): Mamba
        # under-attends local neighbouring-joint structure, so a graph-conv branch
        # over the skeleton adjacency is fused in. Gated by a scalar init to 0 so
        # it starts as a no-op (safe warm-start) and the model learns how much to
        # use it. Bottleneck hidden keeps the whole model < 1M params.
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
        # parent-gather anatomical prior (in canonical joint order)
        parent_feat = self.parent_proj(h[:, self.parent_idx])
        # kinematic-tree ordered bidirectional SSM scan
        hs = h[:, self.order]
        cs = conf[:, self.order] if conf is not None else None
        hs = self.ssm(hs, cs)
        hs = hs[:, self.inv]
        x = x + hs + parent_feat
        if self.gcn:
            # two-layer graph conv over skeleton adjacency (in canonical order)
            g = torch.einsum('ij,njd->nid', self.A_norm.to(h.dtype), h)
            g = self.gcn_act(self.gcn1(g))
            g = torch.einsum('ij,njd->nid', self.A_norm.to(h.dtype), g)
            g = self.gcn2(g)
            x = x + torch.tanh(self.gcn_gate) * g
        x = x + self.mlp(self.norm2(x))
        return x
