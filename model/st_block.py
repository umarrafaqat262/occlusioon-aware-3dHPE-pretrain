"""Spatio-Temporal block: Spatial → Temporal (→ global body token)."""

import torch
import torch.nn as nn
from model.spatial_block import SpatialBlock
from model.temporal_block import TemporalBlock, MultiScaleTemporalBlock


class STBlock(nn.Module):
    """One round of spatial attention (over joints) then temporal SSM (over frames).

    Spatial: each frame independently, joints attend to each other.
    Temporal: each joint independently over its 243-frame trajectory — either the
    V1 single-resolution `TemporalBlock` (scales==[1]) or the V2
    `MultiScaleTemporalBlock` (global-local multi-scale).

    Optionally, a (shared) global body token injects whole-body context into every
    joint at the end of the block — targets global depth/orientation.

    - Spatial  complexity: O(J²) per frame — cheap, J=17
    - Temporal complexity: O(T)  per joint — linear in sequence length
    """

    def __init__(self, d_model, num_heads=8, expand=2,
                 mlp_ratio=2, dropout=0.1, d_state=16, scan_order='kin',
                 temporal_scales=(1,), temporal_fuse='gate', d_state_coarse=16,
                 coarse_conf_pool='avg', coarse_upsample='linear',
                 coarse_ssm=None, body_token=None,
                 spatial_conf_gate=False, spatial_gcn=False, gcn_hidden=None):
        super().__init__()
        self.spatial = SpatialBlock(d_model, num_heads, mlp_ratio, dropout,
                                    d_state=d_state, scan_order=scan_order,
                                    conf_gate=spatial_conf_gate,
                                    gcn=spatial_gcn, gcn_hidden=gcn_hidden)

        scales = tuple(temporal_scales)
        if scales == (1,):
            # V1 path — preserves param names so a V1 checkpoint loads (sanity A/B).
            self.temporal = TemporalBlock(d_model, expand, dropout, d_state=d_state)
        else:
            self.temporal = MultiScaleTemporalBlock(
                d_model, expand=expand, dropout=dropout, d_state=d_state,
                scales=scales, d_state_coarse=d_state_coarse, fuse=temporal_fuse,
                conf_pool=coarse_conf_pool, upsample=coarse_upsample,
                coarse_ssm=coarse_ssm)

        # shared body token stored unregistered (owned by BoneStateMamba) to avoid
        # duplicate state_dict keys; None disables it.
        self._body = (body_token,) if body_token is not None else None

    def forward(self, x, conf=None):
        """
        x:    (B, T, J, D)
        conf: (B, T, J, 1) or None
        Returns: (B, T, J, D)
        """
        B, T, J, D = x.shape

        # --- Spatial: (B*T, J, D) ---
        x_s = x.reshape(B * T, J, D)
        conf_s = conf.reshape(B * T, J, 1) if conf is not None else None
        x_s = self.spatial(x_s, conf_s)
        x = x_s.reshape(B, T, J, D)

        # --- Temporal: (B*J, T, D) ---
        x_t = x.permute(0, 2, 1, 3).reshape(B * J, T, D)
        conf_t = None
        if conf is not None:
            conf_t = conf.permute(0, 2, 1, 3).reshape(B * J, T, 1)
        x_t = self.temporal(x_t, conf_t)
        x = x_t.reshape(B, J, T, D).permute(0, 2, 1, 3).contiguous()

        # --- Global body token (optional) ---
        if self._body is not None:
            x = self._body[0](x)

        return x
