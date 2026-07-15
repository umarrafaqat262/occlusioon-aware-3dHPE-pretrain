"""Bone-state bidirectional SSM block.

Uses a pure-PyTorch bidirectional GRU as the sequence model (no compiled CUDA
extensions required). The interface is identical to what BoneStateMamba expects.
"""

import torch
import torch.nn as nn


class ConfidenceGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, h, conf):
        """h: (B, L, D), conf: (B, L, 1) → gate: (B, L, 1)"""
        return self.net(torch.cat([h, conf], dim=-1))


class BoneStateMambaBlock(nn.Module):
    """Bidirectional GRU block with bone-augmented state and confidence gate.

    Drop-in replacement for a bidirectional Mamba block. Uses GRU so no
    compiled extensions are needed. Architecture is otherwise identical.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        hidden = d_model * expand
        # Bidirectional GRU — forward and backward in one call
        self.gru = nn.GRU(d_model, hidden, batch_first=True, bidirectional=True)
        self.merge = nn.Linear(hidden * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.conf_gate = ConfidenceGate(d_model)

    def forward(self, x, conf=None):
        """x: (B, L, D), conf: (B, L, 1) or None"""
        residual = x
        h, _ = self.gru(x)          # (B, L, hidden*2)
        h = self.merge(h)            # (B, L, D)

        if conf is not None:
            gate = self.conf_gate(h, conf)
            h = gate * h + (1 - gate) * residual
        else:
            h = h + residual

        return self.drop(self.norm(h))
