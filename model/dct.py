"""DCT low-pass denoising front-end (PoseFormerV2 / SCT idea).

Clean-CPN error on H36M is dominated by 2D-detector NOISE, not the lifter. A
low-frequency DCT filter on each joint's temporal trajectory suppresses per-frame
detector jitter while preserving the global motion — a fixed linear transform with
ZERO learnable parameters. We keep the low `n_coef` DCT-II coefficients and invert:

    x_denoised = P @ x ,  P = Bᵀ_{:n} B_{:n}   (B = orthonormal DCT-II matrix)

`P` is a (T,T) buffer applied along time, so the module is parameter-free and shape-
preserving (seq2seq-friendly).
"""

import math
import torch
import torch.nn as nn


def _dct_matrix(T):
    """Orthonormal DCT-II basis, shape (T, T): row k is the k-th cosine basis."""
    n = torch.arange(T).float()
    k = torch.arange(T).float().unsqueeze(1)                    # (T,1)
    B = torch.cos(math.pi / T * (n + 0.5) * k)                  # (T,T)
    B[0] *= 1.0 / math.sqrt(2.0)
    B *= math.sqrt(2.0 / T)
    return B                                                    # orthonormal: Bᵀ B = I


class DCTDenoise(nn.Module):
    """Parameter-free low-pass temporal filter over a (B, T, J, C) tensor.

    keep_ratio in (0,1]: fraction of low-frequency coefficients retained (n_coef =
    ceil(keep_ratio * T)). keep_ratio=1.0 is the identity (no filtering)."""

    def __init__(self, num_frames, keep_ratio=0.25):
        super().__init__()
        T = num_frames
        n = max(1, int(math.ceil(keep_ratio * T)))
        B = _dct_matrix(T)                                      # (T,T)
        P = B[:n].t() @ B[:n]                                   # (T,T) low-pass projector
        self.register_buffer('P', P)
        self.n_coef = n
        self.keep_ratio = keep_ratio

    def forward(self, x):
        """x: (B, T, J, C) -> low-pass filtered (B, T, J, C) along time."""
        if self.keep_ratio >= 1.0:
            return x
        P = self.P.to(x.dtype)                                  # (T,T)
        # einsum over time: out[b,t,j,c] = sum_s P[t,s] x[b,s,j,c]
        return torch.einsum('ts,bsjc->btjc', P, x)
