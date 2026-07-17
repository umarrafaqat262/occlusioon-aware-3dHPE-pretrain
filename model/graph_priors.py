"""Parameter-cheap skeletal priors for the spatial joint tokens.

Two components, both grounded in the literature:

  * laplacian_pe(k)     — parameter-FREE positional encoding: the k smallest
    non-trivial eigenvectors of the normalized skeleton-graph Laplacian, added to
    the joint tokens so the SSM knows where each joint sits on the skeleton
    (PerturbPE / Laplacian-PE, arXiv 2405.17397; ~-0.7mm GT-2D at zero params).

  * KPAGraphConv        — KTPFormer's Kinematics-Prior "attention" is really a
    decoupled ModulatedGCN applied to the tokens BEFORE the SSM input projection
    (arXiv 2404.00658; ~-1.8mm isolated for ~1.6K params). Decoupled self/neighbor
    weights + a learnable global adjacency Â (discovers non-anatomical links) + a
    per-joint learnable modulation M.
"""

import torch
import torch.nn as nn

from common.skeleton import H36M_PARENTS, SKELETON_EDGES


def _adjacency(num_joints=17, self_loops=True):
    """Undirected binary adjacency (J,J) from the skeleton edges."""
    A = torch.zeros(num_joints, num_joints)
    for c, p in SKELETON_EDGES:
        A[c, p] = 1.0
        A[p, c] = 1.0
    if self_loops:
        A += torch.eye(num_joints)
    return A


def _row_normalized_adjacency(num_joints=17):
    """D^{-1} A  (row-normalized, KTPFormer convention)."""
    A = _adjacency(num_joints, self_loops=True)
    deg = A.sum(1, keepdim=True).clamp(min=1.0)
    return A / deg


def laplacian_pe(num_joints=17, k=8):
    """k smallest non-trivial eigenvectors of the normalized Laplacian → (J,k).

    Parameter-free; returned as a plain tensor to register as a buffer."""
    A = _adjacency(num_joints, self_loops=False)
    deg = A.sum(1)
    dinv = torch.diag(deg.clamp(min=1.0).pow(-0.5))
    L = torch.eye(num_joints) - dinv @ A @ dinv                 # normalized Laplacian
    evals, evecs = torch.linalg.eigh(L)                         # ascending eigenvalues
    # skip the trivial (near-zero) first eigenvector; take next k
    k = min(k, num_joints - 1)
    return evecs[:, 1:1 + k].contiguous()                       # (J, k)


class KPAGraphConv(nn.Module):
    """Decoupled ModulatedGCN (KTPFormer KPA). Input/return: (N, J, D)."""

    def __init__(self, d_model, num_joints=17):
        super().__init__()
        self.register_buffer('adj', _row_normalized_adjacency(num_joints))   # (J,J) fixed
        self.adj2 = nn.Parameter(torch.full((num_joints, num_joints), 1e-6)) # learnable global Â
        self.W_self = nn.Linear(d_model, d_model, bias=False)
        self.W_neigh = nn.Linear(d_model, d_model, bias=False)
        self.M = nn.Parameter(torch.ones(num_joints, d_model))               # per-joint modulation
        # LayerNorm (not BatchNorm) — running-stat BN is unstable in a residual
        # Mamba stack under bf16; LN over the feature dim is the safe choice here.
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def forward(self, x):
        """x: (N, J, D) -> (N, J, D)."""
        J = x.shape[1]
        adj = self.adj.to(x.dtype) + self.adj2.to(x.dtype)
        adj = (adj.transpose(0, 1) + adj) * 0.5                 # symmetrize
        eye = torch.eye(J, device=x.device, dtype=x.dtype)
        h0 = self.W_self(x)                                     # (N,J,D)
        h1 = self.W_neigh(x)
        M = self.M.to(x.dtype)
        out = torch.einsum('ij,njd->nid', adj * eye, M * h0) \
            + torch.einsum('ij,njd->nid', adj * (1.0 - eye), M * h1)
        out = self.norm(out)                                   # LayerNorm over feature dim
        return self.act(out)
