"""Phase-0 verification for the BiSSM seam (run from repo root)."""
import torch
from model.ssm import BiSSM, ConfMamba

dev = 'cuda'
torch.manual_seed(0)

# 1. Standard fwd/bwd
m = BiSSM(d_model=64, d_state=16, conf_gate=True).to(dev)
x = torch.randn(4, 243, 64, device=dev, requires_grad=True)
conf = torch.rand(4, 243, 1, device=dev)
y = m(x, conf)
y.sum().backward()
print("BiSSM fwd/bwd OK:", tuple(y.shape), "grad_ok", x.grad is not None)
print("conf=None path OK:", tuple(m(x, None).shape))

# 2. Coasting test: occluded region should be far less sensitive to input noise.
torch.manual_seed(1)
cm = ConfMamba(64, conf_gate=True).to(dev).eval()
L = 243
base = torch.randn(1, L, 64, device=dev)
noise = torch.zeros(1, L, 64, device=dev)
occ = slice(100, 140)               # an "occluded" window
noise[:, occ] = torch.randn(1, 40, 64, device=dev) * 3.0   # big noise in window

with torch.no_grad():
    # conf high everywhere
    c_hi = torch.ones(1, L, 1, device=dev)
    d_hi = (cm(base + noise, c_hi) - cm(base, c_hi))[:, occ].abs().mean().item()
    # conf zero in the occluded window (should coast → ignore the noise)
    c_lo = torch.ones(1, L, 1, device=dev); c_lo[:, occ] = 0.0
    d_lo = (cm(base + noise, c_lo) - cm(base, c_lo))[:, occ].abs().mean().item()

print(f"sensitivity to in-window noise: conf=1 -> {d_hi:.4f} | conf=0 -> {d_lo:.4f}")
print("COAST_OK" if d_lo < d_hi else "COAST_FAIL")
print("ALL_GOOD")
