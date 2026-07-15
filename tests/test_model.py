"""End-to-end KinFK-Mamba forward test (run from repo root)."""
import torch, sys
from common.utils import load_config, count_parameters
from model.bsmamba import BoneStateMamba

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else 'configs/tiny.yaml')
dev = 'cuda'
model = BoneStateMamba(cfg).to(dev)
n = count_parameters(model)
print(f"params: {n:,} ({n/1e6:.3f}M)")

B, T, J = 2, cfg.num_frames, cfg.num_joints
x = torch.randn(B, T, J, 2, device=dev)
conf = torch.rand(B, T, J, 1, device=dev)

pred, bdir, blen = model(x, conf)
print("pred", tuple(pred.shape), "bone_dir", tuple(bdir.shape), "bone_len", tuple(blen.shape))
# bone_len should be constant over time (per-clip-shared)
blen_var = blen.var(dim=1).mean().item()
print(f"bone_len temporal variance (should be ~0): {blen_var:.2e}")

# backward
loss = pred.float().pow(2).mean()
loss.backward()
gok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
print("backward grads ok:", gok)

# occlusion path: zero out some joints' conf + pose
x2 = x.clone(); c2 = conf.clone()
c2[:, :, [3, 6]] = 0.0; x2[:, :, [3, 6]] = 0.0   # occlude both ankles
pred2, _, _ = model(x2, c2)
print("occluded forward ok:", tuple(pred2.shape))
print("ALL_GOOD")
