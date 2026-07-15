"""V2 sanity: param budget, forward correctness, faithful-superset checkpoint load."""
import torch
from common.utils import load_config, count_parameters
from model.bsmamba import BoneStateMamba

device = 'cuda'


def build(cfg_path, **overrides):
    cfg = load_config(cfg_path)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return BoneStateMamba(cfg).to(device).eval(), cfg


def fwd(model, B=2, T=243, conf=True):
    x = torch.randn(B, T, 17, 2, device=device)
    c = torch.rand(B, T, 17, 1, device=device) if conf else None
    with torch.no_grad():
        pred, bdir, blen, p0 = model(x, c)
    return pred


print("=" * 60)
# 1) Faithful-superset: V1 config (no temporal_scales / no body_token) ----------
m1, _ = build('configs/anatproj_sota.yaml')
n1 = count_parameters(m1)
print(f"[1] V1-path params : {n1:,}   (expect 968,092)")
ckpt = torch.load('checkpoints/best_anatproj_sota.pth', map_location=device,
                  weights_only=False)
state = ckpt['ema'] if 'ema' in ckpt else ckpt['model']
missing, unexpected = m1.load_state_dict(state, strict=False)
print(f"    load V1 ckpt -> missing={len(missing)} unexpected={len(unexpected)}")
p = fwd(m1); print(f"    forward ok, out={tuple(p.shape)}, finite={torch.isfinite(p).all().item()}")

print("-" * 60)
# 2) V2 full config -------------------------------------------------------------
m2, cfg2 = build('configs/anatproj_v2.yaml')
n2 = count_parameters(m2)
print(f"[2] V2 params      : {n2:,}   (+{n2-n1:,} vs V1)  scales={cfg2.temporal_scales} "
      f"body_token={getattr(cfg2,'body_token',False)} share_coarse={getattr(cfg2,'share_coarse',True)}")
p = fwd(m2, conf=True);  print(f"    forward (conf) ok, out={tuple(p.shape)}, finite={torch.isfinite(p).all().item()}")
p = fwd(m2, conf=False); print(f"    forward (no conf) ok, out={tuple(p.shape)}, finite={torch.isfinite(p).all().item()}")

print("-" * 60)
# 3) state_dict has no duplicate shared-module keys -----------------------------
keys = list(m2.state_dict().keys())
coarse_keys = [k for k in keys if 'coarse_ssm' in k]
body_keys   = [k for k in keys if 'body_token' in k]
print(f"[3] coarse_ssm keys: {len(coarse_keys)} (all under one path: "
      f"{len(set(k.split('coarse_ssm')[0] for k in coarse_keys))==1})")
print(f"    body_token keys: {len(body_keys)}")

print("-" * 60)
# 4) Rotation-aug consistency (camera-roll preserves root-relative geometry) -----
from common.augmentation import random_rotation
import random as _r; _r.seed(0); torch.manual_seed(0)
x2 = torch.randn(4, 10, 17, 2); x3 = torch.randn(4, 10, 17, 3)
r2, r3 = random_rotation(x2.clone(), x3.clone(), max_deg=20)
# depth (Z) must be untouched; XY norm per joint preserved by rotation
z_ok = torch.allclose(r3[..., 2], x3[..., 2])
xy_norm_ok = torch.allclose(r3[..., :2].norm(dim=-1), x3[..., :2].norm(dim=-1), atol=1e-5)
d2_norm_ok = torch.allclose(r2.norm(dim=-1), x2.norm(dim=-1), atol=1e-5)
print(f"[4] rotation: Z-unchanged={z_ok}  3D-XY-norm-preserved={xy_norm_ok}  2D-norm-preserved={d2_norm_ok}")
print("=" * 60)
print("SANITY DONE")
