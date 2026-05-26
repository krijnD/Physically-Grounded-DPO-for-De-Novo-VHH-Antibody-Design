import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "diffab"))

import torch
from diffab.models import get_model
from diffab.utils.misc import load_config, seed_all

config, _ = load_config('configs/dpo/vhh_dpo.yml')
seed_all(42)

ckpt_path = 'runs/vhh_ft/seed42_jfix/checkpoints/best_ema.pt'
device = 'cuda'

# Mimic the trainer's init order
model_theta = get_model(config.model).to(device)
model_ref   = get_model(config.model).to(device)

ck = torch.load(ckpt_path, map_location=device, weights_only=False)
sd = ck['model']
model_ref.load_state_dict(sd, strict=False)
model_theta.load_state_dict(sd, strict=False)

# Compare every parameter and buffer
diffs_p = []
for (n_a, p_a), (_, p_b) in zip(model_theta.named_parameters(), model_ref.named_parameters()):
    if not torch.equal(p_a.data, p_b.data):
        diffs_p.append((n_a, (p_a - p_b).abs().max().item()))

diffs_b = []
for (n_a, b_a), (_, b_b) in zip(model_theta.named_buffers(), model_ref.named_buffers()):
    if b_a.shape != b_b.shape or not torch.equal(b_a, b_b):
        diffs_b.append((n_a, b_a.shape, b_b.shape))

print(f'Parameter mismatches: {len(diffs_p)}')
for n, d in diffs_p[:10]:
    print(f'  {n}  max abs diff = {d:.6e}')
print(f'Buffer mismatches:    {len(diffs_b)}')
for n, sa, sb in diffs_b[:10]:
    print(f'  {n}  shapes: {sa} vs {sb}')