"""Day99: does the pretrained ResNet18 backbone show the exact collapse Day93
found and fixed in the from-scratch encoder -- every patch within one image
mapping to (nearly) the same vector regardless of position?

Day93's two collapses (see ijepa_model.py's docstrings):
1. across-image: different images -> identical direction (cosine sim 1.0)
2. within-image: different patches of the SAME image -> identical direction
   (cosine sim 1.0) -- the more specific, more damaging one, since it's
   exactly the axis I-JEPA's masked-patch-prediction task depends on.

Both were only found by explicitly checking cosine similarity, not by
looking at raw output variance (which can look "healthy" even when
directions have collapsed -- see ijepa_model.py's variance_loss docstring).
Applying the same check here, using ResNet18's layer4 feature map (7x7x512
for a 224x224 input) as patch tokens instead of the final pooled 512-dim
vector used in probe_pretrained_resnet18.py, since only the pre-pool map
has any spatial/patch structure to test at all.
"""

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

AC_DATA_DIR = "../action-conditioned-video-prediction/data"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

with open(f"{AC_DATA_DIR}/episode_lengths.json") as f:
    episode_lengths = json.load(f)
episode_ids = sorted(episode_lengths.keys())


def load_all_frames(ep_ids):
    frames = [np.load(f"{AC_DATA_DIR}/episodes/{ep}_frames.npy") for ep in sorted(ep_ids)]
    return np.concatenate(frames)


frames = load_all_frames(episode_ids[:5])

resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
# feature map before avgpool/fc: (B, 512, 7, 7) for a 224x224 input
feature_extractor = nn.Sequential(*list(resnet.children())[:-2]).to(DEVICE).eval()
for p in feature_extractor.parameters():
    p.requires_grad = False


def to_tensor(frames, idx):
    f = torch.from_numpy(frames[idx]).float().permute(0, 3, 1, 2) / 255.0
    f = F.interpolate(f, size=224, mode="bilinear", align_corners=False)
    f = (f - IMAGENET_MEAN) / IMAGENET_STD
    return f.to(DEVICE)


rs = np.random.default_rng(0)
idx = rs.choice(len(frames), size=8, replace=False)
f = to_tensor(frames, idx)

with torch.no_grad():
    fmap = feature_extractor(f)  # (8, 512, 7, 7)
    tokens = fmap.flatten(2).permute(0, 2, 1)  # (8, 49, 512) -- 49 spatial "patches"

t = F.normalize(tokens, dim=-1)

# within-image: different patches, same image
img0 = t[0]  # (49, 512)
sim = img0 @ img0.T
off_diag = sim[~torch.eye(49, dtype=torch.bool, device=DEVICE)]
print(f"within-image cosine sim (different patches, same image): mean={off_diag.mean().item():.4f}  min={off_diag.min().item():.4f}  max={off_diag.max().item():.4f}")

# across-image: same patch position (center of the 7x7 grid), different images
center = 24  # index (3,3) in a 7x7 grid, flattened
v = t[:, center, :]  # (8, 512)
sim2 = v @ v.T
off_diag2 = sim2[~torch.eye(8, dtype=torch.bool, device=DEVICE)]
print(f"across-image cosine sim (different images, same patch pos):  mean={off_diag2.mean().item():.4f}  min={off_diag2.min().item():.4f}  max={off_diag2.max().item():.4f}")

with open("outputs/day99_patch_collapse_check.json", "w") as fp:
    json.dump(
        {
            "within_image_cos_sim_mean": off_diag.mean().item(),
            "within_image_cos_sim_min": off_diag.min().item(),
            "within_image_cos_sim_max": off_diag.max().item(),
            "across_image_cos_sim_mean": off_diag2.mean().item(),
        },
        fp,
        indent=2,
    )
print("\nsaved outputs/day99_patch_collapse_check.json")
