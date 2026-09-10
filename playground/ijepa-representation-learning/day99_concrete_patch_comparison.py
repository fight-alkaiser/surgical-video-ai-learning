"""Day99 retrospective aid: pick one real frame, one background patch and one
instrument-region patch, and show concretely -- not just as an aggregate
cosine-similarity number -- what the collapsed (Day93, reproduced) and fixed
encoders actually output for two visually different patches of the same
image.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ijepa_model import IJEPAModel, patchify, NUM_PATCHES, GRID, PATCH_SIZE

DATA_DIR = "../action-conditioned-video-prediction/data"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

frames = np.load(f"{DATA_DIR}/episodes/episode_000002_frames.npy")  # a val episode
frame_idx = 150
frame = frames[frame_idx]  # (64, 64, 3) uint8

f = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # (1,3,64,64)
f = f.to(DEVICE)


def load_model(path):
    m = IJEPAModel().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


collapsed = load_model("outputs/model_ijepa_day93_collapsed_repro.pt")
fixed = load_model("outputs/model_ijepa_seed0_lr0.001_ema0.996_clip1.0.pt")

# patch grid is 8x8 (GRID=8), each patch 8x8 pixels (PATCH_SIZE=8) on the 64x64 frame
# corner patch (0,0) -> likely background/table; a center-ish patch -> likely instrument region
bg_row, bg_col = 0, 0
inst_row, inst_col = 3, 4
bg_idx = bg_row * GRID + bg_col
inst_idx = inst_row * GRID + inst_col

with torch.no_grad():
    patches = patchify(f)
    pos = torch.arange(NUM_PATCHES, device=DEVICE).unsqueeze(0)
    for name, model in [("collapsed (Day93 bug, reproduced)", collapsed), ("fixed (current)", fixed)]:
        tokens = model.context_encoder(patches, pos)[0]  # (64, embed_dim)
        v_bg = F.normalize(tokens[bg_idx : bg_idx + 1], dim=-1)
        v_inst = F.normalize(tokens[inst_idx : inst_idx + 1], dim=-1)
        cos_sim = (v_bg @ v_inst.T).item()
        print(f"{name}: cosine similarity between background patch and instrument-region patch = {cos_sim:.4f}")

# save the actual frame with the two patches boxed, plus the two patches themselves zoomed in
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(frame)
for (row, col), color, label in [((bg_row, bg_col), "cyan", "background patch"), ((inst_row, inst_col), "magenta", "instrument-region patch")]:
    y, x = row * PATCH_SIZE, col * PATCH_SIZE
    axes[0].add_patch(plt.Rectangle((x, y), PATCH_SIZE, PATCH_SIZE, edgecolor=color, facecolor="none", linewidth=2))
axes[0].set_title(f"full frame (episode_000002, frame {frame_idx})\ncyan=background patch, magenta=instrument patch")
axes[0].axis("off")

bg_patch = frame[bg_row * PATCH_SIZE : (bg_row + 1) * PATCH_SIZE, bg_col * PATCH_SIZE : (bg_col + 1) * PATCH_SIZE]
inst_patch = frame[inst_row * PATCH_SIZE : (inst_row + 1) * PATCH_SIZE, inst_col * PATCH_SIZE : (inst_col + 1) * PATCH_SIZE]
axes[1].imshow(bg_patch)
axes[1].set_title("background patch\n(8x8 px, zoomed)")
axes[1].axis("off")
axes[2].imshow(inst_patch)
axes[2].set_title("instrument-region patch\n(8x8 px, zoomed)")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("outputs/day99_concrete_patches.png", dpi=150, bbox_inches="tight")
print("saved outputs/day99_concrete_patches.png")
