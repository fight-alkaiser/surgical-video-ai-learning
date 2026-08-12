import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ----------------------------------------
# The temporal-order backbone did WORSE on the phase probe than
# even plain frozen ImageNet features (0.460 vs 0.511) -- the
# opposite of the hypothesis motivating this day. One plausible
# explanation: the pretext task only ever compares frames from
# the SAME video, so it can succeed by learning within-video
# drift cues (lighting changes, smoke/bleeding accumulation,
# camera settings specific to one recording) that are NOT
# comparable across videos, rather than a phase-like concept
# that generalizes. If true, the test-set features should
# cluster more by WHICH VIDEO a frame came from than by phase.
# This script checks that directly, coloring the same PCA
# projection two ways.
# ----------------------------------------

DATASET_ROOT = Path("/Users/katsutoshimakino/Datasets/CholecT50/CholecT50")
VIDEOS_DIR = DATASET_ROOT / "videos"
LABELS_DIR = DATASET_ROOT / "labels"

VIDEO_IDS = [
    "VID01", "VID02", "VID04", "VID05", "VID06",
    "VID08", "VID10", "VID12", "VID13", "VID14",
]

NUM_PHASES = 7
TEST_RATIO = 0.2
BATCH_SIZE = 32
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

phase_names = None
phase_labels = {}
video_frame_ids = {}

for video_id in VIDEO_IDS:
    with open(LABELS_DIR / f"{video_id}.json") as f:
        data = json.load(f)
    if phase_names is None:
        phase_names = [data["categories"]["phase"][str(i)] for i in range(NUM_PHASES)]
    frame_ids = sorted(int(f) for f in data["annotations"].keys())
    video_frame_ids[video_id] = frame_ids
    for frame in frame_ids:
        phase_labels[(video_id, frame)] = data["annotations"][str(frame)][0][-1]

shuffled_video_ids = VIDEO_IDS[:]
random.shuffle(shuffled_video_ids)
num_test_videos = max(1, round(len(VIDEO_IDS) * TEST_RATIO))
test_video_ids = sorted(shuffled_video_ids[:num_test_videos])

test_pairs = [
    (v, f) for v in test_video_ids for f in video_frame_ids[v]
    if phase_labels[(v, f)] != -1
]

backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = backbone.fc.in_features
backbone.fc = nn.Identity()
backbone.load_state_dict(torch.load(
    Path(__file__).parent / "temporal_order_backbone.pt", map_location="cpu"
))
for p in backbone.parameters():
    p.requires_grad = False
backbone = backbone.to(device)
backbone.eval()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FrameDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        video_id, frame = self.pairs[idx]
        image = Image.open(VIDEOS_DIR / video_id / f"{frame:06d}.png").convert("RGB")
        return transform(image)


loader = DataLoader(FrameDataset(test_pairs), batch_size=BATCH_SIZE, shuffle=False)
features = []
with torch.no_grad():
    for images in loader:
        features.append(backbone(images.to(device)).cpu())
features = torch.cat(features).numpy()

test_phase_ids = np.array([phase_labels[p] for p in test_pairs])
test_video_labels = np.array([p[0] for p in test_pairs])

centered = features - features.mean(axis=0, keepdims=True)
_, _, Vt = np.linalg.svd(centered, full_matrices=False)
features_2d = centered @ Vt[:2].T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

phases_present = sorted(set(test_phase_ids.tolist()))
color_map = {p: plt.cm.tab10(i / max(1, len(phases_present) - 1)) for i, p in enumerate(phases_present)}
for p in phases_present:
    idx = test_phase_ids == p
    axes[0].scatter(features_2d[idx, 0], features_2d[idx, 1],
                     label=phase_names[p], color=color_map[p], s=8, alpha=0.6)
axes[0].set_title("Colored by phase")
axes[0].legend(fontsize=7, loc="best")

videos_present = sorted(set(test_video_labels.tolist()))
video_color_map = {v: plt.cm.Set1(i / max(1, len(videos_present) - 1)) for i, v in enumerate(videos_present)}
for v in videos_present:
    idx = test_video_labels == v
    axes[1].scatter(features_2d[idx, 0], features_2d[idx, 1],
                     label=v, color=video_color_map[v], s=8, alpha=0.6)
axes[1].set_title("Colored by video ID")
axes[1].legend(fontsize=7, loc="best")

fig.suptitle(
    "Temporal-order-pretrained features (PCA to 2D) on held-out test videos:\n"
    "same points, colored by phase (left) vs. by video (right)"
)
fig.tight_layout()
output_dir = Path(__file__).parent
fig.savefig(output_dir / "phase_vs_video_clustering.png", dpi=150)
print(f"Saved plot to {output_dir / 'phase_vs_video_clustering.png'}")
