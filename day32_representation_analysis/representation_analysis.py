import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ----------------------------------------
# Day31's contrastive pretext task never used temporal order or
# any label: it only ever compared two augmented views of the
# SAME frame against other frames treated as interchangeable
# negatives, regardless of which video or moment they came from.
# Day31's instrument linear probe showed the resulting features
# are more useful than plain frozen ImageNet features for
# instrument recognition -- but that doesn't say what the
# representation actually organizes itself around.
#
# This day asks that question directly, reusing the exact
# methodology from Day16-19 (PCA visualization + linear probe)
# but applied to a real visual backbone instead of a symbolic
# sequence model's hidden state: does PHASE structure (7
# classes, CholecT50's own procedure-stage labels) show up in
# the contrastively-adapted features, even though phase was
# never a training target and the pretext task had no temporal
# signal at all? And does it show up more or less than in plain
# frozen ImageNet features?
#
# Same 10 videos, same video-level 8/2 split as every prior
# instrument/verb/target/SSL day.
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
PROBE_EPOCHS = 200
PROBE_LEARNING_RATE = 0.5
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# ----------------------------------------
# Build per-frame phase labels (single-label, unlike
# instrument/verb/target's multi-hot) for every video.
# ----------------------------------------

phase_names = None
phase_labels = {}
video_frame_ids = {}

for video_id in VIDEO_IDS:

    with open(LABELS_DIR / f"{video_id}.json") as f:
        data = json.load(f)

    if phase_names is None:
        phase_names = [
            data["categories"]["phase"][str(i)] for i in range(NUM_PHASES)
        ]

    frame_ids = sorted(int(f) for f in data["annotations"].keys())
    video_frame_ids[video_id] = frame_ids

    for frame in frame_ids:
        phase_id = data["annotations"][str(frame)][0][-1]
        phase_labels[(video_id, frame)] = phase_id

# ----------------------------------------
# Video-level train/test split (identical to Day21/26/27/31).
# ----------------------------------------

shuffled_video_ids = VIDEO_IDS[:]
random.shuffle(shuffled_video_ids)

num_test_videos = max(1, round(len(VIDEO_IDS) * TEST_RATIO))
test_video_ids = sorted(shuffled_video_ids[:num_test_videos])
train_video_ids = sorted(shuffled_video_ids[num_test_videos:])

print(f"Train videos ({len(train_video_ids)}): {train_video_ids}")
print(f"Test videos  ({len(test_video_ids)}): {test_video_ids}")

train_pairs = [
    (v, f) for v in train_video_ids for f in video_frame_ids[v]
    if phase_labels[(v, f)] != -1
]
test_pairs = [
    (v, f) for v in test_video_ids for f in video_frame_ids[v]
    if phase_labels[(v, f)] != -1
]

print(f"Train frames: {len(train_pairs)}, Test frames: {len(test_pairs)}")

# ----------------------------------------
# Two backbones to compare: plain frozen ImageNet (no
# adaptation at all -- Day21's setup, recomputed fresh since it
# needs no training) and Day31's contrastively-adapted one
# (loaded from disk).
# ----------------------------------------

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
        image_path = VIDEOS_DIR / video_id / f"{frame:06d}.png"
        image = Image.open(image_path).convert("RGB")
        return transform(image)


def build_backbone(load_contrastive_weights):
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    if load_contrastive_weights:
        state_dict_path = (
            Path(__file__).parent.parent
            / "day31_contrastive_pretraining" / "contrastive_backbone.pt"
        )
        backbone.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    for param in backbone.parameters():
        param.requires_grad = False
    backbone = backbone.to(device)
    backbone.eval()
    return backbone, num_features


def extract_features(backbone, pairs, label_name):
    loader = DataLoader(
        FrameDataset(pairs), batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0
    )
    features = []
    start = time.time()
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            features.append(backbone(images).cpu())
    features = torch.cat(features).numpy()
    print(f"  Extracted {label_name} features: {features.shape} "
          f"in {time.time() - start:.1f}s")
    return features


train_phase_ids = np.array([phase_labels[p] for p in train_pairs])
test_phase_ids = np.array([phase_labels[p] for p in test_pairs])

variants = {}

for variant_name, load_contrastive in [
    ("imagenet_frozen", False),
    ("contrastive_adapted", True),
]:
    print(f"\n--- {variant_name} ---")
    backbone, num_features = build_backbone(load_contrastive)
    train_features = extract_features(backbone, train_pairs, "train")
    test_features = extract_features(backbone, test_pairs, "test")
    variants[variant_name] = {
        "train_features": train_features,
        "test_features": test_features,
        "num_features": num_features,
    }

# ----------------------------------------
# Linear probe for phase (7-way softmax classification, same
# from-scratch-style method as Day17/19: single linear layer,
# trained with plain SGD, frozen features).
# ----------------------------------------


def run_phase_probe(train_features, train_labels, test_features, test_labels,
                     num_classes, feature_dim):

    rng = np.random.RandomState(RANDOM_SEED)
    W = rng.randn(num_classes, feature_dim) / np.sqrt(feature_dim)
    b = np.zeros(num_classes)

    num_train = train_features.shape[0]
    loss_history = []

    for epoch in range(PROBE_EPOCHS):

        logits = train_features @ W.T + b
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)

        loss = -np.log(
            probs[np.arange(num_train), train_labels] + 1e-12
        ).mean()
        loss_history.append(float(loss))

        dlogits = probs.copy()
        dlogits[np.arange(num_train), train_labels] -= 1
        dlogits /= num_train

        W -= PROBE_LEARNING_RATE * (dlogits.T @ train_features)
        b -= PROBE_LEARNING_RATE * dlogits.sum(axis=0)

    test_logits = test_features @ W.T + b
    predicted = test_logits.argmax(axis=1)
    accuracy = (predicted == test_labels).mean()

    return accuracy, loss_history


results = {"variants": {}}

print()
for variant_name, data_dict in variants.items():

    accuracy, loss_history = run_phase_probe(
        data_dict["train_features"], train_phase_ids,
        data_dict["test_features"], test_phase_ids,
        NUM_PHASES, data_dict["num_features"],
    )

    baseline_phase = np.bincount(train_phase_ids).argmax()
    baseline_accuracy = (test_phase_ids == baseline_phase).mean()

    print(f"{variant_name:22s} phase-probe accuracy = {accuracy:.3f} "
          f"(baseline {baseline_accuracy:.3f})")

    results["variants"][variant_name] = {
        "phase_probe_accuracy": float(accuracy),
        "baseline_accuracy": float(baseline_accuracy),
        "probe_loss_history": loss_history,
    }

# ----------------------------------------
# PCA visualization (plain SVD, no sklearn -- same convention
# as Day16-19) of the TEST set's features, colored by true
# phase, for both variants side by side.
# ----------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, variant_name in zip(axes, ["imagenet_frozen", "contrastive_adapted"]):

    features = variants[variant_name]["test_features"]
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    features_2d = centered @ Vt[:2].T

    phases_present = sorted(set(test_phase_ids.tolist()))
    color_map = {
        p: plt.cm.tab10(i / max(1, len(phases_present) - 1))
        for i, p in enumerate(phases_present)
    }

    for p in phases_present:
        idx = test_phase_ids == p
        ax.scatter(
            features_2d[idx, 0], features_2d[idx, 1],
            label=phase_names[p], color=color_map[p], s=8, alpha=0.6
        )

    ax.set_title(variant_name)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

axes[1].legend(fontsize=7, loc="best", bbox_to_anchor=(1.0, 1.0))
fig.suptitle(
    "Frame features (PCA to 2D) on held-out test videos, colored by "
    "true phase\n(phase never used during pretraining or feature extraction)"
)
fig.tight_layout()
output_dir = Path(__file__).parent
fig.savefig(output_dir / "phase_structure_comparison.png", dpi=150)
print(f"\nSaved plot to {output_dir / 'phase_structure_comparison.png'}")

with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
