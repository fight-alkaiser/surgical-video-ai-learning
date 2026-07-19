import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ----------------------------------------
# Day21 found instrument recognition works well for common
# instruments (grasper F1 0.860, hook F1 0.677) but barely
# works for rare ones (bipolar 0.106, scissors 0.054, clipper
# 0.012, irrigator 0.100) -- Day21's Reflection named three
# candidate fixes: more videos, fine-tuning more of the
# backbone, and a class-weighted loss. Today isolates the
# cheapest and fastest of the three: keep everything else
# exactly as Day21 (same 10 videos, same frozen ResNet18 +
# linear head, same split), and change only the loss function
# from plain BCEWithLogitsLoss to a version that weights each
# instrument's positive examples by how rare that instrument
# is in training data (pos_weight = num_negative /
# num_positive, a standard imbalanced-classification
# technique). This isolates one specific question: how much
# of the rare-instrument problem is fixable by re-weighting
# the SAME data and SAME frozen features, versus needing more
# data or a better feature extractor (backbone fine-tuning,
# left for a later day)?
# ----------------------------------------

DATASET_ROOT = Path("/Users/katsutoshimakino/Datasets/CholecT50/CholecT50")
VIDEOS_DIR = DATASET_ROOT / "videos"
LABELS_DIR = DATASET_ROOT / "labels"

VIDEO_IDS = [
    "VID01", "VID02", "VID04", "VID05", "VID06",
    "VID08", "VID10", "VID12", "VID13", "VID14",
]

NUM_INSTRUMENTS = 6
TEST_RATIO = 0.2
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# ----------------------------------------
# Build per-frame instrument multi-hot labels (identical
# extraction logic to Day21).
# ----------------------------------------

instrument_names = None
instrument_labels = {}
video_frame_ids = {}

for video_id in VIDEO_IDS:

    with open(LABELS_DIR / f"{video_id}.json") as f:
        data = json.load(f)

    if instrument_names is None:
        instrument_names = [
            data["categories"]["instrument"][str(i)]
            for i in range(NUM_INSTRUMENTS)
        ]

    frame_ids = sorted(int(f) for f in data["annotations"].keys())
    video_frame_ids[video_id] = frame_ids

    for frame in frame_ids:
        instruments_present = set()
        for triplet in data["annotations"][str(frame)]:
            instrument_id = triplet[1]
            if instrument_id != -1:
                instruments_present.add(instrument_id)
        label = np.zeros(NUM_INSTRUMENTS, dtype=np.float32)
        for iid in instruments_present:
            label[iid] = 1.0
        instrument_labels[(video_id, frame)] = label

# ----------------------------------------
# Video-level train/test split (identical to Day21-25).
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
]
test_pairs = [
    (v, f) for v in test_video_ids for f in video_frame_ids[v]
]

print(f"Train frames: {len(train_pairs)}, Test frames: {len(test_pairs)}")

# ----------------------------------------
# Step 1: extract and cache frozen ResNet18 features once
# (same backbone as Day20-25).
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


backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in backbone.parameters():
    param.requires_grad = False
num_features = backbone.fc.in_features
backbone.fc = nn.Identity()
backbone = backbone.to(device)
backbone.eval()


def extract_features(pairs, label_name):
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
    print(f"Extracted {label_name} features: {features.shape} "
          f"in {time.time() - start:.1f}s")
    return features


train_features = extract_features(train_pairs, "train")
test_features = extract_features(test_pairs, "test")

train_targets = np.stack([instrument_labels[p] for p in train_pairs])
test_targets = np.stack([instrument_labels[p] for p in test_pairs])

train_features_t = torch.from_numpy(train_features).float()
test_features_t = torch.from_numpy(test_features).float()
train_targets_t = torch.from_numpy(train_targets).float()

# ----------------------------------------
# Step 2: compute per-instrument pos_weight from training
# prevalence (standard imbalanced BCE technique): a rare
# instrument's positive examples get weighted up so that
# missing one costs the loss as much, in expectation, as
# missing a common instrument's positive example.
# ----------------------------------------

train_prevalence = train_targets.mean(axis=0)
num_positive = train_targets.sum(axis=0)
num_negative = len(train_pairs) - num_positive
pos_weight = num_negative / np.maximum(num_positive, 1)

print()
print("Per-instrument training prevalence and pos_weight:")
for i, name in enumerate(instrument_names):
    print(f"  {name:12s} prevalence={train_prevalence[i]:.3f} "
          f"pos_weight={pos_weight[i]:.2f}")

pos_weight_t = torch.from_numpy(pos_weight.astype(np.float32)).to(device)

# ----------------------------------------
# Step 3: train the linear head with class-weighted BCE loss
# on cached features (same architecture as Day21, only the
# loss function differs).
# ----------------------------------------

head = nn.Linear(num_features, NUM_INSTRUMENTS).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE)

num_train = train_features_t.shape[0]
loss_history = []

for epoch in range(NUM_EPOCHS):

    head.train()
    permutation = torch.randperm(num_train)
    epoch_loss = 0.0
    num_batches = 0

    for start_idx in range(0, num_train, BATCH_SIZE):
        idx = permutation[start_idx:start_idx + BATCH_SIZE]
        batch_x = train_features_t[idx].to(device)
        batch_y = train_targets_t[idx].to(device)

        optimizer.zero_grad()
        logits = head(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: train loss = {avg_loss:.4f}")

# ----------------------------------------
# Evaluate: per-instrument accuracy and F1 on held-out test
# videos, vs. the same train-majority baseline and Day21's
# unweighted results, for direct comparison.
# ----------------------------------------

head.eval()
with torch.no_grad():
    test_logits = head(test_features_t.to(device))
    test_predictions = (torch.sigmoid(test_logits) > 0.5).float().cpu().numpy()

all_predictions = test_predictions
all_labels = test_targets

baseline_prediction = (train_prevalence > 0.5).astype(np.float32)
baseline_predictions = np.tile(baseline_prediction, (len(test_pairs), 1))

day21_f1_reference = {
    "grasper": 0.860, "bipolar": 0.106, "hook": 0.677,
    "scissors": 0.054, "clipper": 0.012, "irrigator": 0.100,
}

print()
print(f"{'Instrument':12s} {'Accuracy':>10s} {'F1':>8s} "
      f"{'Baseline Acc':>14s} {'Day21 F1':>10s} {'Test Prev':>10s}")

results = {"instruments": {}}

for i, name in enumerate(instrument_names):

    pred_i = all_predictions[:, i]
    label_i = all_labels[:, i]

    accuracy = (pred_i == label_i).mean()

    tp = ((pred_i == 1) & (label_i == 1)).sum()
    fp = ((pred_i == 1) & (label_i == 0)).sum()
    fn = ((pred_i == 0) & (label_i == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    baseline_accuracy = (baseline_predictions[:, i] == label_i).mean()
    test_prevalence = label_i.mean()

    print(f"{name:12s} {accuracy:10.3f} {f1:8.3f} "
          f"{baseline_accuracy:14.3f} {day21_f1_reference[name]:10.3f} "
          f"{test_prevalence:10.3f}")

    results["instruments"][name] = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "baseline_accuracy": float(baseline_accuracy),
        "day21_f1": day21_f1_reference[name],
        "precision": float(precision),
        "recall": float(recall),
        "test_prevalence": float(test_prevalence),
    }

macro_accuracy = np.mean(
    [results["instruments"][n]["accuracy"] for n in instrument_names]
)
macro_f1 = np.mean([results["instruments"][n]["f1"] for n in instrument_names])
macro_baseline_accuracy = np.mean(
    [results["instruments"][n]["baseline_accuracy"] for n in instrument_names]
)

print()
print(f"Macro accuracy: {macro_accuracy:.3f} "
      f"(baseline: {macro_baseline_accuracy:.3f})")
print(f"Macro F1:       {macro_f1:.3f}")
print()
print("For reference, Day21 (unweighted loss): "
      "macro accuracy 0.894, macro F1 0.302")

results["macro_accuracy"] = float(macro_accuracy)
results["macro_f1"] = float(macro_f1)
results["macro_baseline_accuracy"] = float(macro_baseline_accuracy)
results["loss_history"] = loss_history
results["pos_weight"] = pos_weight.tolist()
results["train_video_ids"] = train_video_ids
results["test_video_ids"] = test_video_ids
results["num_train_frames"] = len(train_pairs)
results["num_test_frames"] = len(test_pairs)

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
