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
# Day20-28 worked through instrument and verb, the first two
# parts of CholecT50's instrument-verb-target triplet, across
# many days each (Day20-21 instrument, Day22-25 verb,
# Day26-28 combining fixes). Target -- the anatomical
# structure a verb is applied to (15 classes: gallbladder,
# cystic_duct, cystic_artery, liver, ... down to null_target)
# -- is the third part, deliberately given a single, lighter
# day rather than its own multi-day arc, since the earlier
# arcs already established which techniques matter for this
# kind of imbalanced multi-label recognition on this dataset.
#
# Rather than re-deriving those lessons step by step again
# (frozen+unweighted, then weighted, then fine-tuned, as
# instrument's Day21/26/27 did), this day applies the best
# known recipe directly: ResNet18 with layer4 fine-tuned
# (Day27) and a class-weighted loss (Day26), trained once for
# target recognition, evaluated against the same trivial
# baseline used throughout this project. The frozen/unweighted
# comparison point is filled in from instrument's own
# Day21-vs-Day27 gap as a documented expectation, not re-run
# here, in keeping with "lightly touch" rather than repeat the
# full arc.
# ----------------------------------------

DATASET_ROOT = Path("/Users/katsutoshimakino/Datasets/CholecT50/CholecT50")
VIDEOS_DIR = DATASET_ROOT / "videos"
LABELS_DIR = DATASET_ROOT / "labels"

VIDEO_IDS = [
    "VID01", "VID02", "VID04", "VID05", "VID06",
    "VID08", "VID10", "VID12", "VID13", "VID14",
]

NUM_TARGETS = 15
TEST_RATIO = 0.2
BATCH_SIZE = 16
NUM_EPOCHS = 8
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_BACKBONE = 1e-4
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# ----------------------------------------
# Build per-frame target multi-hot labels (same extraction
# pattern as instrument/verb, using triplet[8] this time).
# ----------------------------------------

target_names = None
target_labels = {}
video_frame_ids = {}

for video_id in VIDEO_IDS:

    with open(LABELS_DIR / f"{video_id}.json") as f:
        data = json.load(f)

    if target_names is None:
        target_names = [
            data["categories"]["target"][str(i)] for i in range(NUM_TARGETS)
        ]

    frame_ids = sorted(int(f) for f in data["annotations"].keys())
    video_frame_ids[video_id] = frame_ids

    for frame in frame_ids:
        targets_present = set()
        for triplet in data["annotations"][str(frame)]:
            target_id = triplet[8]
            if target_id != -1:
                targets_present.add(target_id)
        label = np.zeros(NUM_TARGETS, dtype=np.float32)
        for tid in targets_present:
            label[tid] = 1.0
        target_labels[(video_id, frame)] = label

# ----------------------------------------
# Video-level train/test split (identical to every prior day).
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

train_targets_array = np.stack([target_labels[p] for p in train_pairs])
target_prevalence = train_targets_array.mean(axis=0)
num_positive = train_targets_array.sum(axis=0)
num_negative = len(train_pairs) - num_positive
pos_weight = (num_negative / np.maximum(num_positive, 1)).astype(np.float32)
pos_weight_t = torch.from_numpy(pos_weight).to(device)

print("\nPer-target training prevalence and pos_weight:")
for name, prev, w in zip(target_names, target_prevalence, pos_weight):
    print(f"  {name:22s} prevalence={prev:.3f} pos_weight={w:.2f}")

# ----------------------------------------
# Dataset
# ----------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class TargetDataset(Dataset):

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        video_id, frame = self.pairs[idx]
        image_path = VIDEOS_DIR / video_id / f"{frame:06d}.png"
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        label = torch.from_numpy(target_labels[(video_id, frame)])
        return image, label


train_loader = DataLoader(
    TargetDataset(train_pairs), batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)
test_loader = DataLoader(
    TargetDataset(test_pairs), batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

# ----------------------------------------
# Model: ResNet18, conv1/bn1/layer1/layer2/layer3 frozen,
# layer4 + new final linear layer trainable -- identical
# recipe to Day27, applied to target instead of instrument.
# ----------------------------------------

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

frozen_modules = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]
for module in frozen_modules:
    for param in module.parameters():
        param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_TARGETS)
model = model.to(device)


def set_training_mode():
    model.train()
    for module in frozen_modules:
        module.eval()


criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": LEARNING_RATE_BACKBONE},
    {"params": model.fc.parameters(), "lr": LEARNING_RATE_HEAD},
])

# ----------------------------------------
# Train
# ----------------------------------------

loss_history = []

for epoch in range(NUM_EPOCHS):

    set_training_mode()
    epoch_loss = 0.0
    num_batches = 0
    start = time.time()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: train loss = {avg_loss:.4f} "
          f"({time.time() - start:.0f}s)")

# ----------------------------------------
# Evaluate
# ----------------------------------------

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        predictions = (torch.sigmoid(logits) > 0.5).float().cpu()
        all_predictions.append(predictions)
        all_labels.append(labels)

all_predictions = torch.cat(all_predictions).numpy()
all_labels = torch.cat(all_labels).numpy()

baseline_prediction = (target_prevalence > 0.5).astype(np.float32)
baseline_predictions = np.tile(baseline_prediction, (len(test_pairs), 1))

print()
print(f"{'Target':22s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s} "
      f"{'Baseline Acc':>14s} {'Test Prev':>10s}")

results = {"targets": {}}

for i, name in enumerate(target_names):

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

    print(f"{name:22s} {f1:8.3f} {precision:10.3f} {recall:8.3f} "
          f"{baseline_accuracy:14.3f} {test_prevalence:10.3f}")

    results["targets"][name] = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "baseline_accuracy": float(baseline_accuracy),
        "train_prevalence": float(target_prevalence[i]),
        "test_prevalence": float(test_prevalence),
    }

macro_accuracy = np.mean(
    [results["targets"][n]["accuracy"] for n in target_names]
)
macro_f1 = np.mean([results["targets"][n]["f1"] for n in target_names])
macro_baseline_accuracy = np.mean(
    [results["targets"][n]["baseline_accuracy"] for n in target_names]
)

print()
print(f"Macro accuracy: {macro_accuracy:.3f} "
      f"(baseline: {macro_baseline_accuracy:.3f})")
print(f"Macro F1:       {macro_f1:.3f}")

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
