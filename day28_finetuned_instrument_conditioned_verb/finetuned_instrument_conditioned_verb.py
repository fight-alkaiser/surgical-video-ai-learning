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
# This day combines the two improvements found separately in
# Days 24-27:
#
#   - Day24: conditioning verb prediction on a real instrument
#     classifier's PREDICTED probabilities raised macro F1 from
#     0.192 (no conditioning) to only 0.241 -- a real but small
#     gain, because Day21's frozen-feature instrument classifier
#     was itself unreliable for exactly the rare instruments
#     (clipper F1 0.012, bipolar 0.106) that would have helped
#     tool-specific verbs the most.
#   - Day27: fine-tuning ResNet18's last block (layer4), instead
#     of using it fully frozen, raised those same instruments'
#     F1 dramatically (clipper 0.012 -> 0.492, bipolar 0.106 ->
#     0.431) by improving feature quality itself, not just a
#     loss-threshold trade-off.
#
# If Day24's shortfall from Day23's oracle ceiling (0.388) was
# mainly caused by Day21's weak instrument classifier, then
# conditioning verb prediction on Day27's much-improved
# instrument classifier should recover more of that gap. Today
# tests this directly: fine-tune the backbone once for
# instrument recognition (Day27's setup), freeze it, cache its
# features (now much better than Day21's), and train a verb
# classifier on [cached fine-tuned features + predicted
# instrument probabilities from the fine-tuned classifier].
#
# Reference points, all on the same 10 videos / video-level
# 8-2 split:
#   Day22 (frozen features, no conditioning):        F1 0.192
#   Day24 (frozen features, predicted conditioning):  F1 0.241
#   Day23 (frozen features, ORACLE conditioning):     F1 0.388
#   Day28 (fine-tuned features, predicted conditioning): ?
# ----------------------------------------

DATASET_ROOT = Path("/Users/katsutoshimakino/Datasets/CholecT50/CholecT50")
VIDEOS_DIR = DATASET_ROOT / "videos"
LABELS_DIR = DATASET_ROOT / "labels"

VIDEO_IDS = [
    "VID01", "VID02", "VID04", "VID05", "VID06",
    "VID08", "VID10", "VID12", "VID13", "VID14",
]

NUM_INSTRUMENTS = 6
NUM_VERBS = 10
TEST_RATIO = 0.2
BATCH_SIZE = 16
NUM_EPOCHS_BACKBONE = 8
NUM_EPOCHS_HEAD = 10
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
# Build per-frame instrument AND verb multi-hot labels
# (identical extraction logic to Day21-27).
# ----------------------------------------

instrument_names = None
verb_names = None
instrument_labels = {}
verb_labels = {}
video_frame_ids = {}

for video_id in VIDEO_IDS:

    with open(LABELS_DIR / f"{video_id}.json") as f:
        data = json.load(f)

    if instrument_names is None:
        instrument_names = [
            data["categories"]["instrument"][str(i)]
            for i in range(NUM_INSTRUMENTS)
        ]
        verb_names = [
            data["categories"]["verb"][str(i)] for i in range(NUM_VERBS)
        ]

    frame_ids = sorted(int(f) for f in data["annotations"].keys())
    video_frame_ids[video_id] = frame_ids

    for frame in frame_ids:

        instruments_present = set()
        verbs_present = set()

        for triplet in data["annotations"][str(frame)]:
            instrument_id = triplet[1]
            verb_id = triplet[7]
            if instrument_id != -1:
                instruments_present.add(instrument_id)
            if verb_id != -1:
                verbs_present.add(verb_id)

        instrument_label = np.zeros(NUM_INSTRUMENTS, dtype=np.float32)
        for iid in instruments_present:
            instrument_label[iid] = 1.0
        instrument_labels[(video_id, frame)] = instrument_label

        verb_label = np.zeros(NUM_VERBS, dtype=np.float32)
        for vid_ in verbs_present:
            verb_label[vid_] = 1.0
        verb_labels[(video_id, frame)] = verb_label

# ----------------------------------------
# Video-level train/test split (identical to Day21-27).
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

train_instrument_targets = np.stack(
    [instrument_labels[p] for p in train_pairs]
)
instrument_prevalence = train_instrument_targets.mean(axis=0)
num_positive = train_instrument_targets.sum(axis=0)
num_negative = len(train_pairs) - num_positive
instrument_pos_weight = (
    num_negative / np.maximum(num_positive, 1)
).astype(np.float32)
instrument_pos_weight_t = torch.from_numpy(instrument_pos_weight).to(device)

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


class FrameDataset(Dataset):

    def __init__(self, pairs, labels_by_pair):
        self.pairs = pairs
        self.labels_by_pair = labels_by_pair

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        video_id, frame = self.pairs[idx]
        image_path = VIDEOS_DIR / video_id / f"{frame:06d}.png"
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        label = torch.from_numpy(self.labels_by_pair[(video_id, frame)])
        return image, label


train_instrument_loader = DataLoader(
    FrameDataset(train_pairs, instrument_labels), batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)

# ----------------------------------------
# Step 1: fine-tune ResNet18 for instrument recognition
# (identical setup to Day27: layer4 + new fc trainable,
# class-weighted loss, earlier layers frozen with BatchNorm
# kept in eval mode).
# ----------------------------------------

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

frozen_modules = [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]
for module in frozen_modules:
    for param in module.parameters():
        param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_INSTRUMENTS)
model = model.to(device)


def set_training_mode():
    model.train()
    for module in frozen_modules:
        module.eval()


criterion = nn.BCEWithLogitsLoss(pos_weight=instrument_pos_weight_t)
optimizer = torch.optim.Adam([
    {"params": model.layer4.parameters(), "lr": LEARNING_RATE_BACKBONE},
    {"params": model.fc.parameters(), "lr": LEARNING_RATE_HEAD},
])

print("\n--- Step 1: fine-tuning backbone for instrument recognition ---")
for epoch in range(NUM_EPOCHS_BACKBONE):

    set_training_mode()
    epoch_loss = 0.0
    num_batches = 0
    start = time.time()

    for images, labels in train_instrument_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    print(f"[instrument] Epoch {epoch + 1}/{NUM_EPOCHS_BACKBONE}: "
          f"train loss = {avg_loss:.4f} ({time.time() - start:.0f}s)")

# ----------------------------------------
# Step 2: freeze everything (the backbone is now fine-tuned
# and fixed), extract and cache its 512-d features for every
# frame -- same shortcut as Day24, now on top of better
# features instead of the original frozen ImageNet ones.
# ----------------------------------------

for param in model.parameters():
    param.requires_grad = False

instrument_head = model.fc
model.fc = nn.Identity()
model.eval()


def extract_features(pairs, label_name):
    loader = DataLoader(
        FrameDataset(pairs, instrument_labels), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    features = []
    start = time.time()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            features.append(model(images).cpu())
    features = torch.cat(features).numpy()
    print(f"Extracted {label_name} features: {features.shape} "
          f"in {time.time() - start:.1f}s")
    return features


print("\n--- Step 2: caching fine-tuned features ---")
train_features = extract_features(train_pairs, "train")
test_features = extract_features(test_pairs, "test")

train_features_t = torch.from_numpy(train_features).float()
test_features_t = torch.from_numpy(test_features).float()

# ----------------------------------------
# Step 3: get PREDICTED instrument probabilities from the
# fine-tuned classifier's own head, on cached features (never
# the true label).
# ----------------------------------------

instrument_head = instrument_head.to(device)
instrument_head.eval()
with torch.no_grad():
    train_instrument_probs = torch.sigmoid(
        instrument_head(train_features_t.to(device))
    ).cpu()
    test_instrument_probs = torch.sigmoid(
        instrument_head(test_features_t.to(device))
    ).cpu()

test_instrument_targets = np.stack(
    [instrument_labels[p] for p in test_pairs]
)
instrument_test_accuracy = (
    (test_instrument_probs.numpy() > 0.5).astype(np.float32)
    == test_instrument_targets
).mean()
print(f"\nFine-tuned instrument classifier mean per-class test accuracy: "
      f"{instrument_test_accuracy:.3f} (Day27 reference: 0.932)")

# ----------------------------------------
# Step 4: train the verb classifier on [cached fine-tuned
# features + predicted instrument probabilities], plain
# (unweighted) BCE loss -- matching Day22/24's verb loss
# exactly, so the comparison isolates feature quality.
# ----------------------------------------

train_verb_targets = np.stack([verb_labels[p] for p in train_pairs])
test_verb_targets = np.stack([verb_labels[p] for p in test_pairs])
train_verb_t = torch.from_numpy(train_verb_targets).float()

train_combined = torch.cat([train_features_t, train_instrument_probs], dim=1)
test_combined = torch.cat([test_features_t, test_instrument_probs], dim=1)

verb_head = nn.Linear(num_features + NUM_INSTRUMENTS, NUM_VERBS).to(device)
verb_criterion = nn.BCEWithLogitsLoss()
verb_optimizer = torch.optim.Adam(verb_head.parameters(), lr=LEARNING_RATE_HEAD)

num_train = train_combined.shape[0]
verb_loss_history = []

print("\n--- Step 3: training verb classifier on fine-tuned + "
      "instrument-conditioned features ---")
for epoch in range(NUM_EPOCHS_HEAD):

    verb_head.train()
    permutation = torch.randperm(num_train)
    epoch_loss = 0.0
    num_batches = 0

    for start_idx in range(0, num_train, BATCH_SIZE):
        idx = permutation[start_idx:start_idx + BATCH_SIZE]
        batch_x = train_combined[idx].to(device)
        batch_y = train_verb_t[idx].to(device)

        verb_optimizer.zero_grad()
        logits = verb_head(batch_x)
        loss = verb_criterion(logits, batch_y)
        loss.backward()
        verb_optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    verb_loss_history.append(avg_loss)
    print(f"[verb] Epoch {epoch + 1}/{NUM_EPOCHS_HEAD}: "
          f"train loss = {avg_loss:.4f}")

# ----------------------------------------
# Evaluate
# ----------------------------------------

verb_head.eval()
with torch.no_grad():
    test_logits = verb_head(test_combined.to(device))
    test_predictions = (torch.sigmoid(test_logits) > 0.5).float().cpu().numpy()

train_verb_prevalence = train_verb_targets.mean(axis=0)
baseline_prediction = (train_verb_prevalence > 0.5).astype(np.float32)
baseline_predictions = np.tile(baseline_prediction, (len(test_pairs), 1))

day22_f1_reference = {
    "grasp": 0.434, "retract": 0.692, "dissect": 0.652, "coagulate": 0.052,
    "clip": 0.000, "cut": 0.000, "aspirate": 0.045, "irrigate": 0.000,
    "pack": 0.000, "null_verb": 0.042,
}
day24_f1_reference = {
    "grasp": 0.402, "retract": 0.744, "dissect": 0.663, "coagulate": 0.023,
    "clip": 0.119, "cut": 0.000, "aspirate": 0.169, "irrigate": 0.000,
    "pack": 0.250, "null_verb": 0.044,
}

print()
print(f"{'Verb':12s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s} "
      f"{'Day22 F1':>10s} {'Day24 F1':>10s}")

results = {"verbs": {}}

for i, name in enumerate(verb_names):

    pred_i = test_predictions[:, i]
    label_i = test_verb_targets[:, i]

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

    print(f"{name:12s} {f1:8.3f} {precision:10.3f} {recall:8.3f} "
          f"{day22_f1_reference[name]:10.3f} {day24_f1_reference[name]:10.3f}")

    results["verbs"][name] = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "day22_f1": day22_f1_reference[name],
        "day24_f1": day24_f1_reference[name],
    }

macro_accuracy = np.mean(
    [results["verbs"][n]["accuracy"] for n in verb_names]
)
macro_f1 = np.mean([results["verbs"][n]["f1"] for n in verb_names])

print()
print(f"Macro accuracy: {macro_accuracy:.3f}")
print(f"Macro F1:       {macro_f1:.3f}")
print()
print("For reference:")
print("  Day22 (frozen, no conditioning):            F1 0.192")
print("  Day24 (frozen, predicted conditioning):      F1 0.241")
print("  Day23 (frozen, ORACLE conditioning):         F1 0.388")

results["macro_accuracy"] = float(macro_accuracy)
results["macro_f1"] = float(macro_f1)
results["instrument_test_accuracy"] = float(instrument_test_accuracy)
results["verb_loss_history"] = verb_loss_history
results["train_video_ids"] = train_video_ids
results["test_video_ids"] = test_video_ids
results["num_train_frames"] = len(train_pairs)
results["num_test_frames"] = len(test_pairs)

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
