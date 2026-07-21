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
# Day26 found class-weighted loss fixes a model's willingness
# to guess a rare instrument (clipper: F1 0.012 -> 0.291), but
# not scissors specifically (0.054 -> 0.050, despite a
# comparable recall gain) -- its precision stayed near zero
# (0.026), suggesting the frozen ImageNet backbone's features
# don't separate scissors from other instruments well, no
# matter how the loss threshold is adjusted. Today tests the
# other fix Day21 named: unfreeze ResNet18's last residual
# block (layer4) and fine-tune it -- along with Day26's
# class-weighted loss, kept constant -- so the visual
# features themselves can adapt to this specific dataset,
# rather than staying fixed at whatever ImageNet happened to
# learn.
#
# conv1/bn1/layer1/layer2/layer3 (the earlier, more generic
# layers -- edges, textures, simple shapes) stay frozen, same
# as Day20-26. Only layer4 (higher-level, more task-specific
# features) and the final linear layer are trained. This
# isolates one variable relative to Day26: same 10 videos,
# same split, same class-weighted loss, only the backbone's
# trainability changes.
#
# This machine has 8GB RAM, which rules out Day24-26's
# feature-caching shortcut (caching layer3's output for every
# frame would need several GB by itself). Training here goes
# back to a live loop like Day21's original script -- forward
# through the whole network every batch, backward only into
# layer4 and the final layer.
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
# Build per-frame instrument multi-hot labels (identical
# extraction logic to Day21/26).
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
# Video-level train/test split (identical to Day21-26).
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

train_targets = np.stack([instrument_labels[p] for p in train_pairs])
train_prevalence = train_targets.mean(axis=0)
num_positive = train_targets.sum(axis=0)
num_negative = len(train_pairs) - num_positive
pos_weight = (num_negative / np.maximum(num_positive, 1)).astype(np.float32)
pos_weight_t = torch.from_numpy(pos_weight).to(device)

print("\nPer-instrument pos_weight (same formula as Day26):")
for name, w in zip(instrument_names, pos_weight):
    print(f"  {name:12s} pos_weight={w:.2f}")

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


class InstrumentDataset(Dataset):

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        video_id, frame = self.pairs[idx]
        image_path = VIDEOS_DIR / video_id / f"{frame:06d}.png"
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        label = torch.from_numpy(instrument_labels[(video_id, frame)])
        return image, label


train_loader = DataLoader(
    InstrumentDataset(train_pairs), batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)
test_loader = DataLoader(
    InstrumentDataset(test_pairs), batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

# ----------------------------------------
# Model: ResNet18, conv1/bn1/layer1/layer2/layer3 frozen,
# layer4 + new final linear layer trainable.
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
    """Put layer4/fc in train mode, but keep the frozen layers
    (including their BatchNorm running statistics) in eval
    mode -- otherwise calling model.train() would let frozen
    layers' BatchNorm stats drift even though their weights
    don't update, silently changing their behavior."""
    model.train()
    for module in frozen_modules:
        module.eval()


criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

trainable_backbone_params = list(model.layer4.parameters())
head_params = list(model.fc.parameters())

optimizer = torch.optim.Adam([
    {"params": trainable_backbone_params, "lr": LEARNING_RATE_BACKBONE},
    {"params": head_params, "lr": LEARNING_RATE_HEAD},
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
# Evaluate: per-instrument accuracy, F1, precision, recall on
# held-out test videos, vs. Day21 (unweighted, frozen) and
# Day26 (weighted, frozen) for direct comparison.
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

baseline_prediction = (train_prevalence > 0.5).astype(np.float32)
baseline_predictions = np.tile(baseline_prediction, (len(test_pairs), 1))

day21_f1_reference = {
    "grasper": 0.860, "bipolar": 0.106, "hook": 0.677,
    "scissors": 0.054, "clipper": 0.012, "irrigator": 0.100,
}
day26_f1_reference = {
    "grasper": 0.859, "bipolar": 0.182, "hook": 0.735,
    "scissors": 0.050, "clipper": 0.291, "irrigator": 0.148,
}

print()
print(f"{'Instrument':12s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s} "
      f"{'Day21 F1':>10s} {'Day26 F1':>10s}")

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

    print(f"{name:12s} {f1:8.3f} {precision:10.3f} {recall:8.3f} "
          f"{day21_f1_reference[name]:10.3f} {day26_f1_reference[name]:10.3f}")

    results["instruments"][name] = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "day21_f1": day21_f1_reference[name],
        "day26_f1": day26_f1_reference[name],
    }

macro_accuracy = np.mean(
    [results["instruments"][n]["accuracy"] for n in instrument_names]
)
macro_f1 = np.mean([results["instruments"][n]["f1"] for n in instrument_names])

print()
print(f"Macro accuracy: {macro_accuracy:.3f}")
print(f"Macro F1:       {macro_f1:.3f}")
print()
print("For reference: Day21 (frozen, unweighted) macro F1 0.302; "
      "Day26 (frozen, class-weighted) macro F1 0.378")

results["macro_accuracy"] = float(macro_accuracy)
results["macro_f1"] = float(macro_f1)
results["loss_history"] = loss_history
results["pos_weight"] = pos_weight.tolist()
results["train_video_ids"] = train_video_ids
results["test_video_ids"] = test_video_ids
results["num_train_frames"] = len(train_pairs)
results["num_test_frames"] = len(test_pairs)

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
