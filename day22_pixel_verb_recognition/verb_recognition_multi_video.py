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
# Day21 recognized instruments (6 classes: grasper, bipolar,
# hook, scissors, clipper, irrigator) from raw frames across
# 10 videos, split at the video level. Instrument is the
# easiest slice of the instrument-verb-target triplet:
# instruments are large, visually distinct metal shapes.
# Today moves to the next slice -- verb (10 classes: grasp,
# retract, dissect, coagulate, clip, cut, aspirate, irrigate,
# pack, null_verb) -- the action being performed, which is a
# harder visual signal (motion/context-dependent, not just a
# static shape) using the exact same pipeline (same 10
# videos, same video-level 8/2 split, same frozen ResNet18 +
# linear head, same evaluation) as Day21, so any difference
# in results is attributable to the task, not the setup.
#
# null_verb (an instrument on screen but not actively doing
# anything) is included as one of the 10 classes, exactly as
# it's defined in CholecT50 -- not filtered out here, unlike
# Day12-19's state-segmentation code, which filtered it when
# building triplet-states. There, filtering null_verb made
# sense because those days modeled active sub-tasks; here,
# the model is asked to recognize whatever verb category is
# annotated, including "idle."
# ----------------------------------------

DATASET_ROOT = Path("/Users/katsutoshimakino/Datasets/CholecT50/CholecT50")
VIDEOS_DIR = DATASET_ROOT / "videos"
LABELS_DIR = DATASET_ROOT / "labels"

VIDEO_IDS = [
    "VID01", "VID02", "VID04", "VID05", "VID06",
    "VID08", "VID10", "VID12", "VID13", "VID14",
]

NUM_VERBS = 10
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
# Build per-frame verb multi-hot labels, for every video, and
# record (video_id, frame_id) pairs so frames from different
# videos can be told apart.
# ----------------------------------------

verb_names = None
frame_labels = {}   # (video_id, frame) -> 10-d multi-hot label
video_frame_ids = {}  # video_id -> sorted list of frame ids

for video_id in VIDEO_IDS:

    with open(LABELS_DIR / f"{video_id}.json") as f:
        data = json.load(f)

    if verb_names is None:
        verb_names = [
            data["categories"]["verb"][str(i)] for i in range(NUM_VERBS)
        ]

    frame_ids = sorted(int(f) for f in data["annotations"].keys())
    video_frame_ids[video_id] = frame_ids

    for frame in frame_ids:
        verbs_present = set()
        for triplet in data["annotations"][str(frame)]:
            verb_id = triplet[7]
            if verb_id != -1:
                verbs_present.add(verb_id)
        label = np.zeros(NUM_VERBS, dtype=np.float32)
        for verb_id in verbs_present:
            label[verb_id] = 1.0
        frame_labels[(video_id, frame)] = label

# ----------------------------------------
# Video-level train/test split (identical convention and
# fixed seed to Day21, so the same 8/2 video split applies).
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
# Dataset
# ----------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class VerbDataset(Dataset):

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        video_id, frame = self.pairs[idx]
        image_path = VIDEOS_DIR / video_id / f"{frame:06d}.png"
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        label = torch.from_numpy(frame_labels[(video_id, frame)])
        return image, label


train_dataset = VerbDataset(train_pairs)
test_dataset = VerbDataset(test_pairs)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# ----------------------------------------
# Model: same frozen ImageNet-pretrained ResNet18 + linear
# head as Day20/21, just with a 10-way output instead of 6.
# ----------------------------------------

backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in backbone.parameters():
    param.requires_grad = False

num_features = backbone.fc.in_features
backbone.fc = nn.Linear(num_features, NUM_VERBS)
model = backbone.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# ----------------------------------------
# Train
# ----------------------------------------

loss_history = []

for epoch in range(NUM_EPOCHS):

    model.train()
    epoch_loss = 0.0
    num_batches = 0

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
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: train loss = {avg_loss:.4f}")

# ----------------------------------------
# Evaluate: per-verb accuracy and F1 on held-out
# (unseen-patient) test videos, vs. a baseline that always
# predicts each verb's train-majority label.
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

train_labels_array = np.stack([
    frame_labels[(v, f)] for (v, f) in train_pairs
])
train_prevalence = train_labels_array.mean(axis=0)
baseline_prediction = (train_prevalence > 0.5).astype(np.float32)
baseline_predictions = np.tile(baseline_prediction, (len(test_pairs), 1))

test_labels_array = np.stack([
    frame_labels[(v, f)] for (v, f) in test_pairs
])

print()
print(f"{'Verb':12s} {'Accuracy':>10s} {'F1':>8s} "
      f"{'Baseline Acc':>14s} {'Train Prev':>12s} {'Test Prev':>10s}")

results = {"verbs": {}}

for i, name in enumerate(verb_names):

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
    test_prevalence = test_labels_array[:, i].mean()

    print(f"{name:12s} {accuracy:10.3f} {f1:8.3f} "
          f"{baseline_accuracy:14.3f} {train_prevalence[i]:12.3f} "
          f"{test_prevalence:10.3f}")

    results["verbs"][name] = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "baseline_accuracy": float(baseline_accuracy),
        "train_prevalence": float(train_prevalence[i]),
        "test_prevalence": float(test_prevalence),
    }

macro_accuracy = np.mean([
    results["verbs"][n]["accuracy"] for n in verb_names
])
macro_f1 = np.mean([
    results["verbs"][n]["f1"] for n in verb_names
])
macro_baseline_accuracy = np.mean([
    results["verbs"][n]["baseline_accuracy"] for n in verb_names
])

print()
print(f"Macro accuracy: {macro_accuracy:.3f} "
      f"(baseline: {macro_baseline_accuracy:.3f})")
print(f"Macro F1:       {macro_f1:.3f}")

results["macro_accuracy"] = float(macro_accuracy)
results["macro_f1"] = float(macro_f1)
results["macro_baseline_accuracy"] = float(macro_baseline_accuracy)
results["loss_history"] = loss_history
results["train_video_ids"] = train_video_ids
results["test_video_ids"] = test_video_ids
results["num_train_frames"] = len(train_pairs)
results["num_test_frames"] = len(test_pairs)

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
