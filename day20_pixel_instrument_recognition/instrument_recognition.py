import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ----------------------------------------
# Day20 starts a new arc. Day01-19 always started from
# CholecT50's human-provided triplet annotations -- a
# symbolic sequence handed to us, never something extracted
# from pixels. The dataset's actual task (and the Rendezvous
# paper -- Nwoye et al., 2022 -- that introduced CholecT50)
# is the other half of the problem: recognizing
# instrument-verb-target triplets directly from raw
# endoscopic video frames.
#
# This is also the first day using PyTorch instead of a
# from-scratch numpy implementation. That's a deliberate
# change, not a relaxation of standards: CNN backprop
# (convolution/pooling gradients) is not a core mechanism
# this project is trying to internalize the way embedding
# lookups, BPTT, and attention were (Day16-19) -- it's a
# well-understood, separate piece of engineering, and
# training one from scratch on real images without a GPU-
# accelerated autograd engine is not practical within a
# day's scope. The Rendezvous paper itself uses an
# ImageNet-pretrained CNN backbone and reserves its actual
# novelty for the attention modules built on top -- so using
# a library for the backbone matches the paper's own design,
# not just convenience.
#
# Today's task is deliberately the simplest slice of the
# full triplet-recognition problem: multi-label instrument
# recognition (6 classes: grasper, bipolar, hook, scissors,
# clipper, irrigator) from a single frame. Verb and target
# recognition, and the interaction-attention modules that
# connect them, are left for later days.
#
# Only VID01 has raw frames downloaded locally (1734 frames,
# ~864MB) -- the other 49 CholecT50 videos are annotation-
# only in this project so far. This is a single-video
# prototype, not a claim of generalization across patients.
# ----------------------------------------

VIDEO_DIR = Path(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/videos/VID01"
)
LABELS_PATH = Path(
    "/Users/katsutoshimakino/Datasets/CholecT50/CholecT50/labels/VID01.json"
)

TRAIN_RATIO = 0.8
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# ----------------------------------------
# Build per-frame instrument multi-hot labels
# ----------------------------------------

with open(LABELS_PATH) as f:
    data = json.load(f)

instrument_names = [
    data["categories"]["instrument"][str(i)] for i in range(6)
]

frame_ids = sorted(int(f) for f in data["annotations"].keys())

frame_labels = {}
for frame in frame_ids:
    instruments_present = set()
    for triplet in data["annotations"][str(frame)]:
        instrument_id = triplet[1]
        if instrument_id != -1:
            instruments_present.add(instrument_id)
    label = np.zeros(6, dtype=np.float32)
    for instrument_id in instruments_present:
        label[instrument_id] = 1.0
    frame_labels[frame] = label

# ----------------------------------------
# Chronological train/test split.
#
# Adjacent frames in a surgical video are nearly identical
# (the camera and scene barely change frame to frame), so a
# random frame-level split would put near-duplicate frames
# in both train and test -- an easy, misleading way to get a
# falsely high accuracy. Splitting by time (first 80% of the
# video for training, last 20% for testing) avoids that
# specific leak, though it's still a much weaker test than a
# held-out patient: the test segment is still the same
# operation, same patient, same lighting/camera.
# ----------------------------------------

split_index = int(len(frame_ids) * TRAIN_RATIO)
train_frame_ids = frame_ids[:split_index]
test_frame_ids = frame_ids[split_index:]

print(f"Total frames: {len(frame_ids)}")
print(f"Train frames: {len(train_frame_ids)} (frames "
      f"{train_frame_ids[0]}-{train_frame_ids[-1]})")
print(f"Test frames:  {len(test_frame_ids)} (frames "
      f"{test_frame_ids[0]}-{test_frame_ids[-1]})")

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

    def __init__(self, frame_id_list):
        self.frame_id_list = frame_id_list

    def __len__(self):
        return len(self.frame_id_list)

    def __getitem__(self, idx):
        frame = self.frame_id_list[idx]
        image_path = VIDEO_DIR / f"{frame:06d}.png"
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        label = torch.from_numpy(frame_labels[frame])
        return image, label


train_dataset = InstrumentDataset(train_frame_ids)
test_dataset = InstrumentDataset(test_frame_ids)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------------------
# Model: frozen ImageNet-pretrained ResNet18 backbone + a
# single trainable linear layer on top -- the same "freeze a
# pretrained feature extractor, train a linear head" pattern
# as the linear probes in Day17-19, just applied to a visual
# backbone instead of a sequence model's hidden state.
# ----------------------------------------

backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in backbone.parameters():
    param.requires_grad = False

num_features = backbone.fc.in_features
backbone.fc = nn.Linear(num_features, 6)
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
# Evaluate: per-instrument accuracy and macro F1 on the
# held-out (later-in-time) test frames, vs. a baseline that
# always predicts each instrument's majority label from the
# training set (present if it appeared in >50% of train
# frames, absent otherwise).
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

train_labels_array = np.stack([frame_labels[f] for f in train_frame_ids])
train_prevalence = train_labels_array.mean(axis=0)
baseline_prediction = (train_prevalence > 0.5).astype(np.float32)
baseline_predictions = np.tile(baseline_prediction, (len(test_frame_ids), 1))

print()
print(f"{'Instrument':12s} {'Accuracy':>10s} {'F1':>8s} "
      f"{'Baseline Acc':>14s} {'Test Prevalence':>16s}")

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
          f"{baseline_accuracy:14.3f} {test_prevalence:16.3f}")

    results["instruments"][name] = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "baseline_accuracy": float(baseline_accuracy),
        "test_prevalence": float(test_prevalence),
    }

macro_accuracy = np.mean([
    results["instruments"][n]["accuracy"] for n in instrument_names
])
macro_f1 = np.mean([
    results["instruments"][n]["f1"] for n in instrument_names
])
macro_baseline_accuracy = np.mean([
    results["instruments"][n]["baseline_accuracy"] for n in instrument_names
])

print()
print(f"Macro accuracy: {macro_accuracy:.3f} "
      f"(baseline: {macro_baseline_accuracy:.3f})")
print(f"Macro F1:       {macro_f1:.3f}")

results["macro_accuracy"] = float(macro_accuracy)
results["macro_f1"] = float(macro_f1)
results["macro_baseline_accuracy"] = float(macro_baseline_accuracy)
results["loss_history"] = loss_history
results["num_train_frames"] = len(train_frame_ids)
results["num_test_frames"] = len(test_frame_ids)

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
