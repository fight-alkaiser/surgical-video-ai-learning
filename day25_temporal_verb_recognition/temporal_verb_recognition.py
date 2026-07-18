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
# Day22-24 all predicted verb from a SINGLE frame. Day22
# proposed, and Day23/24 partly confirmed, that grasp vs.
# retract vs. null_verb (all "grasper touching tissue") may
# be a genuine single-frame information limit: these can look
# nearly identical in one still image, differing mainly in
# force, direction, or what happens over the next few
# frames -- none of which one frame contains. Day23's oracle
# instrument-conditioning experiment barely moved grasp's F1
# (0.434 -> 0.467), consistent with this: even knowing the
# instrument perfectly doesn't resolve the ambiguity, because
# the missing information isn't "which instrument" but
# "what happens over time."
#
# Today tests that directly: instead of a single frame's
# ResNet18 features, a small GRU is given the past W=8 frames'
# features (about 8 seconds of context, since CholecT50 frames
# are extracted at 1fps) and predicts verb for the current
# frame from the GRU's final hidden state. This is the same
# "recurrence over a sequence" mechanism as Day17's from-
# scratch RNN, now implemented with PyTorch (matching this
# project's pixel-based days, Day20 onward) and applied to
# real per-frame visual features instead of symbolic
# triplet-states -- the two halves of this project (Day16-19's
# sequence modeling, Day20-24's pixel recognition) converging
# on the same question from opposite directions.
#
# Everything else -- videos, split, verb label extraction,
# frozen ResNet18 backbone -- is identical to Day22, so any
# difference in results is attributable to adding temporal
# context, not to a different setup.
# ----------------------------------------

DATASET_ROOT = Path("/Users/katsutoshimakino/Datasets/CholecT50/CholecT50")
VIDEOS_DIR = DATASET_ROOT / "videos"
LABELS_DIR = DATASET_ROOT / "labels"

VIDEO_IDS = [
    "VID01", "VID02", "VID04", "VID05", "VID06",
    "VID08", "VID10", "VID12", "VID13", "VID14",
]

NUM_VERBS = 10
WINDOW_SIZE = 8
TEST_RATIO = 0.2
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# ----------------------------------------
# Build per-frame verb multi-hot labels (identical to Day22).
# ----------------------------------------

verb_names = None
verb_labels = {}
video_frame_ids = {}

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
        verb_labels[(video_id, frame)] = label

# ----------------------------------------
# Video-level train/test split (identical to Day21-24).
# ----------------------------------------

shuffled_video_ids = VIDEO_IDS[:]
random.shuffle(shuffled_video_ids)

num_test_videos = max(1, round(len(VIDEO_IDS) * TEST_RATIO))
test_video_ids = sorted(shuffled_video_ids[:num_test_videos])
train_video_ids = sorted(shuffled_video_ids[num_test_videos:])

print(f"Train videos ({len(train_video_ids)}): {train_video_ids}")
print(f"Test videos  ({len(test_video_ids)}): {test_video_ids}")

# ----------------------------------------
# Step 1: extract and cache frozen ResNet18 features for
# every frame of every video (same backbone as Day20-24).
# ----------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FrameDataset(Dataset):

    def __init__(self, video_id, frame_ids):
        self.video_id = video_id
        self.frame_ids = frame_ids

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame = self.frame_ids[idx]
        image_path = VIDEOS_DIR / self.video_id / f"{frame:06d}.png"
        image = Image.open(image_path).convert("RGB")
        return transform(image)


backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in backbone.parameters():
    param.requires_grad = False
num_features = backbone.fc.in_features
backbone.fc = nn.Identity()
backbone = backbone.to(device)
backbone.eval()

video_features = {}  # video_id -> (num_frames, 512) array, in frame order

start = time.time()
for video_id in VIDEO_IDS:
    frame_ids = video_frame_ids[video_id]
    loader = DataLoader(
        FrameDataset(video_id, frame_ids), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    feats = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            feats.append(backbone(images).cpu())
    video_features[video_id] = torch.cat(feats).numpy()
    print(f"  {video_id}: {video_features[video_id].shape} "
          f"({time.time() - start:.0f}s elapsed)")

print(f"Feature extraction done in {time.time() - start:.1f}s")

# ----------------------------------------
# Step 2: build (sequence, target) pairs. For frame index t
# within a video (0-indexed into that video's sorted frame
# list), the input sequence is the past WINDOW_SIZE frames'
# features (t-WINDOW_SIZE+1 .. t), and the target is verb at
# frame t. Only positions with a full window available are
# used (t >= WINDOW_SIZE - 1).
# ----------------------------------------


def build_sequences(video_ids):
    sequences = []
    targets = []
    for video_id in video_ids:
        frame_ids = video_frame_ids[video_id]
        feats = video_features[video_id]
        for t in range(WINDOW_SIZE - 1, len(frame_ids)):
            window = feats[t - WINDOW_SIZE + 1:t + 1]
            sequences.append(window)
            targets.append(verb_labels[(video_id, frame_ids[t])])
    return np.stack(sequences), np.stack(targets)


train_sequences, train_targets = build_sequences(train_video_ids)
test_sequences, test_targets = build_sequences(test_video_ids)

print(f"Train sequences: {train_sequences.shape}, "
      f"Test sequences: {test_sequences.shape}")

train_sequences_t = torch.from_numpy(train_sequences).float()
train_targets_t = torch.from_numpy(train_targets).float()
test_sequences_t = torch.from_numpy(test_sequences).float()
test_targets_t = torch.from_numpy(test_targets).float()

# ----------------------------------------
# Step 3: a small GRU over the cached feature sequence,
# followed by a linear head on its final hidden state.
# ----------------------------------------


class TemporalVerbModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_verbs):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_verbs)

    def forward(self, sequences):
        _, final_hidden = self.gru(sequences)
        return self.head(final_hidden.squeeze(0))


model = TemporalVerbModel(num_features, HIDDEN_DIM, NUM_VERBS).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------------------
# Train
# ----------------------------------------

num_train = train_sequences_t.shape[0]
loss_history = []

for epoch in range(NUM_EPOCHS):

    model.train()
    permutation = torch.randperm(num_train)
    epoch_loss = 0.0
    num_batches = 0

    for start_idx in range(0, num_train, BATCH_SIZE):
        idx = permutation[start_idx:start_idx + BATCH_SIZE]
        batch_x = train_sequences_t[idx].to(device)
        batch_y = train_targets_t[idx].to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: train loss = {avg_loss:.4f}")

# ----------------------------------------
# Evaluate: per-verb accuracy and F1 on held-out test videos,
# vs. the same train-majority baseline as Day22, for direct
# comparison.
# ----------------------------------------

model.eval()

all_predictions = []

with torch.no_grad():
    for start_idx in range(0, test_sequences_t.shape[0], BATCH_SIZE):
        batch_x = test_sequences_t[start_idx:start_idx + BATCH_SIZE].to(device)
        logits = model(batch_x)
        predictions = (torch.sigmoid(logits) > 0.5).float().cpu()
        all_predictions.append(predictions)

all_predictions = torch.cat(all_predictions).numpy()
all_labels = test_targets

train_prevalence = train_targets.mean(axis=0)
baseline_prediction = (train_prevalence > 0.5).astype(np.float32)
baseline_predictions = np.tile(baseline_prediction, (len(test_targets), 1))

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
    test_prevalence = label_i.mean()

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

macro_accuracy = np.mean([results["verbs"][n]["accuracy"] for n in verb_names])
macro_f1 = np.mean([results["verbs"][n]["f1"] for n in verb_names])
macro_baseline_accuracy = np.mean(
    [results["verbs"][n]["baseline_accuracy"] for n in verb_names]
)

print()
print(f"Macro accuracy: {macro_accuracy:.3f} "
      f"(baseline: {macro_baseline_accuracy:.3f})")
print(f"Macro F1:       {macro_f1:.3f}")
print()
print("For reference, Day22 (single frame, no temporal context): "
      "macro accuracy 0.888, macro F1 0.192")

results["macro_accuracy"] = float(macro_accuracy)
results["macro_f1"] = float(macro_f1)
results["macro_baseline_accuracy"] = float(macro_baseline_accuracy)
results["loss_history"] = loss_history
results["window_size"] = WINDOW_SIZE
results["hidden_dim"] = HIDDEN_DIM
results["train_video_ids"] = train_video_ids
results["test_video_ids"] = test_video_ids
results["num_train_sequences"] = int(num_train)
results["num_test_sequences"] = int(test_sequences_t.shape[0])

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
