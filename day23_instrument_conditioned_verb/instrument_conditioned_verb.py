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
# Day22 found verb recognition (macro F1 0.192) markedly
# weaker than Day21's instrument recognition (0.302), and
# split the reason in two: grasp/retract/null_verb looks
# like a genuine single-frame ambiguity, but clip/cut/
# aspirate/irrigate looked like an architecture gap --
# checking instrument-verb co-occurrence directly showed
# these verbs are 79-95% determined by instrument identity
# alone (clipper -> clip 94.9%, scissors -> cut 91.0%,
# bipolar -> coagulate 78.7%, hook -> dissect 86.6%), yet
# Day22's verb classifier never saw instrument identity at
# all -- it was a fully independent linear head on the same
# generic image features, re-deriving redundant information
# from a handful of rare positive examples.
#
# Today tests that specific hypothesis directly: does
# conditioning verb prediction on instrument identity close
# most of the gap for tool-specific verbs? This is an
# *oracle* test -- the model is given the TRUE instrument
# multi-hot label (not Day21's imperfect predictions) as an
# extra input, concatenated to the frozen ResNet18 feature
# vector before the (newly trained) linear head. This
# isolates the question "if the model knew the instrument
# perfectly, would tool-specific verb prediction improve?"
# from "how good is Day21's instrument classifier?" -- the
# two are separate questions, and this is deliberately the
# easier, upper-bound one. A more realistic end-to-end
# pipeline (conditioning on Day21's *predicted* instrument
# probabilities instead of ground truth) is a natural next
# step, not done here.
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
# Build per-frame instrument AND verb multi-hot labels, for
# every video (identical extraction logic to Day21/Day22).
# ----------------------------------------

instrument_names = None
verb_names = None
instrument_labels = {}   # (video_id, frame) -> 6-d multi-hot
verb_labels = {}         # (video_id, frame) -> 10-d multi-hot
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
# Video-level train/test split (identical to Day21/Day22).
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
# Dataset: returns (image, true_instrument_label, verb_label)
# ----------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class InstrumentConditionedVerbDataset(Dataset):

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        video_id, frame = self.pairs[idx]
        image_path = VIDEOS_DIR / video_id / f"{frame:06d}.png"
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        instrument = torch.from_numpy(instrument_labels[(video_id, frame)])
        verb = torch.from_numpy(verb_labels[(video_id, frame)])
        return image, instrument, verb


train_dataset = InstrumentConditionedVerbDataset(train_pairs)
test_dataset = InstrumentConditionedVerbDataset(test_pairs)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# ----------------------------------------
# Model: frozen ImageNet-pretrained ResNet18 (same backbone
# as Day20-22) with its final layer removed, producing a
# 512-d feature vector. That vector is concatenated with the
# 6-d TRUE instrument multi-hot label, and only this new
# (512+6) -> 10 linear layer is trained -- everything else
# is identical to Day22's verb classifier.
# ----------------------------------------

backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in backbone.parameters():
    param.requires_grad = False
num_features = backbone.fc.in_features
backbone.fc = nn.Identity()
backbone = backbone.to(device)
backbone.eval()

verb_head = nn.Linear(num_features + NUM_INSTRUMENTS, NUM_VERBS).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(verb_head.parameters(), lr=LEARNING_RATE)

# ----------------------------------------
# Train
# ----------------------------------------

loss_history = []

for epoch in range(NUM_EPOCHS):

    verb_head.train()
    epoch_loss = 0.0
    num_batches = 0

    for images, instruments, verbs in train_loader:
        images = images.to(device)
        instruments = instruments.to(device)
        verbs = verbs.to(device)

        with torch.no_grad():
            features = backbone(images)

        combined = torch.cat([features, instruments], dim=1)

        optimizer.zero_grad()
        logits = verb_head(combined)
        loss = criterion(logits, verbs)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: train loss = {avg_loss:.4f}")

# ----------------------------------------
# Evaluate: per-verb accuracy and F1 on held-out test videos,
# using the TRUE instrument label at test time too (still an
# oracle test), vs. the same train-majority baseline as
# Day22 for direct comparison.
# ----------------------------------------

verb_head.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, instruments, verbs in test_loader:
        images = images.to(device)
        instruments = instruments.to(device)
        features = backbone(images)
        combined = torch.cat([features, instruments], dim=1)
        logits = verb_head(combined)
        predictions = (torch.sigmoid(logits) > 0.5).float().cpu()
        all_predictions.append(predictions)
        all_labels.append(verbs)

all_predictions = torch.cat(all_predictions).numpy()
all_labels = torch.cat(all_labels).numpy()

train_labels_array = np.stack([
    verb_labels[(v, f)] for (v, f) in train_pairs
])
train_prevalence = train_labels_array.mean(axis=0)
baseline_prediction = (train_prevalence > 0.5).astype(np.float32)
baseline_predictions = np.tile(baseline_prediction, (len(test_pairs), 1))

test_labels_array = np.stack([
    verb_labels[(v, f)] for (v, f) in test_pairs
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
print("For reference, Day22 (no instrument conditioning): "
      "macro accuracy 0.888, macro F1 0.192")

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
