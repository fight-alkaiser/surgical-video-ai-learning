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
# Day23 showed that conditioning verb prediction on the TRUE
# instrument label more than doubles macro F1 (0.192 -> 0.388),
# with each verb's improvement tracking its instrument-verb
# co-occurrence strength almost exactly. That was deliberately
# an oracle experiment -- ground-truth instrument labels are
# never available in a real pipeline, only Day21's imperfect
# predictions are. Today closes that gap: an actual instrument
# classifier is trained, its PREDICTED probabilities (not
# ground truth) condition the verb classifier, and the result
# is evaluated end-to-end -- this is the realistic version of
# Day23's question. Expected to land somewhere between Day22's
# no-conditioning floor (0.192) and Day23's oracle ceiling
# (0.388); how much closer to the ceiling is the open question.
#
# Implementation note: the frozen ResNet18 backbone is
# identical for both the instrument and verb classifiers, so
# its features are computed ONCE per frame and cached, instead
# of recomputing them for every training run. This turns two
# ~15-minute training runs (Day21/Day22-scale) into one ~2-3
# minute feature-extraction pass plus two near-instant linear
# classifier trainings on cached 512-d vectors.
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
# Build per-frame instrument AND verb multi-hot labels
# (identical extraction logic to Day21/22/23).
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
# Video-level train/test split (identical to Day21/22/23).
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
# Step 1: extract and cache frozen ResNet18 features once for
# every frame (train and test).
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

train_instrument_targets = np.stack(
    [instrument_labels[p] for p in train_pairs]
)
test_instrument_targets = np.stack(
    [instrument_labels[p] for p in test_pairs]
)
train_verb_targets = np.stack([verb_labels[p] for p in train_pairs])
test_verb_targets = np.stack([verb_labels[p] for p in test_pairs])

train_features_t = torch.from_numpy(train_features).float()
test_features_t = torch.from_numpy(test_features).float()
train_instrument_t = torch.from_numpy(train_instrument_targets).float()
test_instrument_t = torch.from_numpy(test_instrument_targets).float()
train_verb_t = torch.from_numpy(train_verb_targets).float()
test_verb_t = torch.from_numpy(test_verb_targets).float()

# ----------------------------------------
# Step 2: train the instrument classifier (Day21's model, but
# on cached features -- mathematically identical, much faster)
# ----------------------------------------


def train_linear_head(
    in_dim, out_dim, train_x, train_y, num_epochs, label
):
    head = nn.Linear(in_dim, out_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE)

    num_train = train_x.shape[0]
    loss_history = []

    for epoch in range(num_epochs):
        permutation = torch.randperm(num_train)
        epoch_loss = 0.0
        num_batches = 0

        for start_idx in range(0, num_train, BATCH_SIZE):
            idx = permutation[start_idx:start_idx + BATCH_SIZE]
            batch_x = train_x[idx].to(device)
            batch_y = train_y[idx].to(device)

            optimizer.zero_grad()
            logits = head(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        print(f"[{label}] Epoch {epoch + 1}/{num_epochs}: "
              f"train loss = {avg_loss:.4f}")

    return head, loss_history


print("\n--- Training instrument classifier ---")
instrument_head, instrument_loss_history = train_linear_head(
    num_features, NUM_INSTRUMENTS,
    train_features_t, train_instrument_t,
    NUM_EPOCHS, "instrument"
)

# ----------------------------------------
# Step 3: get PREDICTED instrument probabilities (not ground
# truth) for both train and test frames, from the classifier
# just trained. Train-side predictions condition the verb
# classifier's training inputs; test-side predictions condition
# its evaluation inputs -- both are the model's own (imperfect)
# outputs, never the true label.
# ----------------------------------------

instrument_head.eval()
with torch.no_grad():
    train_instrument_probs = torch.sigmoid(
        instrument_head(train_features_t.to(device))
    ).cpu()
    test_instrument_probs = torch.sigmoid(
        instrument_head(test_features_t.to(device))
    ).cpu()

# Quick sanity check: how good is the instrument classifier
# whose predictions will condition the verb classifier?
test_instrument_preds_binary = (test_instrument_probs > 0.5).float().numpy()
instrument_test_accuracy = (
    test_instrument_preds_binary == test_instrument_targets
).mean()
print(f"\nInstrument classifier mean per-class test accuracy: "
      f"{instrument_test_accuracy:.3f} (Day21 reference: 0.894)")

# ----------------------------------------
# Step 4: train the verb classifier on [cached features +
# PREDICTED instrument probabilities], evaluate the same way.
# ----------------------------------------

train_combined = torch.cat([train_features_t, train_instrument_probs], dim=1)
test_combined = torch.cat([test_features_t, test_instrument_probs], dim=1)

print("\n--- Training verb classifier (predicted-instrument-conditioned) ---")
verb_head, verb_loss_history = train_linear_head(
    num_features + NUM_INSTRUMENTS, NUM_VERBS,
    train_combined, train_verb_t,
    NUM_EPOCHS, "verb"
)

verb_head.eval()
with torch.no_grad():
    test_logits = verb_head(test_combined.to(device))
    test_predictions = (torch.sigmoid(test_logits) > 0.5).float().cpu().numpy()

all_predictions = test_predictions
all_labels = test_verb_targets

train_prevalence = train_verb_targets.mean(axis=0)
baseline_prediction = (train_prevalence > 0.5).astype(np.float32)
baseline_predictions = np.tile(baseline_prediction, (len(test_pairs), 1))

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
print("For reference:")
print("  Day22 (no instrument conditioning):        macro F1 0.192")
print("  Day23 (oracle true-instrument conditioning): macro F1 0.388")

results["macro_accuracy"] = float(macro_accuracy)
results["macro_f1"] = float(macro_f1)
results["macro_baseline_accuracy"] = float(macro_baseline_accuracy)
results["instrument_test_accuracy"] = float(instrument_test_accuracy)
results["instrument_loss_history"] = instrument_loss_history
results["verb_loss_history"] = verb_loss_history
results["train_video_ids"] = train_video_ids
results["test_video_ids"] = test_video_ids
results["num_train_frames"] = len(train_pairs)
results["num_test_frames"] = len(test_pairs)

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
