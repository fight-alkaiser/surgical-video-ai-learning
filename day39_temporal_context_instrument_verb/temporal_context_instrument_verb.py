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
# Day38 found that a substantially more accurate instrument
# predictor (macro F1 0.512, vs. 0.399) recovered none of verb
# recognition's oracle-realistic gap -- verb macro F1 stayed at
# 0.305 either way. The offered explanation: verb difficulty and
# instrument-prediction difficulty likely share a frame-level
# cause (plausibly instrument-tip occlusion), and the owner
# proposed a specific, testable refinement -- temporal context
# might let a model recover instrument identity even through a
# momentarily occluded frame (by tracking the instrument across
# nearby frames), but verb depends on the tip's motion during
# the occluded moment itself, so temporal context might not help
# verb the same way.
#
# CholecT50 frames are sampled at a fixed 1-second interval
# (frame_id increments by 1 within a video), so a 3-frame window
# (t-1, t, t+1) spans roughly 2 seconds of real time -- coarse,
# but enough to test the asymmetry directly.
#
# Today builds a windowed linear-probe input: concatenated
# frozen-ImageNet features from frames t-1, t, t+1 (1536-dim),
# and compares macro F1 against the single-frame baseline (512-dim,
# t only), for BOTH instrument and verb, on the exact same subset
# of frames (excluding each video's first/last frame, which lack
# a full window) so any difference is attributable to added
# temporal context, not a different data subset.
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
PROBE_EPOCHS = 15
PROBE_LEARNING_RATE = 1e-3
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# ----------------------------------------
# Build per-frame instrument and verb multi-hot labels
# (identical extraction logic to Day21/22/26/37/38).
# ----------------------------------------

instrument_names = None
verb_names = None
instrument_labels = {}
verb_labels = {}
video_frame_ids = {}
video_frame_set = {}

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
    video_frame_set[video_id] = set(frame_ids)

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
# Video-level train/test split (identical to every prior day).
# ----------------------------------------

shuffled_video_ids = VIDEO_IDS[:]
random.shuffle(shuffled_video_ids)

num_test_videos = max(1, round(len(VIDEO_IDS) * TEST_RATIO))
test_video_ids = sorted(shuffled_video_ids[:num_test_videos])
train_video_ids = sorted(shuffled_video_ids[num_test_videos:])

print(f"Train videos ({len(train_video_ids)}): {train_video_ids}")
print(f"Test videos  ({len(test_video_ids)}): {test_video_ids}")

# ----------------------------------------
# Windowed pairs: a frame qualifies only if both its neighbors
# (t-1, t+1) exist in the same video. This drops each video's
# first and last frame. Neighbors always fall in the same
# train/test split as the center frame, since the split is by
# whole video.
# ----------------------------------------


def windowed_pairs(video_ids):
    pairs = []
    for v in video_ids:
        for f in video_frame_ids[v]:
            if (f - 1) in video_frame_set[v] and (f + 1) in video_frame_set[v]:
                pairs.append((v, f))
    return pairs


train_pairs = windowed_pairs(train_video_ids)
test_pairs = windowed_pairs(test_video_ids)

print(f"Train frames (windowed): {len(train_pairs)}, "
      f"Test frames (windowed): {len(test_pairs)}")

train_instrument_targets = np.stack([instrument_labels[p] for p in train_pairs])
test_instrument_targets = np.stack([instrument_labels[p] for p in test_pairs])
train_verb_targets = np.stack([verb_labels[p] for p in train_pairs])
test_verb_targets = np.stack([verb_labels[p] for p in test_pairs])

# ----------------------------------------
# Feature extraction: frozen ImageNet ResNet18, single frames.
# Extract every frame that appears in ANY window (center, t-1,
# or t+1) once, cache by (video, frame), then assemble
# single-frame and windowed inputs by lookup -- avoids
# re-extracting shared frames.
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
backbone.fc = nn.Identity()
for param in backbone.parameters():
    param.requires_grad = False
backbone = backbone.to(device)
backbone.eval()
feature_dim = 512


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
    print(f"  Extracted {label_name} features: {features.shape} "
          f"in {time.time() - start:.1f}s")
    return features


def needed_frames(pairs):
    needed = set()
    for v, f in pairs:
        needed.add((v, f - 1))
        needed.add((v, f))
        needed.add((v, f + 1))
    return sorted(needed)


print("\n=== Extracting frozen ImageNet features (all frames needed for windows) ===")
train_needed = needed_frames(train_pairs)
test_needed = needed_frames(test_pairs)
train_needed_features = extract_features(train_needed, "train")
test_needed_features = extract_features(test_needed, "test")

train_feature_lookup = {p: f for p, f in zip(train_needed, train_needed_features)}
test_feature_lookup = {p: f for p, f in zip(test_needed, test_needed_features)}


def build_single_frame_matrix(pairs, lookup):
    return np.stack([lookup[p] for p in pairs])


def build_windowed_matrix(pairs, lookup):
    rows = []
    for v, f in pairs:
        window = np.concatenate([lookup[(v, f - 1)], lookup[(v, f)], lookup[(v, f + 1)]])
        rows.append(window)
    return np.stack(rows)


train_single = build_single_frame_matrix(train_pairs, train_feature_lookup)
test_single = build_single_frame_matrix(test_pairs, test_feature_lookup)
train_windowed = build_windowed_matrix(train_pairs, train_feature_lookup)
test_windowed = build_windowed_matrix(test_pairs, test_feature_lookup)

# ----------------------------------------
# Class-weighted linear probe (Day26's recipe), generic over
# input dimensionality.
# ----------------------------------------


def train_probe(train_x, train_y, num_classes, input_dim):

    num_pos = train_y.sum(axis=0)
    num_neg = len(train_y) - num_pos
    pw = (num_neg / np.maximum(num_pos, 1)).astype(np.float32)
    pw_t = torch.from_numpy(pw).to(device)

    probe = nn.Linear(input_dim, num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_t)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LEARNING_RATE)

    train_x_t = torch.from_numpy(train_x).float()
    train_y_t = torch.from_numpy(train_y).float()
    num_train = train_x_t.shape[0]

    for epoch in range(PROBE_EPOCHS):
        permutation = torch.randperm(num_train)
        for start_idx in range(0, num_train, BATCH_SIZE):
            idx = permutation[start_idx:start_idx + BATCH_SIZE]
            batch_x = train_x_t[idx].to(device)
            batch_y = train_y_t[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(probe(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return probe


def evaluate_probe(probe, test_x, test_y, class_names):

    test_x_t = torch.from_numpy(test_x).float()
    probe.eval()
    with torch.no_grad():
        logits = probe(test_x_t.to(device))
        predictions = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()

    per_class = {}
    for i, name in enumerate(class_names):
        pred_i = predictions[:, i]
        label_i = test_y[:, i]
        tp = ((pred_i == 1) & (label_i == 1)).sum()
        fp = ((pred_i == 1) & (label_i == 0)).sum()
        fn = ((pred_i == 0) & (label_i == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[name] = {
            "f1": float(f1), "precision": float(precision), "recall": float(recall),
            "test_prevalence": float(label_i.mean()),
        }

    macro_f1 = float(np.mean([per_class[n]["f1"] for n in class_names]))
    return macro_f1, per_class


results = {"instrument": {}, "verb": {}}

for task_name, train_y, test_y, class_names, num_classes in [
    ("instrument", train_instrument_targets, test_instrument_targets, instrument_names, NUM_INSTRUMENTS),
    ("verb", train_verb_targets, test_verb_targets, verb_names, NUM_VERBS),
]:

    print(f"\n=== {task_name}: single-frame baseline ===")
    probe_single = train_probe(train_single, train_y, num_classes, feature_dim)
    macro_f1_single, per_class_single = evaluate_probe(probe_single, test_single, test_y, class_names)
    print(f"  {task_name} macro F1 (single-frame): {macro_f1_single:.3f}")

    print(f"=== {task_name}: 3-frame window (t-1, t, t+1) ===")
    probe_window = train_probe(train_windowed, train_y, num_classes, feature_dim * 3)
    macro_f1_window, per_class_window = evaluate_probe(probe_window, test_windowed, test_y, class_names)
    print(f"  {task_name} macro F1 (windowed): {macro_f1_window:.3f}")

    results[task_name]["single_frame"] = {"macro_f1": macro_f1_single, "per_class": per_class_single}
    results[task_name]["windowed"] = {"macro_f1": macro_f1_window, "per_class": per_class_window}

# ----------------------------------------
# Summary
# ----------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Task':12s} {'Single-frame':>14s} {'3-frame window':>16s} {'Diff':>8s}")
for task_name in ["instrument", "verb"]:
    s = results[task_name]["single_frame"]["macro_f1"]
    w = results[task_name]["windowed"]["macro_f1"]
    print(f"{task_name:12s} {s:14.3f} {w:16.3f} {w - s:8.3f}")

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
