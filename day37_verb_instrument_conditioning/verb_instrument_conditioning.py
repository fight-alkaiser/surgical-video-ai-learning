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
# Day22 diagnosed verb recognition's low ceiling (macro F1 0.192,
# later 0.309 with class-weighting alone per Day36) as split
# between two causes: a single-frame information limit (grasp vs.
# retract is often genuinely ambiguous in one still image) and an
# architecture gap -- the model was never given instrument
# identity, even though verb meaning is instrument-dependent (e.g.
# "cut" only makes sense for scissors, "clip" only for the
# clipper). Day31-36 showed SSL feature adaptation cannot fix
# verb, which is consistent with the bottleneck not being feature
# quality. Today tests the architecture-gap half of the diagnosis
# directly: does giving the verb probe access to instrument
# identity close any of the gap?
#
# Three conditions, same frozen ImageNet backbone throughout (SSL
# already shown not to matter for verb, so it is excluded here to
# keep this test isolated to one variable: instrument
# conditioning):
#   (A) baseline    -- 512-dim ResNet18 features only
#   (B) oracle      -- features + GROUND-TRUTH instrument one-hot
#                       (6-dim), an upper bound on how much
#                       instrument identity could help in principle
#   (C) realistic   -- features + PREDICTED instrument probabilities
#                       from a separately trained class-weighted
#                       instrument probe (Day26's recipe), a
#                       realistic pipeline where instrument itself
#                       must be inferred
#
# If neither (B) nor (C) improves over (A), the architecture-gap
# hypothesis is refuted and the single-frame information limit is
# the dominant explanation. If (B) improves but (C) does not, the
# gap is real in principle but not exploitable with this project's
# instrument-recognition accuracy.
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
# Build per-frame instrument and verb multi-hot labels (identical
# extraction logic to Day21/22/26).
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

train_instrument_targets = np.stack([instrument_labels[p] for p in train_pairs])
test_instrument_targets = np.stack([instrument_labels[p] for p in test_pairs])
train_verb_targets = np.stack([verb_labels[p] for p in train_pairs])
test_verb_targets = np.stack([verb_labels[p] for p in test_pairs])

# ----------------------------------------
# Feature extraction: plain frozen ImageNet ResNet18 only (SSL
# already ruled out for verb, per Day36's unifying conclusion).
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


def extract_features(backbone, pairs, label_name):
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


backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
backbone.fc = nn.Identity()
for param in backbone.parameters():
    param.requires_grad = False
backbone = backbone.to(device)
backbone.eval()
feature_dim = 512

print("\n=== Extracting frozen ImageNet features ===")
train_features = extract_features(backbone, train_pairs, "train")
test_features = extract_features(backbone, test_pairs, "test")

# ----------------------------------------
# Class-weighted linear probe (Day26's recipe), generic over
# input dimensionality so it can be reused for instrument and for
# each verb condition.
# ----------------------------------------


def train_probe(train_x, train_y, num_classes, input_dim):

    num_positive = train_y.sum(axis=0)
    num_negative = len(train_y) - num_positive
    pos_weight = (num_negative / np.maximum(num_positive, 1)).astype(np.float32)
    pos_weight_t = torch.from_numpy(pos_weight).to(device)

    probe = nn.Linear(input_dim, num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
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


def probe_predict(probe, x):
    x_t = torch.from_numpy(x).float()
    probe.eval()
    with torch.no_grad():
        logits = probe(x_t.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def evaluate_probe(probe, test_x, test_y, class_names):

    probs = probe_predict(probe, test_x)
    predictions = (probs > 0.5).astype(np.float32)

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


results = {"verb": {}}

# ----------------------------------------
# Condition A: baseline -- features only (should reproduce
# Day36's 0.309).
# ----------------------------------------

print("\n=== Condition A: baseline (features only) ===")
probe_a = train_probe(train_features, train_verb_targets, NUM_VERBS, feature_dim)
macro_f1_a, per_class_a = evaluate_probe(probe_a, test_features, test_verb_targets, verb_names)
print(f"  Verb macro F1: {macro_f1_a:.3f}")
results["verb"]["A_baseline"] = {"macro_f1": macro_f1_a, "per_class": per_class_a}

# ----------------------------------------
# Condition B: oracle -- features + ground-truth instrument
# one-hot.
# ----------------------------------------

print("\n=== Condition B: oracle (features + ground-truth instrument) ===")
train_features_oracle = np.concatenate([train_features, train_instrument_targets], axis=1)
test_features_oracle = np.concatenate([test_features, test_instrument_targets], axis=1)
probe_b = train_probe(
    train_features_oracle, train_verb_targets, NUM_VERBS, feature_dim + NUM_INSTRUMENTS
)
macro_f1_b, per_class_b = evaluate_probe(
    probe_b, test_features_oracle, test_verb_targets, verb_names
)
print(f"  Verb macro F1: {macro_f1_b:.3f}")
results["verb"]["B_oracle"] = {"macro_f1": macro_f1_b, "per_class": per_class_b}

# ----------------------------------------
# Condition C: realistic -- features + predicted instrument
# probabilities from a separately trained instrument probe.
# ----------------------------------------

print("\n=== Training instrument probe (for condition C) ===")
instrument_probe = train_probe(
    train_features, train_instrument_targets, NUM_INSTRUMENTS, feature_dim
)
instrument_macro_f1, instrument_per_class = evaluate_probe(
    instrument_probe, test_features, test_instrument_targets, instrument_names
)
print(f"  Instrument macro F1 (sanity check vs. Day26's 0.378): {instrument_macro_f1:.3f}")
results["instrument_probe_sanity_check"] = {
    "macro_f1": instrument_macro_f1, "per_class": instrument_per_class
}

train_instrument_pred = probe_predict(instrument_probe, train_features)
test_instrument_pred = probe_predict(instrument_probe, test_features)

print("\n=== Condition C: realistic (features + predicted instrument) ===")
train_features_realistic = np.concatenate([train_features, train_instrument_pred], axis=1)
test_features_realistic = np.concatenate([test_features, test_instrument_pred], axis=1)
probe_c = train_probe(
    train_features_realistic, train_verb_targets, NUM_VERBS, feature_dim + NUM_INSTRUMENTS
)
macro_f1_c, per_class_c = evaluate_probe(
    probe_c, test_features_realistic, test_verb_targets, verb_names
)
print(f"  Verb macro F1: {macro_f1_c:.3f}")
results["verb"]["C_realistic"] = {"macro_f1": macro_f1_c, "per_class": per_class_c}

# ----------------------------------------
# Summary
# ----------------------------------------

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n{'Condition':40s} {'Verb macro F1':>15s}")
print(f"{'A: baseline (features only)':40s} {macro_f1_a:15.3f}")
print(f"{'B: oracle (+ ground-truth instrument)':40s} {macro_f1_b:15.3f}")
print(f"{'C: realistic (+ predicted instrument)':40s} {macro_f1_c:15.3f}")

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
