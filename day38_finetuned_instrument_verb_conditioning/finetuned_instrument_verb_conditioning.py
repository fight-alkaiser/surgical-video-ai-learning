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
# Day37 tested whether giving the verb probe access to
# instrument identity closes verb recognition's gap: oracle
# conditioning (ground-truth instrument) raised verb macro F1
# from 0.309 to 0.484, but realistic conditioning (predicted
# instrument from a frozen-feature probe, itself only macro F1
# 0.399) recovered none of it (0.305) -- the predictor was too
# noisy to help.
#
# Day27 already showed a much more accurate instrument
# recognizer exists: fine-tuning ResNet18's last residual block
# (layer4) raised instrument macro F1 from 0.378 (frozen,
# class-weighted) to 0.512 -- but that day didn't save the
# checkpoint. Today re-runs Day27's exact recipe (same seed, same
# split -- reproducible), saves the checkpoint this time, and
# plugs its predictions into Day37's realistic-conditioning setup
# in place of the weaker frozen-feature instrument probe. One
# variable changes relative to Day37's condition C: instrument
# prediction accuracy. Everything else -- verb probe recipe,
# frozen-ImageNet verb features, split -- stays identical.
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
FINETUNE_BATCH_SIZE = 16
FINETUNE_EPOCHS = 8
FINETUNE_LR_HEAD = 1e-3
FINETUNE_LR_BACKBONE = 1e-4
PROBE_BATCH_SIZE = 32
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
# (identical extraction logic to Day21/22/26/27/37).
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
# Video-level train/test split -- identical to every prior day
# (same seed, same VIDEO_IDS order, same shuffle call).
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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ----------------------------------------
# Part 1: fine-tune ResNet18 (layer4 + fc) for instrument
# recognition -- Day27's exact recipe, checkpoint saved this
# time.
# ----------------------------------------


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


finetune_train_loader = DataLoader(
    InstrumentDataset(train_pairs), batch_size=FINETUNE_BATCH_SIZE,
    shuffle=True, num_workers=0
)
finetune_test_loader = DataLoader(
    InstrumentDataset(test_pairs), batch_size=FINETUNE_BATCH_SIZE,
    shuffle=False, num_workers=0
)

num_positive = train_instrument_targets.sum(axis=0)
num_negative = len(train_pairs) - num_positive
pos_weight = (num_negative / np.maximum(num_positive, 1)).astype(np.float32)
pos_weight_t = torch.from_numpy(pos_weight).to(device)

finetune_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

frozen_modules = [
    finetune_model.conv1, finetune_model.bn1,
    finetune_model.layer1, finetune_model.layer2, finetune_model.layer3,
]
for module in frozen_modules:
    for param in module.parameters():
        param.requires_grad = False

num_features = finetune_model.fc.in_features
finetune_model.fc = nn.Linear(num_features, NUM_INSTRUMENTS)
finetune_model = finetune_model.to(device)


def set_training_mode():
    finetune_model.train()
    for module in frozen_modules:
        module.eval()


criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

optimizer = torch.optim.Adam([
    {"params": finetune_model.layer4.parameters(), "lr": FINETUNE_LR_BACKBONE},
    {"params": finetune_model.fc.parameters(), "lr": FINETUNE_LR_HEAD},
])

print("\n=== Fine-tuning ResNet18 layer4 for instrument recognition (Day27 recipe) ===")
loss_history = []
finetune_start = time.time()

for epoch in range(FINETUNE_EPOCHS):

    set_training_mode()
    epoch_loss = 0.0
    num_batches = 0
    epoch_start = time.time()

    for images, labels in finetune_train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = finetune_model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}/{FINETUNE_EPOCHS}: train loss = {avg_loss:.4f} "
          f"({time.time() - epoch_start:.0f}s)")

print(f"Fine-tuning total time: {time.time() - finetune_start:.0f}s")

checkpoint_path = Path(__file__).parent / "finetuned_instrument_backbone.pt"
torch.save(finetune_model.state_dict(), checkpoint_path)
print(f"Saved checkpoint to {checkpoint_path}")

# ----------------------------------------
# Evaluate the fine-tuned instrument classifier on the test
# set (sanity check vs. Day27's macro F1 0.512), and get its
# predicted probabilities for BOTH train and test frames -- the
# signal that will condition the verb probe.
# ----------------------------------------

finetune_model.eval()


def finetuned_instrument_predictions(loader):
    all_probs = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logits = finetune_model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs)


print("\n=== Extracting fine-tuned instrument predictions (train + test) ===")
pred_start = time.time()
train_instrument_pred_finetuned = finetuned_instrument_predictions(finetune_train_loader)
test_instrument_pred_finetuned = finetuned_instrument_predictions(finetune_test_loader)
print(f"  Done in {time.time() - pred_start:.0f}s")

test_instrument_pred_binary = (test_instrument_pred_finetuned > 0.5).astype(np.float32)
instrument_per_class = {}
for i, name in enumerate(instrument_names):
    pred_i = test_instrument_pred_binary[:, i]
    label_i = test_instrument_targets[:, i]
    tp = ((pred_i == 1) & (label_i == 1)).sum()
    fp = ((pred_i == 1) & (label_i == 0)).sum()
    fn = ((pred_i == 0) & (label_i == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    instrument_per_class[name] = {
        "f1": float(f1), "precision": float(precision), "recall": float(recall),
    }

instrument_macro_f1 = float(np.mean([instrument_per_class[n]["f1"] for n in instrument_names]))
print(f"\nFine-tuned instrument macro F1 (sanity check vs. Day27's 0.512): {instrument_macro_f1:.3f}")

# ----------------------------------------
# Part 2: frozen ImageNet features for the verb probe -- exact
# same recipe as Day37, so condition A here should reproduce
# Day37's 0.309.
# ----------------------------------------


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
        FrameDataset(pairs), batch_size=PROBE_BATCH_SIZE, shuffle=False,
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


imagenet_backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
imagenet_backbone.fc = nn.Identity()
for param in imagenet_backbone.parameters():
    param.requires_grad = False
imagenet_backbone = imagenet_backbone.to(device)
imagenet_backbone.eval()
feature_dim = 512

print("\n=== Extracting frozen ImageNet features for verb probe ===")
train_features = extract_features(imagenet_backbone, train_pairs, "train")
test_features = extract_features(imagenet_backbone, test_pairs, "test")

# ----------------------------------------
# Class-weighted linear probe (Day26/37's recipe), generic over
# input dimensionality.
# ----------------------------------------


def train_probe(train_x, train_y, num_classes, input_dim):

    num_pos = train_y.sum(axis=0)
    num_neg = len(train_y) - num_pos
    pw = (num_neg / np.maximum(num_pos, 1)).astype(np.float32)
    pw_t = torch.from_numpy(pw).to(device)

    probe = nn.Linear(input_dim, num_classes).to(device)
    criterion_probe = nn.BCEWithLogitsLoss(pos_weight=pw_t)
    optimizer_probe = torch.optim.Adam(probe.parameters(), lr=PROBE_LEARNING_RATE)

    train_x_t = torch.from_numpy(train_x).float()
    train_y_t = torch.from_numpy(train_y).float()
    num_train = train_x_t.shape[0]

    for epoch in range(PROBE_EPOCHS):
        permutation = torch.randperm(num_train)
        for start_idx in range(0, num_train, PROBE_BATCH_SIZE):
            idx = permutation[start_idx:start_idx + PROBE_BATCH_SIZE]
            batch_x = train_x_t[idx].to(device)
            batch_y = train_y_t[idx].to(device)
            optimizer_probe.zero_grad()
            loss = criterion_probe(probe(batch_x), batch_y)
            loss.backward()
            optimizer_probe.step()

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


results = {
    "finetuned_instrument_sanity_check": {
        "macro_f1": instrument_macro_f1, "per_class": instrument_per_class
    },
    "verb": {},
}

# ----------------------------------------
# Condition A: baseline -- features only (should reproduce
# Day37's 0.309).
# ----------------------------------------

print("\n=== Condition A: baseline (features only) ===")
probe_a = train_probe(train_features, train_verb_targets, NUM_VERBS, feature_dim)
macro_f1_a, per_class_a = evaluate_probe(probe_a, test_features, test_verb_targets, verb_names)
print(f"  Verb macro F1: {macro_f1_a:.3f}")
results["verb"]["A_baseline"] = {"macro_f1": macro_f1_a, "per_class": per_class_a}

# ----------------------------------------
# Condition D: features + predicted instrument probabilities
# from the FINE-TUNED instrument classifier (macro F1 ~0.512),
# replacing Day37's weaker frozen-feature instrument probe
# (macro F1 0.399) that produced Day37's condition C (0.305).
# ----------------------------------------

print("\n=== Condition D: features + fine-tuned instrument predictions ===")
train_features_d = np.concatenate([train_features, train_instrument_pred_finetuned], axis=1)
test_features_d = np.concatenate([test_features, test_instrument_pred_finetuned], axis=1)
probe_d = train_probe(
    train_features_d, train_verb_targets, NUM_VERBS, feature_dim + NUM_INSTRUMENTS
)
macro_f1_d, per_class_d = evaluate_probe(
    probe_d, test_features_d, test_verb_targets, verb_names
)
print(f"  Verb macro F1: {macro_f1_d:.3f}")
results["verb"]["D_finetuned_realistic"] = {"macro_f1": macro_f1_d, "per_class": per_class_d}

# ----------------------------------------
# Summary
# ----------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n{'Condition':50s} {'Verb macro F1':>15s}")
print(f"{'A: baseline (features only)':50s} {macro_f1_a:15.3f}")
print(f"{'D: + fine-tuned instrument pred (macro F1 ' + f'{instrument_macro_f1:.3f})':50s} {macro_f1_d:15.3f}")
print("\nFor reference (Day37, different instrument predictor):")
print("  B: oracle (+ ground-truth instrument):                 0.484")
print("  C: realistic (+ frozen-probe instrument, F1 0.399):    0.305")

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
