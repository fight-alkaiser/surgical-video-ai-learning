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
# Same evaluation as Day31's linear_probe_evaluation.py,
# applied to Day33's larger-batch (N=64, 128 views) backbone
# instead of Day31's (N=32, 64 views). Same 10 videos, same
# video-level split, same class-weighted probe recipe (Day26),
# so the only thing that differs from Day31's number is the
# pretraining batch size.
#
# Reference points, same 10 videos / video-level 8-2 split:
#   Day21 (ImageNet frozen, no adaptation at all):             F1 0.302
#   Day27 (ImageNet + supervised layer4 fine-tuning):          F1 0.512
#   Day31 (contrastive, N=32/64 views):                        F1 0.407
#   Day33 (contrastive, N=64/128 views, this script):          ?
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

backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = backbone.fc.in_features
backbone.fc = nn.Identity()

state_dict_path = Path(__file__).parent / "contrastive_backbone_large_batch.pt"
backbone.load_state_dict(torch.load(state_dict_path, map_location="cpu"))

for param in backbone.parameters():
    param.requires_grad = False
backbone = backbone.to(device)
backbone.eval()

print(f"\nLoaded large-batch contrastively-pretrained backbone from "
      f"{state_dict_path}")

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

train_targets = np.stack([instrument_labels[p] for p in train_pairs])
test_targets = np.stack([instrument_labels[p] for p in test_pairs])

train_features_t = torch.from_numpy(train_features).float()
test_features_t = torch.from_numpy(test_features).float()
train_targets_t = torch.from_numpy(train_targets).float()

train_prevalence = train_targets.mean(axis=0)
num_positive = train_targets.sum(axis=0)
num_negative = len(train_pairs) - num_positive
pos_weight = (num_negative / np.maximum(num_positive, 1)).astype(np.float32)
pos_weight_t = torch.from_numpy(pos_weight).to(device)

probe = nn.Linear(num_features, NUM_INSTRUMENTS).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LEARNING_RATE)

num_train = train_features_t.shape[0]
loss_history = []

for epoch in range(PROBE_EPOCHS):

    probe.train()
    permutation = torch.randperm(num_train)
    epoch_loss = 0.0
    num_batches = 0

    for start_idx in range(0, num_train, BATCH_SIZE):
        idx = permutation[start_idx:start_idx + BATCH_SIZE]
        batch_x = train_features_t[idx].to(device)
        batch_y = train_targets_t[idx].to(device)

        optimizer.zero_grad()
        logits = probe(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    print(f"[probe] Epoch {epoch + 1}/{PROBE_EPOCHS}: "
          f"train loss = {avg_loss:.4f}")

probe.eval()
with torch.no_grad():
    test_logits = probe(test_features_t.to(device))
    test_predictions = (torch.sigmoid(test_logits) > 0.5).float().cpu().numpy()

day21_f1_reference = {
    "grasper": 0.860, "bipolar": 0.106, "hook": 0.677,
    "scissors": 0.054, "clipper": 0.012, "irrigator": 0.100,
}
day27_f1_reference = {
    "grasper": 0.906, "bipolar": 0.431, "hook": 0.907,
    "scissors": 0.101, "clipper": 0.492, "irrigator": 0.236,
}
day31_f1_reference = {
    "grasper": 0.862, "bipolar": 0.310, "hook": 0.719,
    "scissors": 0.069, "clipper": 0.298, "irrigator": 0.181,
}

print()
print(f"{'Instrument':12s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s} "
      f"{'Day21':>8s} {'Day31(N32)':>11s} {'Day27':>8s}")

results = {"instruments": {}}

for i, name in enumerate(instrument_names):

    pred_i = test_predictions[:, i]
    label_i = test_targets[:, i]

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
          f"{day21_f1_reference[name]:8.3f} {day31_f1_reference[name]:11.3f} "
          f"{day27_f1_reference[name]:8.3f}")

    results["instruments"][name] = {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "day21_f1": day21_f1_reference[name],
        "day31_f1": day31_f1_reference[name],
        "day27_f1": day27_f1_reference[name],
    }

macro_f1 = np.mean([results["instruments"][n]["f1"] for n in instrument_names])

print()
print(f"Macro F1 (Day33, N=64/128 views): {macro_f1:.3f}")
print()
print("For reference:")
print("  Day21 (ImageNet frozen, no adaptation):            macro F1 0.302")
print("  Day31 (contrastive, N=32 images / 64 views):       macro F1 0.407")
print("  Day27 (ImageNet + supervised layer4 fine-tuning):  macro F1 0.512")

results["macro_f1"] = float(macro_f1)
results["loss_history"] = loss_history
results["train_video_ids"] = train_video_ids
results["test_video_ids"] = test_video_ids

output_dir = Path(__file__).parent
with open(output_dir / "probe_results.json", "w") as f:
    json.dump(results, f, indent=2)
