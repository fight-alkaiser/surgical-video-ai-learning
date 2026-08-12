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
# Evaluate the temporal-order-pretrained backbone the same way
# as every prior SSL day: freeze it, cache features, linear
# probe -- both for instrument recognition (Day21/26/27/31/33's
# task) and phase recognition (Day32's task), so this new
# backbone can be compared against contrastive pretraining on
# both axes.
#
# Reference points, same 10 videos / video-level 8-2 split:
#   Instrument macro F1 -- Day21 (frozen) 0.302, Day31 (contrastive) 0.407,
#     Day33 (contrastive, larger batch) 0.406, Day27 (supervised) 0.512
#   Phase-probe accuracy -- Day32: ImageNet frozen 0.511, contrastive 0.532
# ----------------------------------------

DATASET_ROOT = Path("/Users/katsutoshimakino/Datasets/CholecT50/CholecT50")
VIDEOS_DIR = DATASET_ROOT / "videos"
LABELS_DIR = DATASET_ROOT / "labels"

VIDEO_IDS = [
    "VID01", "VID02", "VID04", "VID05", "VID06",
    "VID08", "VID10", "VID12", "VID13", "VID14",
]

NUM_INSTRUMENTS = 6
NUM_PHASES = 7
TEST_RATIO = 0.2
BATCH_SIZE = 32
PROBE_EPOCHS = 15
PROBE_LEARNING_RATE = 1e-3
PHASE_PROBE_EPOCHS = 200
PHASE_PROBE_LEARNING_RATE = 0.5
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# ----------------------------------------
# Build per-frame instrument multi-hot and phase labels.
# ----------------------------------------

instrument_names = None
phase_names = None
instrument_labels = {}
phase_labels = {}
video_frame_ids = {}

for video_id in VIDEO_IDS:

    with open(LABELS_DIR / f"{video_id}.json") as f:
        data = json.load(f)

    if instrument_names is None:
        instrument_names = [
            data["categories"]["instrument"][str(i)]
            for i in range(NUM_INSTRUMENTS)
        ]
        phase_names = [
            data["categories"]["phase"][str(i)] for i in range(NUM_PHASES)
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

        phase_labels[(video_id, frame)] = data["annotations"][str(frame)][0][-1]

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
train_pairs_valid_phase = [p for p in train_pairs if phase_labels[p] != -1]
test_pairs_valid_phase = [p for p in test_pairs if phase_labels[p] != -1]

print(f"Train frames: {len(train_pairs)}, Test frames: {len(test_pairs)}")

# ----------------------------------------
# Load the temporal-order-pretrained, now-frozen backbone.
# ----------------------------------------

backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = backbone.fc.in_features
backbone.fc = nn.Identity()

state_dict_path = Path(__file__).parent / "temporal_order_backbone.pt"
backbone.load_state_dict(torch.load(state_dict_path, map_location="cpu"))

for param in backbone.parameters():
    param.requires_grad = False
backbone = backbone.to(device)
backbone.eval()

print(f"\nLoaded temporal-order-pretrained backbone from {state_dict_path}")

# ----------------------------------------
# Cache features (Day24's shortcut).
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


all_pairs = sorted(set(train_pairs) | set(test_pairs))
pair_to_row = {p: i for i, p in enumerate(all_pairs)}
all_features = extract_features(all_pairs, "all")

train_features = np.stack([all_features[pair_to_row[p]] for p in train_pairs])
test_features = np.stack([all_features[pair_to_row[p]] for p in test_pairs])

# ----------------------------------------
# Probe 1: instrument recognition (class-weighted, Day26 recipe)
# ----------------------------------------

train_instrument_targets = np.stack(
    [instrument_labels[p] for p in train_pairs]
)
test_instrument_targets = np.stack(
    [instrument_labels[p] for p in test_pairs]
)

train_features_t = torch.from_numpy(train_features).float()
test_features_t = torch.from_numpy(test_features).float()
train_instrument_t = torch.from_numpy(train_instrument_targets).float()

train_prevalence = train_instrument_targets.mean(axis=0)
num_positive = train_instrument_targets.sum(axis=0)
num_negative = len(train_pairs) - num_positive
pos_weight = (num_negative / np.maximum(num_positive, 1)).astype(np.float32)
pos_weight_t = torch.from_numpy(pos_weight).to(device)

probe = nn.Linear(num_features, NUM_INSTRUMENTS).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LEARNING_RATE)

num_train = train_features_t.shape[0]
print("\n--- Instrument probe ---")
for epoch in range(PROBE_EPOCHS):
    probe.train()
    permutation = torch.randperm(num_train)
    for start_idx in range(0, num_train, BATCH_SIZE):
        idx = permutation[start_idx:start_idx + BATCH_SIZE]
        batch_x = train_features_t[idx].to(device)
        batch_y = train_instrument_t[idx].to(device)
        optimizer.zero_grad()
        loss = criterion(probe(batch_x), batch_y)
        loss.backward()
        optimizer.step()

probe.eval()
with torch.no_grad():
    test_logits = probe(test_features_t.to(device))
    test_predictions = (torch.sigmoid(test_logits) > 0.5).float().cpu().numpy()

day21_f1 = {"grasper": 0.860, "bipolar": 0.106, "hook": 0.677,
            "scissors": 0.054, "clipper": 0.012, "irrigator": 0.100}
day31_f1 = {"grasper": 0.862, "bipolar": 0.310, "hook": 0.719,
            "scissors": 0.069, "clipper": 0.298, "irrigator": 0.181}

print(f"{'Instrument':12s} {'F1':>8s} {'Day21':>8s} {'Day31':>8s}")
instrument_results = {}
for i, name in enumerate(instrument_names):
    pred_i = test_predictions[:, i]
    label_i = test_instrument_targets[:, i]
    tp = ((pred_i == 1) & (label_i == 1)).sum()
    fp = ((pred_i == 1) & (label_i == 0)).sum()
    fn = ((pred_i == 0) & (label_i == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"{name:12s} {f1:8.3f} {day21_f1[name]:8.3f} {day31_f1[name]:8.3f}")
    instrument_results[name] = {"f1": float(f1), "precision": float(precision), "recall": float(recall)}

instrument_macro_f1 = np.mean([instrument_results[n]["f1"] for n in instrument_names])
print(f"\nInstrument macro F1: {instrument_macro_f1:.3f}")
print("Reference: Day21 0.302, Day31 0.407, Day33 0.406, Day27 0.512")

# ----------------------------------------
# Probe 2: phase recognition (softmax, Day17/19/32 recipe)
# ----------------------------------------

train_phase_features = np.stack(
    [all_features[pair_to_row[p]] for p in train_pairs_valid_phase]
)
test_phase_features = np.stack(
    [all_features[pair_to_row[p]] for p in test_pairs_valid_phase]
)
train_phase_ids = np.array([phase_labels[p] for p in train_pairs_valid_phase])
test_phase_ids = np.array([phase_labels[p] for p in test_pairs_valid_phase])


def run_phase_probe(train_features, train_labels, test_features, test_labels,
                     num_classes, feature_dim):
    rng = np.random.RandomState(RANDOM_SEED)
    W = rng.randn(num_classes, feature_dim) / np.sqrt(feature_dim)
    b = np.zeros(num_classes)
    num_train = train_features.shape[0]
    for epoch in range(PHASE_PROBE_EPOCHS):
        logits = train_features @ W.T + b
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        dlogits = probs.copy()
        dlogits[np.arange(num_train), train_labels] -= 1
        dlogits /= num_train
        W -= PHASE_PROBE_LEARNING_RATE * (dlogits.T @ train_features)
        b -= PHASE_PROBE_LEARNING_RATE * dlogits.sum(axis=0)
    test_logits = test_features @ W.T + b
    predicted = test_logits.argmax(axis=1)
    return (predicted == test_labels).mean()


phase_accuracy = run_phase_probe(
    train_phase_features, train_phase_ids,
    test_phase_features, test_phase_ids,
    NUM_PHASES, num_features,
)
phase_baseline = (
    test_phase_ids == np.bincount(train_phase_ids).argmax()
).mean()

print("\n--- Phase probe ---")
print(f"Phase-probe accuracy: {phase_accuracy:.3f} (baseline {phase_baseline:.3f})")
print("Reference: Day32 ImageNet frozen 0.511, contrastive-adapted 0.532")

results = {
    "instruments": instrument_results,
    "instrument_macro_f1": float(instrument_macro_f1),
    "phase_probe_accuracy": float(phase_accuracy),
    "phase_baseline_accuracy": float(phase_baseline),
    "train_video_ids": train_video_ids,
    "test_video_ids": test_video_ids,
}

output_dir = Path(__file__).parent
with open(output_dir / "probe_results.json", "w") as f:
    json.dump(results, f, indent=2)
