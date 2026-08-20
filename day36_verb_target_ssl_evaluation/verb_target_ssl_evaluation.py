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
# Day31-35 evaluated the SSL-pretrained backbones (contrastive,
# temporal-order) on only two downstream tasks: instrument
# recognition and phase recognition. Day34 found a split verdict
# between them (temporal-order won on instrument, lost badly on
# phase), which raised an open question Day35 flagged directly:
# does the same kind of split show up on tasks not tested yet?
#
# Today reuses the exact backbones already saved to disk from
# Day31 (contrastive) and Day34 (temporal-order) -- no retraining
# -- and evaluates them on verb (10 classes, Day22's task) and
# target (15 classes, Day29's task), with the same class-weighted
# linear probe recipe used throughout (Day26/31/34), so every
# number here is directly comparable to the instrument/phase
# results already established.
#
# Three backbone variants x two tasks = six probes, all on the
# same 10 videos / video-level 8-2 split used everywhere in this
# project.
# ----------------------------------------

DATASET_ROOT = Path("/Users/katsutoshimakino/Datasets/CholecT50/CholecT50")
VIDEOS_DIR = DATASET_ROOT / "videos"
LABELS_DIR = DATASET_ROOT / "labels"

VIDEO_IDS = [
    "VID01", "VID02", "VID04", "VID05", "VID06",
    "VID08", "VID10", "VID12", "VID13", "VID14",
]

NUM_VERBS = 10
NUM_TARGETS = 15
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
# Build per-frame verb and target multi-hot labels (identical
# extraction logic to Day22/29).
# ----------------------------------------

verb_names = None
target_names = None
verb_labels = {}
target_labels = {}
video_frame_ids = {}

for video_id in VIDEO_IDS:

    with open(LABELS_DIR / f"{video_id}.json") as f:
        data = json.load(f)

    if verb_names is None:
        verb_names = [
            data["categories"]["verb"][str(i)] for i in range(NUM_VERBS)
        ]
        target_names = [
            data["categories"]["target"][str(i)] for i in range(NUM_TARGETS)
        ]

    frame_ids = sorted(int(f) for f in data["annotations"].keys())
    video_frame_ids[video_id] = frame_ids

    for frame in frame_ids:

        verbs_present = set()
        targets_present = set()

        for triplet in data["annotations"][str(frame)]:
            verb_id = triplet[7]
            target_id = triplet[8]
            if verb_id != -1:
                verbs_present.add(verb_id)
            if target_id != -1:
                targets_present.add(target_id)

        verb_label = np.zeros(NUM_VERBS, dtype=np.float32)
        for vid_ in verbs_present:
            verb_label[vid_] = 1.0
        verb_labels[(video_id, frame)] = verb_label

        target_label = np.zeros(NUM_TARGETS, dtype=np.float32)
        for tid in targets_present:
            target_label[tid] = 1.0
        target_labels[(video_id, frame)] = target_label

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

train_verb_targets = np.stack([verb_labels[p] for p in train_pairs])
test_verb_targets = np.stack([verb_labels[p] for p in test_pairs])
train_target_targets = np.stack([target_labels[p] for p in train_pairs])
test_target_targets = np.stack([target_labels[p] for p in test_pairs])

# ----------------------------------------
# Feature extraction (Day24's caching shortcut), per backbone
# variant.
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


def build_backbone(checkpoint_path):
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    if checkpoint_path is not None:
        backbone.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    for param in backbone.parameters():
        param.requires_grad = False
    backbone = backbone.to(device)
    backbone.eval()
    return backbone, num_features


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


REPO_ROOT = Path(__file__).parent.parent

variant_checkpoints = {
    "imagenet_frozen": None,
    "contrastive (Day31)": REPO_ROOT / "day31_contrastive_pretraining" / "contrastive_backbone.pt",
    "temporal_order (Day34)": REPO_ROOT / "day34_temporal_order_pretraining" / "temporal_order_backbone.pt",
}

# ----------------------------------------
# Class-weighted linear probe (Day26's recipe), reused for both
# verb and target.
# ----------------------------------------


def train_probe(train_features, train_targets, num_classes, feature_dim):

    train_prevalence = train_targets.mean(axis=0)
    num_positive = train_targets.sum(axis=0)
    num_negative = len(train_targets) - num_positive
    pos_weight = (num_negative / np.maximum(num_positive, 1)).astype(np.float32)
    pos_weight_t = torch.from_numpy(pos_weight).to(device)

    probe = nn.Linear(feature_dim, num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = torch.optim.Adam(probe.parameters(), lr=PROBE_LEARNING_RATE)

    train_features_t = torch.from_numpy(train_features).float()
    train_targets_t = torch.from_numpy(train_targets).float()
    num_train = train_features_t.shape[0]

    for epoch in range(PROBE_EPOCHS):
        permutation = torch.randperm(num_train)
        for start_idx in range(0, num_train, BATCH_SIZE):
            idx = permutation[start_idx:start_idx + BATCH_SIZE]
            batch_x = train_features_t[idx].to(device)
            batch_y = train_targets_t[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(probe(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    return probe


def evaluate_probe(probe, test_features, test_targets, class_names):

    test_features_t = torch.from_numpy(test_features).float()
    probe.eval()
    with torch.no_grad():
        logits = probe(test_features_t.to(device))
        predictions = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()

    per_class = {}
    for i, name in enumerate(class_names):
        pred_i = predictions[:, i]
        label_i = test_targets[:, i]
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


results = {"verb": {}, "target": {}}

for variant_name, checkpoint_path in variant_checkpoints.items():

    print(f"\n=== {variant_name} ===")
    backbone, num_features = build_backbone(checkpoint_path)
    train_features = extract_features(backbone, train_pairs, "train")
    test_features = extract_features(backbone, test_pairs, "test")

    print("  Training verb probe...")
    verb_probe = train_probe(train_features, train_verb_targets, NUM_VERBS, num_features)
    verb_macro_f1, verb_per_class = evaluate_probe(
        verb_probe, test_features, test_verb_targets, verb_names
    )
    print(f"  Verb macro F1: {verb_macro_f1:.3f}")

    print("  Training target probe...")
    target_probe = train_probe(train_features, train_target_targets, NUM_TARGETS, num_features)
    target_macro_f1, target_per_class = evaluate_probe(
        target_probe, test_features, test_target_targets, target_names
    )
    print(f"  Target macro F1: {target_macro_f1:.3f}")

    results["verb"][variant_name] = {"macro_f1": verb_macro_f1, "per_class": verb_per_class}
    results["target"][variant_name] = {"macro_f1": target_macro_f1, "per_class": target_per_class}

# ----------------------------------------
# Summary
# ----------------------------------------

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"\n{'Variant':25s} {'Verb macro F1':>15s} {'Target macro F1':>17s}")
for variant_name in variant_checkpoints:
    v = results["verb"][variant_name]["macro_f1"]
    t = results["target"][variant_name]["macro_f1"]
    print(f"{variant_name:25s} {v:15.3f} {t:17.3f}")

print("\nFor reference (different methodology, not directly comparable "
      "row-for-row, but same underlying task):")
print("  Day22 verb (frozen, UNweighted, no SSL):         macro F1 0.192")
print("  Day29 target (fine-tuned + weighted, supervised): macro F1 0.207")

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
