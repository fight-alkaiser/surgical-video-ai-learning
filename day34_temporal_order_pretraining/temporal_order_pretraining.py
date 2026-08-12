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
# Day31-33's contrastive pretraining used zero temporal
# information: its pretext task only ever compared two
# augmented views of the SAME frame against other frames
# treated as interchangeable negatives, regardless of which
# video or moment they came from. Day32 found this captured
# only a modest amount of phase structure (linear-probe
# accuracy 0.511 -> 0.532) -- consistent with phase being partly
# a temporal/procedural concept a frame-independent pretext task
# has no particular reason to learn well.
#
# Today tries a genuinely different self-supervised signal that
# DOES use temporal structure, still with zero labels: given two
# frames from the same video, predict which one comes later in
# the procedure. Concretely, a small "progress head" (a single
# linear layer) maps each frame's backbone features to one
# scalar; frames sampled later in the video should get a higher
# score than frames sampled earlier, trained with a pairwise
# ranking loss (RankNet-style). This is the classic "temporal
# order verification" family of self-supervised methods (e.g.
# Misra et al.'s Shuffle and Learn, 2016), simplified to pairs.
#
# Same backbone split as every prior SSL day (conv1-layer3
# frozen, layer4 trainable), same 8 training videos, same video-
# level split. No instrument/verb/target/phase label is used
# anywhere in this pretraining -- only each frame's position in
# its own video, which is not a CholecT50 annotation at all, just
# the frame's index.
# ----------------------------------------

DATASET_ROOT = Path("/Users/katsutoshimakino/Datasets/CholecT50/CholecT50")
VIDEOS_DIR = DATASET_ROOT / "videos"
LABELS_DIR = DATASET_ROOT / "labels"

VIDEO_IDS = [
    "VID01", "VID02", "VID04", "VID05", "VID06",
    "VID08", "VID10", "VID12", "VID13", "VID14",
]

TEST_RATIO = 0.2
BATCH_SIZE = 32          # N pairs -> 2N frames per batch, same as Day31
NUM_EPOCHS = 15
PAIRS_PER_EPOCH = 14212  # matches Day31's frames/epoch, for comparable compute
LEARNING_RATE_BACKBONE = 1e-4
LEARNING_RATE_HEAD = 1e-3
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# ----------------------------------------
# Video-level train/test split (identical convention/seed to
# every prior day). Only train_video_ids are used for
# pretraining.
# ----------------------------------------

video_frame_ids = {}
for video_id in VIDEO_IDS:
    with open(LABELS_DIR / f"{video_id}.json") as f:
        data = json.load(f)
    video_frame_ids[video_id] = sorted(int(f) for f in data["annotations"].keys())

shuffled_video_ids = VIDEO_IDS[:]
random.shuffle(shuffled_video_ids)

num_test_videos = max(1, round(len(VIDEO_IDS) * TEST_RATIO))
test_video_ids = sorted(shuffled_video_ids[:num_test_videos])
train_video_ids = sorted(shuffled_video_ids[num_test_videos:])

print(f"Train videos ({len(train_video_ids)}): {train_video_ids}")
print(f"Test videos  ({len(test_video_ids)}, held out entirely): {test_video_ids}")

train_video_frame_lists = [video_frame_ids[v] for v in train_video_ids]

# ----------------------------------------
# Dataset: each item is a random pair of frames from the same
# (randomly chosen) training video, with a binary target (1 if
# the first frame comes later in the video than the second).
# Light, single-frame augmentation only (random crop + flip) --
# unlike Day31's heavy color jitter/blur, there's no need for
# view-invariance here, just generalization.
# ----------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

frame_augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class TemporalOrderPairDataset(Dataset):
    """Each __getitem__ draws a fresh random pair from a randomly
    chosen training video -- __len__ only defines how many pairs
    constitute one epoch, for comparable training budget to
    Day31's per-epoch frame count."""

    def __init__(self, video_ids, video_frame_ids, num_pairs):
        self.video_ids = video_ids
        self.video_frame_ids = video_frame_ids
        self.num_pairs = num_pairs
        # Independent RNG so DataLoader workers (if any) don't
        # collide with the global seeded random module.
        self.rng = random.Random(RANDOM_SEED)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        video_id = self.rng.choice(self.video_ids)
        frame_list = self.video_frame_ids[video_id]
        i, j = self.rng.sample(range(len(frame_list)), 2)
        frame_a, frame_b = frame_list[i], frame_list[j]

        image_a = Image.open(
            VIDEOS_DIR / video_id / f"{frame_a:06d}.png"
        ).convert("RGB")
        image_b = Image.open(
            VIDEOS_DIR / video_id / f"{frame_b:06d}.png"
        ).convert("RGB")

        view_a = frame_augmentation(image_a)
        view_b = frame_augmentation(image_b)

        # target = 1.0 if frame_a comes LATER in the video than
        # frame_b (i.e. i > j), else 0.0
        target = 1.0 if i > j else 0.0

        return view_a, view_b, target


train_loader = DataLoader(
    TemporalOrderPairDataset(train_video_ids, video_frame_ids, PAIRS_PER_EPOCH),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# ----------------------------------------
# Model: same frozen/trainable split as every prior SSL day,
# plus a single-scalar "progress head" instead of a projection
# head. The backbone is called once per frame (both frames of a
# pair share the same weights -- a Siamese setup).
# ----------------------------------------

backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

frozen_modules = [
    backbone.conv1, backbone.bn1, backbone.layer1,
    backbone.layer2, backbone.layer3,
]
for module in frozen_modules:
    for param in module.parameters():
        param.requires_grad = False

num_features = backbone.fc.in_features
backbone.fc = nn.Identity()
backbone = backbone.to(device)

progress_head = nn.Linear(num_features, 1).to(device)


def set_training_mode():
    backbone.train()
    for module in frozen_modules:
        module.eval()
    progress_head.train()


optimizer = torch.optim.Adam([
    {"params": backbone.layer4.parameters(), "lr": LEARNING_RATE_BACKBONE},
    {"params": progress_head.parameters(), "lr": LEARNING_RATE_HEAD},
])

criterion = nn.BCEWithLogitsLoss()

# ----------------------------------------
# Train
# ----------------------------------------

loss_history = []
accuracy_history = []

for epoch in range(NUM_EPOCHS):

    set_training_mode()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    num_batches = 0
    start = time.time()

    for view_a, view_b, target in train_loader:
        view_a = view_a.to(device)
        view_b = view_b.to(device)
        target = target.float().to(device)

        optimizer.zero_grad()

        score_a = progress_head(backbone(view_a)).squeeze(1)
        score_b = progress_head(backbone(view_b)).squeeze(1)
        logit = score_a - score_b  # >0 means "a is predicted later than b"

        loss = criterion(logit, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predicted = (logit > 0).float()
        epoch_correct += (predicted == target).sum().item()
        epoch_total += target.shape[0]
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    accuracy = epoch_correct / epoch_total
    loss_history.append(avg_loss)
    accuracy_history.append(accuracy)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: loss = {avg_loss:.4f}, "
          f"order accuracy = {accuracy:.3f} ({time.time() - start:.0f}s)")

output_dir = Path(__file__).parent
torch.save(backbone.state_dict(), output_dir / "temporal_order_backbone.pt")

with open(output_dir / "pretraining_results.json", "w") as f:
    json.dump({
        "loss_history": loss_history,
        "order_accuracy_history": accuracy_history,
        "batch_size": BATCH_SIZE,
        "pairs_per_epoch": PAIRS_PER_EPOCH,
        "num_epochs": NUM_EPOCHS,
        "train_video_ids": train_video_ids,
        "test_video_ids": test_video_ids,
    }, f, indent=2)

print("\nTemporal-order pretraining done. Backbone saved to "
      f"{output_dir / 'temporal_order_backbone.pt'}")
