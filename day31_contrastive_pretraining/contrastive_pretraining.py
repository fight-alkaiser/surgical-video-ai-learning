import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ----------------------------------------
# Day01-29 were all supervised: every day started from a human-
# provided label (a triplet, a phase, an instrument/verb/target).
# Foundation models are built the other way around -- learn a
# useful representation from unlabeled data first, using a
# "pretext task" invented from the data itself, then adapt
# cheaply to downstream labeled tasks. Today tries the most
# influential modern version of that idea, contrastive learning
# (SimCLR, Chen et al. 2020): take one frame, create two
# randomly-augmented views of it, train the model to recognize
# that they came from the same frame (pulled together in
# embedding space) while every other frame in the batch is
# pushed apart. No instrument/verb/target/phase label is used
# anywhere in this script.
#
# This is deliberately a small, self-built reproduction of the
# MECHANISM, not a real foundation model: full-scale SimCLR uses
# batch sizes in the hundreds to thousands (more negatives per
# batch measurably improves it); this machine's 8GB RAM limits
# the batch size to a small fraction of that. The question this
# day asks is narrower and still meaningful: starting from
# ImageNet-pretrained weights, does adapting layer4 to this
# surgical dataset with NO labels at all move the frozen-feature
# baseline (Day21, instrument macro F1 0.302) any closer to what
# instrument LABELS bought via fine-tuning (Day27, F1 0.512)?
#
# Only the 8 TRAINING videos (same video-level split as every
# instrument/verb/target day) are used for pretraining -- the 2
# test videos are never touched, with or without labels, keeping
# this comparable to every earlier day's evaluation protocol.
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
BATCH_SIZE = 32          # N images -> 2N augmented views per batch
NUM_EPOCHS = 15
LEARNING_RATE_BACKBONE = 1e-4
LEARNING_RATE_HEAD = 1e-3
TEMPERATURE = 0.5
PROJECTION_DIM = 128
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# ----------------------------------------
# Video-level train/test split (identical convention/seed to
# every instrument/verb/target day). Only train_video_ids are
# used for contrastive pretraining.
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
print(f"Test videos  ({len(test_video_ids)}, held out entirely -- "
      f"not used, not even without labels): {test_video_ids}")

train_pairs = [
    (v, f) for v in train_video_ids for f in video_frame_ids[v]
]
print(f"Pretraining frames (unlabeled): {len(train_pairs)}")

# ----------------------------------------
# SimCLR-style augmentation: two independent random views of
# the same frame. Standard recipe (random resized crop, flip,
# color jitter, grayscale, blur) -- kept as-is rather than
# custom-tuned for surgical images, so this stays a recognizable
# reproduction of the published method. Worth flagging as a
# limitation: color jitter in particular perturbs cues (tissue
# color, bleeding) that are clinically meaningful in real
# surgical video, unlike in natural-image benchmarks.
# ----------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

simclr_augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class ContrastivePretrainDataset(Dataset):
    """Returns two independently-augmented views of the same frame."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        video_id, frame = self.pairs[idx]
        image_path = VIDEOS_DIR / video_id / f"{frame:06d}.png"
        image = Image.open(image_path).convert("RGB")
        view1 = simclr_augmentation(image)
        view2 = simclr_augmentation(image)
        return view1, view2


train_loader = DataLoader(
    ContrastivePretrainDataset(train_pairs), batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0, drop_last=True
)

# ----------------------------------------
# Model: ResNet18, conv1/bn1/layer1/layer2/layer3 frozen (same
# split as Day27), layer4 trainable, plus a small projection
# head (512 -> 512 -> 128) used only during contrastive training
# and discarded afterward -- standard SimCLR practice; the
# projection head's job is to make the contrastive loss easier
# to optimize, and the representation used for downstream tasks
# is the layer BEFORE it (the 512-d backbone output), not the
# projection itself.
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

projection_head = nn.Sequential(
    nn.Linear(num_features, num_features),
    nn.ReLU(inplace=True),
    nn.Linear(num_features, PROJECTION_DIM),
).to(device)


def set_training_mode():
    backbone.train()
    for module in frozen_modules:
        module.eval()
    projection_head.train()


optimizer = torch.optim.Adam([
    {"params": backbone.layer4.parameters(), "lr": LEARNING_RATE_BACKBONE},
    {"params": projection_head.parameters(), "lr": LEARNING_RATE_HEAD},
])


def nt_xent_loss(z1, z2, temperature):
    """Normalized temperature-scaled cross-entropy loss (Chen et al.,
    2020). z1[i] and z2[i] are the two views of the same frame i;
    every other view in the batch (both z1 and z2 of other frames)
    is a negative."""

    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)               # (2N, D)
    z = F.normalize(z, dim=1)

    similarity = z @ z.T / temperature            # (2N, 2N)
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    similarity = similarity.masked_fill(self_mask, float("-inf"))

    positive_indices = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size),
    ]).to(z.device)

    loss = F.cross_entropy(similarity, positive_indices)
    return loss


# ----------------------------------------
# Train
# ----------------------------------------

loss_history = []

for epoch in range(NUM_EPOCHS):

    set_training_mode()
    epoch_loss = 0.0
    num_batches = 0
    start = time.time()

    for view1, view2 in train_loader:
        view1, view2 = view1.to(device), view2.to(device)

        optimizer.zero_grad()

        features1 = backbone(view1)
        features2 = backbone(view2)
        z1 = projection_head(features1)
        z2 = projection_head(features2)

        loss = nt_xent_loss(z1, z2, TEMPERATURE)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: NT-Xent loss = {avg_loss:.4f} "
          f"({time.time() - start:.0f}s)")

output_dir = Path(__file__).parent
torch.save(backbone.state_dict(), output_dir / "contrastive_backbone.pt")

with open(output_dir / "pretraining_results.json", "w") as f:
    json.dump({
        "loss_history": loss_history,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "temperature": TEMPERATURE,
        "train_video_ids": train_video_ids,
        "test_video_ids": test_video_ids,
        "num_pretrain_frames": len(train_pairs),
    }, f, indent=2)

print("\nContrastive pretraining done. Backbone saved to "
      f"{output_dir / 'contrastive_backbone.pt'}")
