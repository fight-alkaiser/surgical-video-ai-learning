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
# Day31 flagged its own biggest named limitation: full-scale
# SimCLR uses batch sizes in the hundreds to thousands (more
# negatives per contrastive batch is known to help), while this
# 8GB RAM machine could only manage N=32 images (64 augmented
# views) per batch. Today tests that specific, named caveat
# directly: doubling the batch size to N=64 (128 views) -- as
# far as this machine's RAM comfortably allows -- with the
# learning rate scaled up 2x to match (the standard "linear
# scaling rule" for large-batch training, Goyal et al. 2017:
# without it, a larger batch just means fewer optimizer steps
# per epoch, confounding "does more negatives help" with "did
# we undertrain from fewer updates").
#
# Identical in every other respect to Day31: same 8 training
# videos, same frozen conv1-layer3 / trainable layer4 split,
# same NT-Xent loss, same 15 epochs, same evaluation protocol
# to follow (class-weighted linear probe on instrument
# recognition, same video split).
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
BATCH_SIZE = 64          # N images -> 2N=128 augmented views per batch
                          # (Day31 used N=32 -> 64 views)
NUM_EPOCHS = 15
BATCH_SCALE = BATCH_SIZE / 32  # relative to Day31, for linear LR scaling
LEARNING_RATE_BACKBONE = 1e-4 * BATCH_SCALE
LEARNING_RATE_HEAD = 1e-3 * BATCH_SCALE
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
# every prior day). Only train_video_ids are used for
# contrastive pretraining.
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

train_pairs = [
    (v, f) for v in train_video_ids for f in video_frame_ids[v]
]
print(f"Pretraining frames (unlabeled): {len(train_pairs)}")
print(f"Batch size: {BATCH_SIZE} images -> {2 * BATCH_SIZE} views/batch "
      f"(Day31 reference: 32 images -> 64 views)")
print(f"Learning rates (linearly scaled {BATCH_SCALE:.1f}x from Day31): "
      f"backbone={LEARNING_RATE_BACKBONE:.2e}, head={LEARNING_RATE_HEAD:.2e}")

# ----------------------------------------
# SimCLR-style augmentation (identical to Day31).
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
# Model (identical architecture to Day31).
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
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)

    similarity = z @ z.T / temperature
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
          f"({time.time() - start:.0f}s, {num_batches} batches)")

output_dir = Path(__file__).parent
torch.save(backbone.state_dict(), output_dir / "contrastive_backbone_large_batch.pt")

with open(output_dir / "pretraining_results.json", "w") as f:
    json.dump({
        "loss_history": loss_history,
        "batch_size": BATCH_SIZE,
        "views_per_batch": 2 * BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate_backbone": LEARNING_RATE_BACKBONE,
        "learning_rate_head": LEARNING_RATE_HEAD,
        "temperature": TEMPERATURE,
        "train_video_ids": train_video_ids,
        "test_video_ids": test_video_ids,
        "num_pretrain_frames": len(train_pairs),
    }, f, indent=2)

print("\nContrastive pretraining (large batch) done. Backbone saved to "
      f"{output_dir / 'contrastive_backbone_large_batch.pt'}")
