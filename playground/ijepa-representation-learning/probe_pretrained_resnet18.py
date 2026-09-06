"""Day95: does a small, frozen ImageNet-pretrained backbone beat both the
from-scratch I-JEPA encoder (Day93-94) and the random-init control?

Day94 found the from-scratch-trained I-JEPA encoder was *worse* than a
randomly initialized one of the same architecture at recovering the real
per-frame action (R^2~0.01 vs ~0.22) -- training on 200 episodes on a
CUDA-less Mac mini couldn't beat doing nothing. This tests the natural next
candidate: a backbone that was never trained on this tiny dataset at all,
but on ImageNet at a completely different scale, used here frozen
(inference only, no fine-tuning -- lighter on this Mac mini than training
anything from scratch was).

Same probe methodology as probe_representation_quality.py (Day94) and
../action-conditioned-video-prediction/probe_action_from_latents.py
(Day91): freeze the encoder, mean-pool its features, train a small MLP
probe to the real per-frame action, compare against the mean-action
baseline.
"""

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

AC_DATA_DIR = "../action-conditioned-video-prediction/data"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64
PROBE_EPOCHS = 50

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

with open(f"{AC_DATA_DIR}/episode_lengths.json") as f:
    episode_lengths = json.load(f)
episode_ids = sorted(episode_lengths.keys())
original_20 = [f"episode_{i:06d}" for i in range(20)]
rng = np.random.default_rng(0)
shuffled = rng.permutation(original_20)
val_episodes = set(shuffled[:4])
train_episodes = set(episode_ids) - val_episodes


def load_frames_and_actions(ep_ids):
    all_frames, all_actions = [], []
    for ep in sorted(ep_ids):
        frames = np.load(f"{AC_DATA_DIR}/episodes/{ep}_frames.npy")
        actions = np.load(f"{AC_DATA_DIR}/episodes/{ep}_actions.npy")
        n = min(len(frames), len(actions))
        all_frames.append(frames[:n])
        all_actions.append(actions[:n])
    return np.concatenate(all_frames), np.concatenate(all_actions)


train_frames, train_actions = load_frames_and_actions(train_episodes)
val_frames, val_actions = load_frames_and_actions(val_episodes)
print(f"train frames: {len(train_frames)}, val frames: {len(val_frames)}")

action_mean = train_actions.mean(axis=0)
action_std = train_actions.std(axis=0) + 1e-6


def to_tensor(frames, idx):
    f = torch.from_numpy(frames[idx]).float().permute(0, 3, 1, 2) / 255.0  # (B,3,64,64)
    f = F.interpolate(f, size=224, mode="bilinear", align_corners=False)  # ResNet18 was trained at 224x224
    f = (f - IMAGENET_MEAN) / IMAGENET_STD
    return f.to(DEVICE)


print("loading ImageNet-pretrained ResNet18 (torchvision, ~44MB download on first use)...")
resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()  # drop the 1000-way classifier; keep the 512-dim pooled feature
resnet = resnet.to(DEVICE).eval()
for p in resnet.parameters():
    p.requires_grad = False


def encode_all(frames):
    reps = []
    with torch.no_grad():
        for i in range(0, len(frames), BATCH_SIZE):
            idx = np.arange(i, min(i + BATCH_SIZE, len(frames)))
            f = to_tensor(frames, idx)
            reps.append(resnet(f).cpu())
    return torch.cat(reps)


print("encoding train/val frames...")
train_rep = encode_all(train_frames)
val_rep = encode_all(val_frames)
print(f"feature shape: {train_rep.shape}")


class Probe(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def train_and_eval_probe(train_rep, train_act, val_rep, val_act):
    probe = Probe(train_rep.shape[-1], train_act.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    tr, ta = train_rep.to(DEVICE), train_act.to(DEVICE)
    vr, va = val_rep.to(DEVICE), val_act.to(DEVICE)
    best_val = float("inf")
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(perm), 256):
            idx = perm[i : i + 256]
            loss = F.mse_loss(probe(tr[idx]), ta[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(probe(vr), va).item()
        best_val = min(best_val, val_loss)
    return best_val


train_act_t = torch.from_numpy((train_actions - action_mean) / action_std).float()
val_act_t = torch.from_numpy((val_actions - action_mean) / action_std).float()
mean_baseline_mse = F.mse_loss(train_act_t.mean(dim=0, keepdim=True).expand_as(val_act_t), val_act_t).item()

resnet_mse = train_and_eval_probe(train_rep, train_act_t, val_rep, val_act_t)

print("\n=== summary ===")
print(f"mean-action baseline:        {mean_baseline_mse:.4f}")
print(f"Day94 random I-JEPA encoder: 0.6285  (R^2 vs mean: 0.2240)")
print(f"Day94 trained I-JEPA encoder:0.8008  (R^2 vs mean: 0.0111)")
print(f"pretrained ResNet18 (frozen):{resnet_mse:.4f}  (R^2 vs mean: {1 - resnet_mse / mean_baseline_mse:.4f})")

with open("outputs/day95_probe_results.json", "w") as f:
    json.dump(
        {
            "mean_baseline_mse": mean_baseline_mse,
            "resnet18_mse": resnet_mse,
            "r2_resnet18_vs_mean": 1 - resnet_mse / mean_baseline_mse,
        },
        f,
        indent=2,
    )
print("\nsaved outputs/day95_probe_results.json")
