"""Day155: re-probe Day95's frozen ImageNet ResNet18 under the Day155 protocol.

Day155 found that standardizing the pooled features before the probe changes the
random-encoder reference a lot (R^2 ~0.2 -> ~0.45), because those encoders' outputs
are tiny. Day95's ResNet18 number (0.69) was measured without standardization, so
it is re-measured here the same way as day154_probe_checkpoints.py --both-scalings:
raw and standardized (train statistics) features, probe seeds 0/1/2.

Encoder setup (224x224 resize, ImageNet normalization, fc removed, 512-dim pooled
feature) is the same as probe_pretrained_resnet18.py.
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
PROBE_SEEDS = [0, 1, 2]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

with open(f"{AC_DATA_DIR}/episode_lengths.json") as f:
    episode_ids = sorted(json.load(f).keys())
original_20 = [f"episode_{i:06d}" for i in range(20)]
val_episodes = set(np.random.default_rng(0).permutation(original_20)[:4])
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
action_mean, action_std = train_actions.mean(0), train_actions.std(0) + 1e-6
train_act_t = torch.from_numpy((train_actions - action_mean) / action_std).float()
val_act_t = torch.from_numpy((val_actions - action_mean) / action_std).float()
mean_baseline_mse = F.mse_loss(train_act_t.mean(0, keepdim=True).expand_as(val_act_t), val_act_t).item()

resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE).eval()


@torch.no_grad()
def encode_all(frames):
    reps = []
    for i in range(0, len(frames), BATCH_SIZE):
        f = torch.from_numpy(frames[i : i + BATCH_SIZE]).float().permute(0, 3, 1, 2) / 255.0
        f = F.interpolate(f, size=224, mode="bilinear", align_corners=False)
        f = ((f - IMAGENET_MEAN) / IMAGENET_STD).to(DEVICE)
        reps.append(resnet(f).cpu())
    return torch.cat(reps)


class Probe(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def probe_r2(tr, vr, seed):
    torch.manual_seed(seed)
    probe = Probe(tr.shape[-1], train_act_t.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    tr, ta, vr, va = tr.to(DEVICE), train_act_t.to(DEVICE), vr.to(DEVICE), val_act_t.to(DEVICE)
    best = float("inf")
    for _ in range(PROBE_EPOCHS):
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
            best = min(best, F.mse_loss(probe(vr), va).item())
    return 1 - best / mean_baseline_mse


train_rep, val_rep = encode_all(train_frames), encode_all(val_frames)
mu, sd = train_rep.mean(0, keepdim=True), train_rep.std(0, keepdim=True) + 1e-6
result = {
    "mean_baseline_mse": mean_baseline_mse,
    "feat_scale": train_rep.std(0).mean().item(),
    "raw": [probe_r2(train_rep, val_rep, s) for s in PROBE_SEEDS],
    "std": [probe_r2((train_rep - mu) / sd, (val_rep - mu) / sd, s) for s in PROBE_SEEDS],
}
print(
    f"ResNet18: scale {result['feat_scale']:.3f}  "
    f"R2[raw] {np.mean(result['raw']):.3f}+-{np.std(result['raw']):.3f}  "
    f"R2[std] {np.mean(result['std']):.3f}+-{np.std(result['std']):.3f}"
)
with open("outputs/day155_resnet18_standardized.json", "w") as f:
    json.dump(result, f, indent=2)
