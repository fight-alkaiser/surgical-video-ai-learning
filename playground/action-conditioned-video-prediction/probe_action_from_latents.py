"""Day91: does the trained encoder's (z_t, z_{t+H}) pair even contain
information about which action was taken -- independent of the CFM
predictor entirely?

Day78-90 all modified either the action-window encoder architecture or the
training objective, and none of it moved the "zero action beats real
action" result. Rereading the DIAMOND paper (Day70-71 notes) raised a
different possibility: this project's online/target encoders are trained
only against the CFM prediction loss plus a VICReg-style anti-collapse
variance term -- nothing pushes them to preserve fine, action-relevant
visual detail specifically (only to not collapse to a single point). If
the encoder throws away exactly the detail an action would explain, no
amount of improving the action pathway or the sampling procedure can fix
it, because the target the predictor is trying to hit doesn't carry the
signal in the first place.

This script freezes a trained checkpoint's encoders and trains a small
probe to regress the actual action window from (z_t, z_{t+H}) alone, with
the predictor and action encoder never involved. If the probe can't beat
a plain mean-action baseline, the encoder itself -- not the predictor or
the action pathway -- is the more likely bottleneck.
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cfm_model import CFMActionModel

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=10)
parser.add_argument("--checkpoint", type=str, default="outputs/model_cfm_h10_noise_n200_seed2.pt")
parser.add_argument("--probe-epochs", type=int, default=50)
args = parser.parse_args()
H = args.horizon

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64

with open("data/episode_lengths.json") as f:
    episode_lengths = json.load(f)
episode_ids = sorted(episode_lengths.keys())

original_20 = [f"episode_{i:06d}" for i in range(20)]
rng = np.random.default_rng(0)
shuffled = rng.permutation(original_20)
val_episodes = set(shuffled[:4])
train_episodes = set(episode_ids) - val_episodes


def build_pairs(ep_ids):
    all_frame_t, all_action_t, all_frame_t1 = [], [], []
    for ep_id in ep_ids:
        frames = np.load(f"data/episodes/{ep_id}_frames.npy")
        actions = np.load(f"data/episodes/{ep_id}_actions.npy")
        if len(frames) <= H:
            continue
        action_window = np.stack([actions[i : i + H] for i in range(len(actions) - H)])
        all_frame_t.append(frames[:-H])
        all_action_t.append(action_window)
        all_frame_t1.append(frames[H:])
    return np.concatenate(all_frame_t), np.concatenate(all_action_t), np.concatenate(all_frame_t1)


train_frame_t, train_action_t, train_frame_t1 = build_pairs(train_episodes)
val_frame_t, val_action_t, val_frame_t1 = build_pairs(val_episodes)
print(f"train pairs: {len(train_frame_t)}, val pairs: {len(val_frame_t)}")

action_dim_per_step = train_action_t.shape[-1]
action_mean = train_action_t.reshape(-1, action_dim_per_step).mean(axis=0)
action_std = train_action_t.reshape(-1, action_dim_per_step).std(axis=0) + 1e-6


def to_tensor_batch(frame_t, action_t, frame_t1, idx):
    f = torch.from_numpy(frame_t[idx]).float().permute(0, 3, 1, 2) / 255.0
    a = torch.from_numpy((action_t[idx] - action_mean) / action_std).float().reshape(len(idx), -1)  # flattened (B, H*D)
    f1 = torch.from_numpy(frame_t1[idx]).float().permute(0, 3, 1, 2) / 255.0
    return f.to(DEVICE), a.to(DEVICE), f1.to(DEVICE)


# load a trained checkpoint, frozen -- only the encoders are used, the predictor
# and action encoder are irrelevant to this question and never touched
model = CFMActionModel(action_dim_per_step=action_dim_per_step, horizon=H).to(DEVICE)
model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
model.eval()
for p in model.parameters():
    p.requires_grad = False


@torch.no_grad()
def encode_all(frame_t, action_t, frame_t1):
    zt_list, zt1_list, a_list = [], [], []
    for i in range(0, len(frame_t), BATCH_SIZE):
        idx = np.arange(i, min(i + BATCH_SIZE, len(frame_t)))
        f, a, f1 = to_tensor_batch(frame_t, action_t, frame_t1, idx)
        zt_list.append(model.online_encoder(f).cpu())
        zt1_list.append(model.target_encoder(f1).cpu())
        a_list.append(a.cpu())
    return torch.cat(zt_list), torch.cat(zt1_list), torch.cat(a_list)


print("encoding train/val frames with the frozen encoder...")
train_zt, train_zt1, train_a = encode_all(train_frame_t, train_action_t, train_frame_t1)
val_zt, val_zt1, val_a = encode_all(val_frame_t, val_action_t, val_frame_t1)
action_flat_dim = train_a.shape[-1]  # H * action_dim_per_step
print(f"z_t {train_zt.shape}, z_t+H {train_zt1.shape}, action (flat) {train_a.shape}")


class Probe(nn.Module):
    """Small MLP: (z_t, z_t+H) -> flattened action window. No image, no
    predictor, no action encoder -- purely asking what's recoverable from the
    encoder's own outputs."""

    def __init__(self, embed_dim, action_flat_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, action_flat_dim),
        )

    def forward(self, z_t, z_t1):
        return self.net(torch.cat([z_t, z_t1], dim=-1))


def train_probe(zt, zt1, a, val_zt, val_zt1, val_a, label):
    probe = Probe(zt.shape[-1], a.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    zt, zt1, a = zt.to(DEVICE), zt1.to(DEVICE), a.to(DEVICE)
    val_zt, val_zt1, val_a = val_zt.to(DEVICE), val_zt1.to(DEVICE), val_a.to(DEVICE)
    best_val = float("inf")
    for epoch in range(args.probe_epochs):
        probe.train()
        perm = torch.randperm(len(zt))
        for i in range(0, len(perm), 256):
            idx = perm[i : i + 256]
            pred = probe(zt[idx], zt1[idx])
            loss = F.mse_loss(pred, a[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(probe(val_zt, val_zt1), val_a).item()
        best_val = min(best_val, val_loss)
        if epoch % 10 == 0 or epoch == args.probe_epochs - 1:
            print(f"  [{label}] epoch {epoch:3d}  val_mse {val_loss:.4f}")
    return best_val


print("\n--- probe: real (z_t, z_t+H) -> the action that actually happened ---")
real_mse = train_probe(train_zt, train_zt1, train_a, val_zt, val_zt1, val_a, "real")

print("\n--- control: mean-action baseline (predict the training mean action, ignore z entirely) ---")
mean_action = train_a.mean(dim=0, keepdim=True)
mean_baseline_mse = F.mse_loss(mean_action.expand_as(val_a), val_a).item()
print(f"  mean-action baseline val_mse {mean_baseline_mse:.4f}")

print("\n--- control: shuffled -- same (z_t, z_t+H) pairs, but actions randomly reassigned ---")
rand_idx = np.random.default_rng(2).permutation(len(train_a))
shuffled_train_a = train_a[rand_idx]
rand_idx_val = np.random.default_rng(3).permutation(len(val_a))
shuffled_val_a = val_a[rand_idx_val]
shuffled_mse = train_probe(train_zt, train_zt1, shuffled_train_a, val_zt, val_zt1, shuffled_val_a, "shuffled")

print("\n=== summary ===")
print(f"mean-action baseline (ignores z entirely): {mean_baseline_mse:.4f}")
print(f"probe on shuffled actions (z has no relation to target): {shuffled_mse:.4f}")
print(f"probe on real actions:                                   {real_mse:.4f}")
r2_vs_mean = 1 - real_mse / mean_baseline_mse
print(f"\nR^2 of real-action probe vs. mean-action baseline: {r2_vs_mean:.4f}")

with open("outputs/day91_probe_results.json", "w") as f:
    json.dump(
        {
            "checkpoint": args.checkpoint,
            "mean_action_baseline_mse": mean_baseline_mse,
            "shuffled_probe_mse": shuffled_mse,
            "real_probe_mse": real_mse,
            "r2_vs_mean_baseline": r2_vs_mean,
        },
        f,
        indent=2,
    )
print("\nsaved outputs/day91_probe_results.json")
