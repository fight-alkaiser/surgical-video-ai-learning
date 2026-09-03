"""Day92: is the R^2~0.18 signal found in Day91's probe (probe_action_from_
latents.py) spread evenly across all 16 action dimensions, or concentrated
in a few?

The 16-dim action is [left_xyz(3), left_quat(4), left_gripper(1),
right_xyz(3), right_quat(4), right_gripper(1)] (see README.md, "Data").
Gripper open/close is a coarse, close-to-binary signal; xyz/quaternion are
continuous and fine-grained. If the encoder mainly preserves the coarse
gripper signal and not the fine pose changes, that would sharpen Day91's
"the encoder preserves real signal" finding into something more specific
-- and would matter for what Day91's "may be out of reach at this scale"
conclusion actually means (maybe *some* of it isn't, even if the fine
motion is).

Reuses the same frozen-encoder probe setup as Day91, but reports R^2 per
action dimension instead of one aggregate number.
"""

import argparse
import json

import matplotlib.pyplot as plt
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

action_dim_per_step = train_action_t.shape[-1]  # 16
action_mean = train_action_t.reshape(-1, action_dim_per_step).mean(axis=0)
action_std = train_action_t.reshape(-1, action_dim_per_step).std(axis=0) + 1e-6


def to_tensor_batch(frame_t, action_t, frame_t1, idx):
    f = torch.from_numpy(frame_t[idx]).float().permute(0, 3, 1, 2) / 255.0
    a = torch.from_numpy((action_t[idx] - action_mean) / action_std).float()  # (B, H, 16), normalized, NOT flattened
    f1 = torch.from_numpy(frame_t1[idx]).float().permute(0, 3, 1, 2) / 255.0
    return f.to(DEVICE), a.to(DEVICE), f1.to(DEVICE)


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
        a_list.append(a.cpu())  # (B, H, 16)
    return torch.cat(zt_list), torch.cat(zt1_list), torch.cat(a_list)


print("encoding train/val frames with the frozen encoder...")
train_zt, train_zt1, train_a = encode_all(train_frame_t, train_action_t, train_frame_t1)
val_zt, val_zt1, val_a = encode_all(val_frame_t, val_action_t, val_frame_t1)
# average over the H window per dimension -- collapses (B, H, 16) -> (B, 16), enough to ask
# "which physical dimension carries signal", without the probe needing to also solve "which
# timestep in the window" on top of that
train_a_mean_over_h = train_a.mean(dim=1)  # (B, 16)
val_a_mean_over_h = val_a.mean(dim=1)
print(f"z_t {train_zt.shape}, action (mean over window) {train_a_mean_over_h.shape}")


class Probe(nn.Module):
    def __init__(self, embed_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z_t, z_t1):
        return self.net(torch.cat([z_t, z_t1], dim=-1))


def train_probe(zt, zt1, a, val_zt, val_zt1, val_a):
    probe = Probe(zt.shape[-1], a.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    zt, zt1, a = zt.to(DEVICE), zt1.to(DEVICE), a.to(DEVICE)
    val_zt, val_zt1, val_a = val_zt.to(DEVICE), val_zt1.to(DEVICE), val_a.to(DEVICE)
    best_val_pred = None
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
            val_pred = probe(val_zt, val_zt1)
            val_loss = F.mse_loss(val_pred, val_a).item()
        if val_loss < best_val:
            best_val = val_loss
            best_val_pred = val_pred.detach().cpu()
        if epoch % 10 == 0 or epoch == args.probe_epochs - 1:
            print(f"  epoch {epoch:3d}  val_mse {val_loss:.4f}")
    return best_val_pred


print("\n--- training probe: (z_t, z_t+H) -> 16-dim action (averaged over window) ---")
val_pred = train_probe(train_zt, train_zt1, train_a_mean_over_h, val_zt, val_zt1, val_a_mean_over_h)

dim_names = (
    [f"L_xyz_{i}" for i in range(3)] + [f"L_quat_{i}" for i in range(4)] + ["L_gripper"]
    + [f"R_xyz_{i}" for i in range(3)] + [f"R_quat_{i}" for i in range(4)] + ["R_gripper"]
)
groups = {
    "left_xyz": [0, 1, 2],
    "left_quat": [3, 4, 5, 6],
    "left_gripper": [7],
    "right_xyz": [8, 9, 10],
    "right_quat": [11, 12, 13, 14],
    "right_gripper": [15],
}

val_a_np = val_a_mean_over_h.numpy()
val_pred_np = val_pred.numpy()
mean_action_np = train_a_mean_over_h.mean(dim=0).numpy()

per_dim_r2 = []
print("\ndimension   R^2")
for d in range(action_dim_per_step):
    mse_pred = float(np.mean((val_pred_np[:, d] - val_a_np[:, d]) ** 2))
    mse_baseline = float(np.mean((mean_action_np[d] - val_a_np[:, d]) ** 2))
    r2 = 1 - mse_pred / mse_baseline
    per_dim_r2.append(r2)
    print(f"{dim_names[d]:12s}{r2:.4f}")

group_r2 = {}
for name, dims in groups.items():
    mse_pred = float(np.mean((val_pred_np[:, dims] - val_a_np[:, dims]) ** 2))
    mse_baseline = float(np.mean((mean_action_np[dims] - val_a_np[:, dims]) ** 2))
    group_r2[name] = 1 - mse_pred / mse_baseline

print("\ngroup           R^2")
for name, r2 in group_r2.items():
    print(f"{name:16s}{r2:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#2e7d32" if "gripper" in n else "#1565c0" if "xyz" in n else "#6a1b9a" for n in groups]
ax.bar(list(group_r2.keys()), list(group_r2.values()), color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("R^2 vs. mean-action baseline")
ax.set_title("Day92: which action dimensions does the encoder actually preserve?")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("outputs/day92_per_dimension_r2.png", dpi=150, bbox_inches="tight")
print("\nsaved outputs/day92_per_dimension_r2.png")

with open("outputs/day92_per_dimension_r2.json", "w") as f:
    json.dump({"checkpoint": args.checkpoint, "per_dim_r2": dict(zip(dim_names, per_dim_r2)), "group_r2": group_r2}, f, indent=2)
