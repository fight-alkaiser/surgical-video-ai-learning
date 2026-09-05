"""Day94: stop chasing the val_loss oscillation's noise, and directly ask the
question that actually matters -- is the trained encoder's representation
useful, independent of whether the predictor's training loss looks clean.

Reuses this series' own probe methodology (../action-conditioned-video-
prediction/probe_action_from_latents.py, Day91): freeze the encoder, train a
small MLP probe from its output to the actual per-frame action, and compare
against a mean-action baseline. The twist here is the second comparison this
script adds: a *randomly initialized* (untrained) encoder of the identical
architecture, probed the same way. If the trained encoder doesn't clearly
beat the random one, whatever signal a probe finds is coming from the
architecture's fixed random projection or basic pixel statistics, not from
anything I-JEPA training actually taught it.

Per I-JEPA's own evaluation convention, this uses the *context* encoder
(the one that becomes the deployed representation) applied to the *full,
unmasked* image, pooled (mean over patches) to one vector per frame.
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ijepa_model import IJEPAModel, patchify, NUM_PATCHES

AC_DATA_DIR = "../action-conditioned-video-prediction/data"

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="outputs/model_ijepa_seed0_lr0.001_ema0.996_clip1.0.pt")
parser.add_argument("--probe-epochs", type=int, default=50)
args = parser.parse_args()

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64

with open(f"{AC_DATA_DIR}/episode_lengths.json") as f:
    episode_lengths = json.load(f)
episode_ids = sorted(episode_lengths.keys())
original_20 = [f"episode_{i:06d}" for i in range(20)]
rng = np.random.default_rng(0)
shuffled = rng.permutation(original_20)
val_episodes = set(shuffled[:4])
train_episodes = set(episode_ids) - val_episodes


def load_frames_and_actions(ep_ids):
    """Actions here are the *instantaneous* per-frame action (actions[i] for
    frame i) -- a single-timestep signal, since there's no horizon in this
    project's single-image setup. The action array has one fewer row than
    frames in this project's convention (see prepare_data.py upstream), so
    align to the shorter length."""
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
    return (torch.from_numpy(frames[idx]).float().permute(0, 3, 1, 2) / 255.0).to(DEVICE)


def encode_all(model, frames):
    reps = []
    with torch.no_grad():
        for i in range(0, len(frames), BATCH_SIZE):
            idx = np.arange(i, min(i + BATCH_SIZE, len(frames)))
            f = to_tensor(frames, idx)
            patches = patchify(f)
            pos = torch.arange(NUM_PATCHES, device=DEVICE).unsqueeze(0).expand(f.shape[0], -1)
            tokens = model.context_encoder(patches, pos)  # (B, 64, embed_dim)
            reps.append(tokens.mean(dim=1).cpu())  # mean-pool over patches
    return torch.cat(reps)


class Probe(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def train_and_eval_probe(train_rep, train_act, val_rep, val_act, label):
    probe = Probe(train_rep.shape[-1], train_act.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    tr, ta = train_rep.to(DEVICE), train_act.to(DEVICE)
    vr, va = val_rep.to(DEVICE), val_act.to(DEVICE)
    best_val = float("inf")
    for epoch in range(args.probe_epochs):
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
    print(f"  [{label}] best val_mse: {best_val:.4f}")
    return best_val


train_act_t = torch.from_numpy((train_actions - action_mean) / action_std).float()
val_act_t = torch.from_numpy((val_actions - action_mean) / action_std).float()
mean_baseline_mse = F.mse_loss(train_act_t.mean(dim=0, keepdim=True).expand_as(val_act_t), val_act_t).item()
print(f"mean-action baseline val_mse: {mean_baseline_mse:.4f}")

print("\n--- trained encoder ---")
trained_model = IJEPAModel().to(DEVICE)
trained_model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
trained_model.eval()
train_rep_trained = encode_all(trained_model, train_frames)
val_rep_trained = encode_all(trained_model, val_frames)
trained_mse = train_and_eval_probe(train_rep_trained, train_act_t, val_rep_trained, val_act_t, "trained")

print("\n--- random (untrained) encoder, identical architecture ---")
torch.manual_seed(123)
random_model = IJEPAModel().to(DEVICE)
random_model.eval()
train_rep_random = encode_all(random_model, train_frames)
val_rep_random = encode_all(random_model, val_frames)
random_mse = train_and_eval_probe(train_rep_random, train_act_t, val_rep_random, val_act_t, "random")

print("\n=== summary ===")
print(f"mean-action baseline: {mean_baseline_mse:.4f}")
print(f"random encoder:       {random_mse:.4f}  (R^2 vs mean: {1 - random_mse / mean_baseline_mse:.4f})")
print(f"trained encoder:      {trained_mse:.4f}  (R^2 vs mean: {1 - trained_mse / mean_baseline_mse:.4f})")
print(f"\ntrained vs random improvement: {1 - trained_mse / random_mse:.4f}")

with open("outputs/day94_probe_results.json", "w") as f:
    json.dump(
        {
            "checkpoint": args.checkpoint,
            "mean_baseline_mse": mean_baseline_mse,
            "random_encoder_mse": random_mse,
            "trained_encoder_mse": trained_mse,
            "r2_trained_vs_mean": 1 - trained_mse / mean_baseline_mse,
            "r2_random_vs_mean": 1 - random_mse / mean_baseline_mse,
        },
        f,
        indent=2,
    )
