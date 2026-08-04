"""Train the JEPA-style action-conditioned predictor (latent space, no pixels).

Same data, same episode-level train/val split, same horizon as train.py,
so results are directly comparable to the pixel-space experiments.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from jepa_model import JEPAActionModel, normalized_mse_loss, variance_loss

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=10)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--var-weight", type=float, default=5.0, help="weight on the anti-collapse variance term")
args = parser.parse_args()
H = args.horizon

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
LR = 1e-3

with open("data/episode_lengths.json") as f:
    episode_lengths = json.load(f)
episode_ids = sorted(episode_lengths.keys())
rng = np.random.default_rng(0)  # same seed as train.py -- same train/val split
shuffled = rng.permutation(episode_ids)
val_episodes = set(shuffled[:4])
train_episodes = set(shuffled[4:])
print(f"train episodes ({len(train_episodes)}): {sorted(train_episodes)}")
print(f"val episodes ({len(val_episodes)}):   {sorted(val_episodes)}")


def build_pairs(ep_ids):
    all_frame_t, all_action_t, all_frame_t1 = [], [], []
    for ep_id in ep_ids:
        frames = np.load(f"data/episodes/{ep_id}_frames.npy")
        actions = np.load(f"data/episodes/{ep_id}_actions.npy")
        if len(frames) <= H:
            continue
        action_window = np.stack([actions[i : i + H].reshape(-1) for i in range(len(actions) - H)])
        all_frame_t.append(frames[:-H])
        all_action_t.append(action_window)
        all_frame_t1.append(frames[H:])
    return np.concatenate(all_frame_t), np.concatenate(all_action_t), np.concatenate(all_frame_t1)


train_frame_t, train_action_t, train_frame_t1 = build_pairs(train_episodes)
val_frame_t, val_action_t, val_frame_t1 = build_pairs(val_episodes)
print(f"train pairs: {len(train_frame_t)}, val pairs: {len(val_frame_t)}")

action_mean = train_action_t.mean(axis=0)
action_std = train_action_t.std(axis=0) + 1e-6


def to_tensor_batch(frame_t, action_t, frame_t1, idx):
    f = torch.from_numpy(frame_t[idx]).float().permute(0, 3, 1, 2) / 255.0
    a = torch.from_numpy((action_t[idx] - action_mean) / action_std).float()
    f1 = torch.from_numpy(frame_t1[idx]).float().permute(0, 3, 1, 2) / 255.0
    return f.to(DEVICE), a.to(DEVICE), f1.to(DEVICE)


model = JEPAActionModel(action_dim=train_action_t.shape[1]).to(DEVICE)
opt = torch.optim.Adam(list(model.online_encoder.parameters()) + list(model.predictor.parameters()), lr=LR)

history = {"train_loss": [], "val_loss": [], "z_std": []}

for epoch in range(args.epochs):
    model.train()
    perm = np.random.permutation(len(train_frame_t))
    epoch_losses, epoch_zstd = [], []
    for i in range(0, len(perm), BATCH_SIZE):
        batch_idx = perm[i : i + BATCH_SIZE]
        f, a, f1 = to_tensor_batch(train_frame_t, train_action_t, train_frame_t1, batch_idx)
        pred_z, target_z, z_t = model(f, a, f1)
        pred_loss = normalized_mse_loss(pred_z, target_z)
        collapse_penalty = variance_loss(z_t)  # keeps z_t from collapsing to a near-constant vector
        loss = pred_loss + args.var_weight * collapse_penalty
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()
        epoch_losses.append(pred_loss.item())
        epoch_zstd.append(z_t.detach().std(dim=0).mean().item())  # collapse diagnostic

    model.eval()
    val_losses = []
    with torch.no_grad():
        for i in range(0, len(val_frame_t), BATCH_SIZE):
            idx = np.arange(i, min(i + BATCH_SIZE, len(val_frame_t)))
            f, a, f1 = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, idx)
            pred_z, target_z, _ = model(f, a, f1)
            val_losses.append(normalized_mse_loss(pred_z, target_z).item())
    val_loss = float(np.mean(val_losses))
    train_loss = float(np.mean(epoch_losses))
    z_std = float(np.mean(epoch_zstd))
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["z_std"].append(z_std)

    if epoch % 10 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:4d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  z_std {z_std:.4f}")

# --- baselines on held-out episodes, all in the same normalized-MSE latent metric ---
model.eval()
with torch.no_grad():
    idx_all = np.arange(len(val_frame_t))
    losses_real, losses_shuffled, losses_zero, losses_copy_zt = [], [], [], []
    rand_idx = np.random.default_rng(1).permutation(len(val_action_t))
    for i in range(0, len(idx_all), BATCH_SIZE):
        idx = idx_all[i : i + BATCH_SIZE]
        f, a_real, f1 = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, idx)
        shuf_idx = rand_idx[i : i + BATCH_SIZE]
        _, a_shuf, _ = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, shuf_idx)

        z_t = model.online_encoder(f)
        target_z = model.target_encoder(f1)

        pred_real = model.predictor(z_t, a_real)
        pred_shuf = model.predictor(z_t, a_shuf)
        pred_zero = model.predictor(z_t, torch.zeros_like(a_real))

        losses_real.append(normalized_mse_loss(pred_real, target_z).item())
        losses_shuffled.append(normalized_mse_loss(pred_shuf, target_z).item())
        losses_zero.append(normalized_mse_loss(pred_zero, target_z).item())
        losses_copy_zt.append(normalized_mse_loss(z_t, target_z).item())  # "predict no change in latent space"

print()
print(f"val loss -- real action:          {np.mean(losses_real):.4f}")
print(f"val loss -- shuffled action:      {np.mean(losses_shuffled):.4f}")
print(f"val loss -- zero action:          {np.mean(losses_zero):.4f}")
print(f"val loss -- copy z_t as z_t+H:    {np.mean(losses_copy_zt):.4f}  (latent-space 'do nothing' baseline)")

history["val_loss_real_action"] = float(np.mean(losses_real))
history["val_loss_shuffled_action"] = float(np.mean(losses_shuffled))
history["val_loss_zero_action"] = float(np.mean(losses_zero))
history["val_loss_copy_zt_baseline"] = float(np.mean(losses_copy_zt))
history["train_episodes"] = sorted(train_episodes)
history["val_episodes"] = sorted(val_episodes)

with open(f"outputs/history_jepa_h{H}.json", "w") as fp:
    json.dump(history, fp, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val (held-out episodes)")
axes[0].axhline(np.mean(losses_copy_zt), color="gray", linestyle="--", label="copy z_t baseline")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("normalized MSE in latent space")
axes[0].legend()
axes[0].set_title("JEPA-style loss")

axes[1].plot(history["z_std"])
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("mean std of z_t across batch")
axes[1].set_title("Collapse check (should stay well above 0)")

plt.tight_layout()
plt.savefig(f"outputs/loss_curve_jepa_h{H}.png", dpi=120)
print(f"saved outputs/loss_curve_jepa_h{H}.png")

torch.save(model.state_dict(), f"outputs/model_jepa_h{H}.pt")
