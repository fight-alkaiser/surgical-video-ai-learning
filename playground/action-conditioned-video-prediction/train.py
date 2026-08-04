"""Train the toy action-conditioned next-frame predictor on multiple Open-H episodes.

Train/val split is by EPISODE, not by time within one episode -- validation
episodes are never seen during training at all. This is a stricter and more
honest test than the single-episode temporal split used in the first attempt.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from model import ActionConditionedPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=10, help="predict frame_{t+H} from frame_t")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument(
    "--weighted-loss",
    action="store_true",
    help="weight the MSE by a precomputed motion-saliency map (data/motion_weight_map.npy), "
    "emphasizing pixels where the instrument actually moves instead of the whole frame equally",
)
args = parser.parse_args()
H = args.horizon
tag = f"h{H}_lateFiLM" + ("_weighted" if args.weighted_loss else "")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
LR = 1e-3

pixel_weight = None
if args.weighted_loss:
    motion_map = np.load("data/motion_weight_map.npy")  # (64, 64)
    motion_map = motion_map / motion_map.mean()  # normalize so avg weight is 1 (comparable scale to unweighted MSE)
    pixel_weight = torch.from_numpy(motion_map).float().to(DEVICE)  # (64, 64), broadcasts over (B, C, H, W)

with open("data/episode_lengths.json") as f:
    episode_lengths = json.load(f)
episode_ids = sorted(episode_lengths.keys())

rng = np.random.default_rng(0)
shuffled = rng.permutation(episode_ids)
n_val_ep = 4
val_episodes = set(shuffled[:n_val_ep])
train_episodes = set(shuffled[n_val_ep:])
print(f"train episodes ({len(train_episodes)}): {sorted(train_episodes)}")
print(f"val episodes ({len(val_episodes)}):   {sorted(val_episodes)}")


def build_pairs(ep_ids):
    """Build (frame_t, action_window, frame_{t+H}) pairs, never crossing episode boundaries."""
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
    target = torch.from_numpy(frame_t1[idx]).float().permute(0, 3, 1, 2) / 255.0
    return f.to(DEVICE), a.to(DEVICE), target.to(DEVICE)


model = ActionConditionedPredictor(action_dim=train_action_t.shape[1]).to(DEVICE)
# Day62 finding: weight_decay on the action_embedder drove it to exactly zero (the action
# pathway contributes little to the loss, so decay wins). Exclude it from weight_decay so
# the action pathway isn't penalized just for existing.
action_embedder_params = list(model.action_embedder.parameters())
other_params = [p for p in model.parameters() if not any(p is q for q in action_embedder_params)]
opt = torch.optim.Adam(
    [
        {"params": other_params, "weight_decay": 1e-4},
        {"params": action_embedder_params, "weight_decay": 0.0},
    ],
    lr=LR,
)
unweighted_loss_fn = nn.MSELoss()


def loss_fn(pred, target):
    if pixel_weight is None:
        return unweighted_loss_fn(pred, target)
    sq_err = (pred - target) ** 2  # (B, C, H, W)
    return (sq_err * pixel_weight).mean()

history = {"train_loss": [], "val_loss": []}

for epoch in range(args.epochs):
    model.train()
    perm = np.random.permutation(len(train_frame_t))
    epoch_losses = []
    for i in range(0, len(perm), BATCH_SIZE):
        batch_idx = perm[i : i + BATCH_SIZE]
        f, a, target = to_tensor_batch(train_frame_t, train_action_t, train_frame_t1, batch_idx)
        pred = model(f, a)
        loss = loss_fn(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for i in range(0, len(val_frame_t), BATCH_SIZE):
            idx = np.arange(i, min(i + BATCH_SIZE, len(val_frame_t)))
            f, a, target = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, idx)
            val_pred = model(f, a)
            val_losses.append(loss_fn(val_pred, target).item())
    val_loss = float(np.mean(val_losses))

    train_loss = float(np.mean(epoch_losses))
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    if epoch % 10 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:4d}  train_loss {train_loss:.5f}  val_loss {val_loss:.5f}")

# copy-last-frame baseline, computed on the same val set (both weighted-as-trained and
# plain unweighted, so weighted runs stay comparable to the earlier unweighted experiments)
baseline_losses, baseline_losses_unweighted = [], []
model_losses_unweighted = []
for i in range(0, len(val_frame_t), BATCH_SIZE):
    idx = np.arange(i, min(i + BATCH_SIZE, len(val_frame_t)))
    f, a, target = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, idx)
    baseline_losses.append(loss_fn(f, target).item())
    baseline_losses_unweighted.append(unweighted_loss_fn(f, target).item())
    with torch.no_grad():
        val_pred = model(f, a)
    model_losses_unweighted.append(unweighted_loss_fn(val_pred, target).item())
baseline_loss = float(np.mean(baseline_losses))
baseline_loss_unweighted = float(np.mean(baseline_losses_unweighted))
model_loss_unweighted = float(np.mean(model_losses_unweighted))
print(f"copy-last-frame baseline val_loss (as trained, {'weighted' if pixel_weight is not None else 'unweighted'}): {baseline_loss:.5f}")
print(f"copy-last-frame baseline val_loss (plain unweighted): {baseline_loss_unweighted:.5f}")
print(f"model val_loss (plain unweighted): {model_loss_unweighted:.5f}")
history["copy_baseline_val_loss"] = baseline_loss
history["copy_baseline_val_loss_unweighted"] = baseline_loss_unweighted
history["model_val_loss_unweighted"] = model_loss_unweighted
history["train_episodes"] = sorted(train_episodes)
history["val_episodes"] = sorted(val_episodes)

with open(f"outputs/history_{tag}.json", "w") as fp:
    json.dump(history, fp, indent=2)

plt.figure(figsize=(6, 4))
plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"], label="val (held-out episodes)")
plt.axhline(baseline_loss, color="gray", linestyle="--", label="copy-last-frame baseline")
plt.xlabel("epoch")
plt.ylabel("MSE loss" + (" (motion-weighted)" if pixel_weight is not None else ""))
plt.legend()
plt.title(f"Action-conditioned prediction, horizon={H}, {len(train_episodes)} train / {len(val_episodes)} val episodes")
plt.tight_layout()
plt.savefig(f"outputs/loss_curve_{tag}.png", dpi=120)
print(f"saved outputs/loss_curve_{tag}.png")

torch.save(model.state_dict(), f"outputs/model_{tag}.pt")

# qualitative check on a held-out episode
model.eval()
with torch.no_grad():
    idx = np.linspace(0, len(val_frame_t) - 1, 6).astype(int)
    f, a, target = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, idx)
    pred = model(f, a).cpu().permute(0, 2, 3, 1).numpy()
target_np = target.cpu().permute(0, 2, 3, 1).numpy()

fig, axes = plt.subplots(2, len(idx), figsize=(2 * len(idx), 4))
for col in range(len(idx)):
    axes[0, col].imshow(target_np[col])
    axes[0, col].axis("off")
    axes[1, col].imshow(pred[col])
    axes[1, col].axis("off")
plt.tight_layout()
plt.savefig(f"outputs/qualitative_comparison_{tag}.png", dpi=120)
print(f"saved outputs/qualitative_comparison_{tag}.png")
