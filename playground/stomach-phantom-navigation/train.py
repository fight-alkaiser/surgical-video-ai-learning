"""Day106: train the CFM action-conditioned latent predictor and evaluate
real vs. shuffled vs. zero action, same methodology as
../action-conditioned-video-prediction/'s Day78/96-98 (paired_loss +
best_of_n_error, no biased single-sample-vs-target comparison -- see that
project's README, Day78 section, for why).
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from cfm_model import CFMActionModel, per_example_normalized_mse

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=20, help="frames ahead to predict (data is 20fps, so 20 = 1 second)")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--sample-steps", type=int, default=16)
parser.add_argument("--num-samples", type=int, default=8)
parser.add_argument(
    "--action-mode",
    choices=["flatten", "sequence"],
    default="flatten",
    help="Day110: 'flatten' (Day106-109 default) concatenates the (H, action_dim) window into one "
    "vector. 'sequence' runs it through a small GRU instead, giving the network the step order for "
    "free -- this task's action is an instantaneous motor velocity, so 'where things ended up' "
    "requires integrating the window over time, which flattening doesn't make easy to recover.",
)
args = parser.parse_args()
H = args.horizon
torch.manual_seed(args.seed)
np.random.seed(args.seed)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = args.batch_size

with open("data/episode_lengths.json") as f:
    episode_lengths = json.load(f)
episode_ids = sorted(episode_lengths.keys())

# held-out split: fixed permutation of whatever episodes are on disk, ~20% val.
# Unlike ../action-conditioned-video-prediction/, there's no "original N"
# precedent to preserve here yet -- this is this project's first run.
rng = np.random.default_rng(0)
shuffled = rng.permutation(episode_ids)
n_val = max(1, len(episode_ids) // 5)
val_episodes = set(shuffled[:n_val])
train_episodes = set(episode_ids) - val_episodes
print(f"train episodes ({len(train_episodes)}): {sorted(train_episodes)}")
print(f"val episodes ({len(val_episodes)}):   {sorted(val_episodes)}")


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
    a = torch.from_numpy((action_t[idx] - action_mean) / action_std).float()
    f1 = torch.from_numpy(frame_t1[idx]).float().permute(0, 3, 1, 2) / 255.0
    return f.to(DEVICE), a.to(DEVICE), f1.to(DEVICE)


tag = f"h{H}_n{len(episode_ids)}_seed{args.seed}"
if args.action_mode != "flatten":
    tag += f"_{args.action_mode}"
model = CFMActionModel(action_dim_per_step=action_dim_per_step, horizon=H, action_mode=args.action_mode).to(DEVICE)
trainable_params = list(model.velocity.parameters())
if model.action_encoder is not None:
    trainable_params += list(model.action_encoder.parameters())
opt = torch.optim.Adam(trainable_params, lr=args.lr)

history = {"train_loss": [], "val_loss": []}
best_val_loss = float("inf")
best_epoch = -1
best_state = None

for epoch in range(args.epochs):
    model.train()
    perm = np.random.permutation(len(train_frame_t))
    epoch_losses = []
    for i in range(0, len(perm), BATCH_SIZE):
        batch_idx = perm[i : i + BATCH_SIZE]
        f, a, f1 = to_tensor_batch(train_frame_t, train_action_t, train_frame_t1, batch_idx)
        loss, _ = model.training_step(f, a, f1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(3):
            for i in range(0, len(val_frame_t), BATCH_SIZE):
                idx = np.arange(i, min(i + BATCH_SIZE, len(val_frame_t)))
                f, a, f1 = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, idx)
                loss, _ = model.training_step(f, a, f1)
                val_losses.append(loss.item())
    val_loss = float(np.mean(val_losses))
    train_loss = float(np.mean(epoch_losses))
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    smoothed = float(np.mean(history["val_loss"][-5:]))
    if smoothed < best_val_loss:
        best_val_loss = smoothed
        best_epoch = epoch
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:4d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}")

print(f"\nbest smoothed val_loss {best_val_loss:.4f} at epoch {best_epoch} (of {args.epochs}); restoring that checkpoint")
model.load_state_dict(best_state)
history["best_epoch"] = best_epoch
history["best_val_loss"] = best_val_loss

# --- evaluation: real vs shuffled vs zero action, paired_loss + best_of_n_error ---
model.eval()
with torch.no_grad():
    idx_all = np.arange(len(val_frame_t))
    rand_idx = np.random.default_rng(1).permutation(len(val_action_t))
    N_LOSS_REPEATS = 4

    results = {}
    for condition in ["real", "shuffled", "zero"]:
        paired_losses, best_of_n_errors = [], []
        for i in range(0, len(idx_all), BATCH_SIZE):
            idx = idx_all[i : i + BATCH_SIZE]
            f, a_real, f1 = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, idx)

            if condition == "real":
                a = a_real
            elif condition == "zero":
                a = torch.zeros_like(a_real)
            else:
                shuf_idx = rand_idx[i : i + BATCH_SIZE]
                _, a, _ = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, shuf_idx)

            for _ in range(N_LOSS_REPEATS):
                loss, _ = model.training_step(f, a, f1)
                paired_losses.append(loss.item())

            target_z = model.target_encoder(f1)
            z_t = model.online_encoder(f)
            samples = torch.stack(
                [model.sample(z_t, a, steps=args.sample_steps) for _ in range(args.num_samples)], dim=0
            )
            per_sample_dist = torch.stack([per_example_normalized_mse(s, target_z) for s in samples], dim=0)
            best_of_n_errors.append(per_sample_dist.min(dim=0).values.mean().item())

        results[condition] = {
            "paired_loss": float(np.mean(paired_losses)),
            "best_of_n_error": float(np.mean(best_of_n_errors)),
        }

print()
for condition, r in results.items():
    parts = [f"{k}: {v:.4f}" for k, v in r.items()]
    print(f"{condition:>10s} -- " + "   ".join(parts))

history["eval"] = results
history["train_episodes"] = sorted(train_episodes)
history["val_episodes"] = sorted(val_episodes)

with open(f"outputs/history_cfm_{tag}.json", "w") as fp:
    json.dump(history, fp, indent=2)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history["train_loss"], label="train")
ax.plot(history["val_loss"], label="val (held-out episodes)")
ax.axvline(best_epoch, color="gray", linestyle="--", label=f"best_epoch={best_epoch}")
ax.set_xlabel("epoch")
ax.set_ylabel("CFM velocity MSE loss")
ax.legend()
ax.set_title(f"Day106: stomach-phantom CFM training ({tag})")
plt.tight_layout()
plt.savefig(f"outputs/loss_curve_cfm_{tag}.png", dpi=120)
print(f"saved outputs/loss_curve_cfm_{tag}.png")

torch.save(model.state_dict(), f"outputs/model_cfm_{tag}.pt")
