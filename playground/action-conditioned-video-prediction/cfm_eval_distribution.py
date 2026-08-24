"""Day82: directly inspect the shape of the real- vs. zero-conditioned sample
distributions, instead of proxying via best_of_n_error or ODE step count
(Day81 showed step count wasn't the explanation).

Two things, both re-using the Day80 checkpoints without retraining:

1. num_samples sweep -- best_of_n_error is a "best of N random draws" metric,
   so its outcome depends on N. If `real`'s distribution is genuinely wider
   (higher variance) than `zero`'s, few draws (N=8) would unfairly favor the
   tighter cluster; more draws should let `real` catch up if that's the whole
   story. Draw a large pool of samples once per condition and recompute
   best-of-N by taking a running min over N = 8, 16, 32, 64, 128, 256.

2. bias/variance decomposition -- for each condition, compute:
     bias^2      = || mean(samples) - target ||^2   (is the cluster centered
                    near the truth?)
     variance    = mean || sample - mean(samples) ||^2  (how spread out is
                    the cluster?)
   These sum to (approximately) the expected squared error of a single
   random draw -- separating "is the model's average guess accurate" from
   "how much does a single sample bounce around."

Also dumps a 2D PCA projection of one example pair's samples (real vs
shuffled vs zero, target marked) for a qualitative look at the actual shapes.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from cfm_model import CFMActionModel

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gated", action="store_true")
parser.add_argument("--pool-size", type=int, default=256, help="total samples drawn per condition per pair")
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--n-pairs", type=int, default=64, help="number of val (z_t, action) pairs to evaluate on")
args = parser.parse_args()
H = args.horizon

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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
        action_window = np.stack([actions[i : i + H].reshape(-1) for i in range(len(actions) - H)])
        all_frame_t.append(frames[:-H])
        all_action_t.append(action_window)
        all_frame_t1.append(frames[H:])
    return np.concatenate(all_frame_t), np.concatenate(all_action_t), np.concatenate(all_frame_t1)


train_frame_t, train_action_t, _ = build_pairs(train_episodes)
action_mean = train_action_t.mean(axis=0)
action_std = train_action_t.std(axis=0) + 1e-6
val_frame_t, val_action_t, val_frame_t1 = build_pairs(val_episodes)


def to_tensor_batch(frame_t, action_t, frame_t1, idx):
    f = torch.from_numpy(frame_t[idx]).float().permute(0, 3, 1, 2) / 255.0
    a = torch.from_numpy((action_t[idx] - action_mean) / action_std).float()
    f1 = torch.from_numpy(frame_t1[idx]).float().permute(0, 3, 1, 2) / 255.0
    return f.to(DEVICE), a.to(DEVICE), f1.to(DEVICE)


tag = f"h{H}_noise_n{len(episode_ids)}_seed{args.seed}" + ("_gated" if args.gated else "")
model = CFMActionModel(action_dim=train_action_t.shape[1], gated=args.gated).to(DEVICE)
model.load_state_dict(torch.load(f"outputs/model_cfm_{tag}.pt", map_location=DEVICE))
model.eval()

rng_pairs = np.random.default_rng(2)
idx = rng_pairs.choice(len(val_frame_t), size=min(args.n_pairs, len(val_frame_t)), replace=False)
rand_idx = np.random.default_rng(1).permutation(len(val_action_t))[: len(idx)]

f, a_real, f1 = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, idx)
_, a_shuf, _ = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, rand_idx)
with torch.no_grad():
    target_z = model.target_encoder(f1)
    z_t = model.online_encoder(f)

conditions = {"real": a_real, "shuffled": a_shuf, "zero": torch.zeros_like(a_real)}
n_list = [8, 16, 32, 64, 128, 256]
n_list = [n for n in n_list if n <= args.pool_size]

results = {"best_of_n": {n: {} for n in n_list}, "bias_variance": {}}
sample_pools = {}

target_z_n = F.normalize(target_z, dim=-1)  # same normalization convention as per_example_normalized_mse elsewhere

with torch.no_grad():
    for cond, a in conditions.items():
        pool = torch.stack([model.sample(z_t, a, steps=args.steps, source="noise") for _ in range(args.pool_size)], dim=0)
        sample_pools[cond] = pool  # (pool_size, n_pairs, embed_dim) -- raw, unnormalized (kept for the PCA plot)
        pool_n = F.normalize(pool, dim=-1)  # normalized, for all distance/bias/variance numbers below

        dist_to_target = ((pool_n - target_z_n[None]) ** 2).sum(dim=-1)  # (pool_size, n_pairs)
        running_min = torch.cummin(dist_to_target, dim=0).values  # running best-of-k over the pool
        for n in n_list:
            results["best_of_n"][n][cond] = running_min[n - 1].mean().item()

        mean_sample_n = pool_n.mean(dim=0)  # (n_pairs, embed_dim) -- mean of normalized samples
        bias_sq = ((mean_sample_n - target_z_n) ** 2).sum(dim=-1).mean().item()
        variance = ((pool_n - mean_sample_n[None]) ** 2).sum(dim=-1).mean().item()
        results["bias_variance"][cond] = {"bias_sq": bias_sq, "variance": variance, "bias_sq_plus_variance": bias_sq + variance}

print("=== best-of-N (as N grows, using the same 256-sample pool) ===")
for n in n_list:
    print(f"N={n:4d}  " + "   ".join(f"{c}={results['best_of_n'][n][c]:.4f}" for c in conditions))

print("\n=== bias/variance decomposition ===")
for cond in conditions:
    bv = results["bias_variance"][cond]
    print(f"{cond:>10s}  bias^2={bv['bias_sq']:.4f}  variance={bv['variance']:.4f}  sum={bv['bias_sq_plus_variance']:.4f}")

with open(f"outputs/history_cfm_distribution_{tag}.json", "w") as fp:
    json.dump(results, fp, indent=2)
print(f"\nsaved outputs/history_cfm_distribution_{tag}.json")

# --- 2D PCA visualization for one example pair ---
example = 0
colors = {"real": "#2e7d32", "shuffled": "#c62828", "zero": "#9e9e9e"}
all_points = torch.cat([sample_pools[c][:, example] for c in conditions], dim=0)  # (3*pool_size, embed_dim)
all_points = all_points - all_points.mean(dim=0, keepdim=True)
U, S, V = torch.pca_lowrank(all_points, q=2)
proj = all_points @ V[:, :2]
proj = proj.cpu().numpy()

raw_points_mean = torch.cat([sample_pools[c][:, example] for c in conditions], dim=0).mean(dim=0, keepdim=True)
target_centered = target_z[example : example + 1] - raw_points_mean
target_proj = (target_centered @ V[:, :2]).cpu().numpy()

fig, ax = plt.subplots(figsize=(6, 6))
n = args.pool_size
for i, cond in enumerate(conditions):
    pts = proj[i * n : (i + 1) * n]
    ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.35, color=colors[cond], label=cond)
ax.scatter(target_proj[:, 0], target_proj[:, 1], marker="*", s=400, color="black", label="true target", zorder=5)
ax.set_title(f"Day82: {args.pool_size} samples per condition, one example pair\n(PCA projection of the 64-dim latent)")
ax.legend()
plt.tight_layout()
plt.savefig(f"outputs/day82_sample_distribution_pca_{tag}.png", dpi=150)
print(f"saved outputs/day82_sample_distribution_pca_{tag}.png")
