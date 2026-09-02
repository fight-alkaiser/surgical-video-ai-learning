"""Day91 (part 2): does real-action sample generation get worse specifically
for actions that are further from the training-set mean (i.e. further from
the "zero" condition, which is exactly that mean in normalized space)?

Rules out or supports the "regression to the mean" hypothesis: the velocity
field may be adequately trained near the centroid of the action distribution
(where "zero" sits by construction) but less reliable -- more integration
error accumulated over the ODE sample -- for atypical, larger-magnitude real
actions, even though the local velocity estimate (paired_loss) is still
locally accurate at those conditions.

For each held-out example: compute ||real action|| (distance from the
training mean, in normalized units) and the per-example sample-generation
error under both the real and zero conditions, then check whether error
grows with action magnitude for "real" but not for "zero" (zero is constant
across examples, so it's a natural control for "this transition was just
harder to predict regardless of action").
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from cfm_model import CFMActionModel, per_example_normalized_mse

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=10)
parser.add_argument("--checkpoint", type=str, default="outputs/model_cfm_h10_noise_n200_seed2.pt")
parser.add_argument("--num-samples", type=int, default=16)
parser.add_argument("--sample-steps", type=int, default=16)
args = parser.parse_args()
H = args.horizon

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32

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

action_dim_per_step = train_action_t.shape[-1]
action_mean = train_action_t.reshape(-1, action_dim_per_step).mean(axis=0)
action_std = train_action_t.reshape(-1, action_dim_per_step).std(axis=0) + 1e-6


def to_tensor_batch(frame_t, action_t, frame_t1, idx):
    f = torch.from_numpy(frame_t[idx]).float().permute(0, 3, 1, 2) / 255.0
    a = torch.from_numpy((action_t[idx] - action_mean) / action_std).float()  # (B, H, D), normalized
    f1 = torch.from_numpy(frame_t1[idx]).float().permute(0, 3, 1, 2) / 255.0
    return f.to(DEVICE), a.to(DEVICE), f1.to(DEVICE)


model = CFMActionModel(action_dim_per_step=action_dim_per_step, horizon=H).to(DEVICE)
model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
model.eval()

idx_all = np.arange(len(val_frame_t))
action_norms, real_errors, zero_errors = [], [], []

with torch.no_grad():
    for i in range(0, len(idx_all), BATCH_SIZE):
        idx = idx_all[i : i + BATCH_SIZE]
        f, a_real, f1 = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, idx)
        a_zero = torch.zeros_like(a_real)

        # ||real action - mean|| per example, in normalized units, averaged over the H-step window
        norm = a_real.reshape(a_real.shape[0], -1).norm(dim=-1) / (H ** 0.5)
        action_norms.append(norm.cpu().numpy())

        target_z = model.target_encoder(f1)
        z_t = model.online_encoder(f)

        for a, err_list in [(a_real, real_errors), (a_zero, zero_errors)]:
            samples = torch.stack(
                [model.sample(z_t, a, steps=args.sample_steps, source="noise") for _ in range(args.num_samples)], dim=0
            )
            per_sample_dist = torch.stack([per_example_normalized_mse(s, target_z) for s in samples], dim=0)
            best_of_n = per_sample_dist.min(dim=0).values  # per-example best-of-N error
            err_list.append(best_of_n.cpu().numpy())

action_norms = np.concatenate(action_norms)
real_errors = np.concatenate(real_errors)
zero_errors = np.concatenate(zero_errors)

r_real = float(np.corrcoef(action_norms, real_errors)[0, 1])
r_zero = float(np.corrcoef(action_norms, zero_errors)[0, 1])
print(f"correlation(||real action - mean||, real-condition error):  r = {r_real:.4f}")
print(f"correlation(||real action - mean||, zero-condition error):  r = {r_zero:.4f}  (control -- zero ignores the action)")

# quartile breakdown for a clearer picture than a single correlation coefficient
quartiles = np.quantile(action_norms, [0.25, 0.5, 0.75])
bins = np.digitize(action_norms, quartiles)
print("\naction-magnitude quartile -> mean best_of_n_error:")
print(f"{'quartile':10s}{'n':>6s}{'real':>10s}{'zero':>10s}")
quartile_stats = []
for q in range(4):
    mask = bins == q
    real_mean = float(real_errors[mask].mean())
    zero_mean = float(zero_errors[mask].mean())
    quartile_stats.append({"quartile": q, "n": int(mask.sum()), "real_mean_error": real_mean, "zero_mean_error": zero_mean})
    print(f"Q{q+1:<9d}{mask.sum():>6d}{real_mean:>10.4f}{zero_mean:>10.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(4)
width = 0.35
ax.bar(x - width / 2, [s["real_mean_error"] for s in quartile_stats], width, label="real", color="#2e7d32")
ax.bar(x + width / 2, [s["zero_mean_error"] for s in quartile_stats], width, label="zero", color="#9e9e9e")
ax.set_xticks(x)
ax.set_xticklabels(["Q1\n(smallest action)", "Q2", "Q3", "Q4\n(largest action)"])
ax.set_ylabel("mean best_of_n_error")
ax.set_title(f"Day91: sample-generation error vs. action magnitude quartile\ncorr(action norm, error): real r={r_real:.3f}, zero r={r_zero:.3f}")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/day91_action_magnitude_vs_bias.png", dpi=150, bbox_inches="tight")
print("\nsaved outputs/day91_action_magnitude_vs_bias.png")

with open("outputs/day91_action_magnitude_vs_bias.json", "w") as f:
    json.dump(
        {"checkpoint": args.checkpoint, "r_real": r_real, "r_zero": r_zero, "quartiles": quartile_stats},
        f,
        indent=2,
    )
