"""Day81: does the paired_loss / best_of_n_error disagreement (Day80) come from
ODE integration error, not a real difference in what the model learned?

Day80 found that at 100 episodes, `real` action beats `zero` on paired_loss
(the direct, no-sampling metric) but loses to `zero` on best_of_n_error (which
requires actually integrating the ODE from noise with a 16-step Euler solver).
Euler is a first-order method: its error depends on how curved the velocity
field's path is, not just on how accurate the field is pointwise. If `real`
action's paths are less straight than `zero`'s, few-step Euler would penalize
`real` specifically -- a sampling artifact, not evidence the model prefers
`zero`. This script re-evaluates the already-trained Day80 checkpoints at
several step counts, without retraining, to check whether the gap closes as
steps increase.
"""

import argparse
import json

import numpy as np
import torch

from cfm_model import CFMActionModel, per_example_normalized_mse

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gated", action="store_true")
parser.add_argument("--steps", type=int, nargs="+", default=[16, 32, 64, 128])
parser.add_argument("--num-samples", type=int, default=8)
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


# action_mean/std must match training: computed over the same train set the checkpoint saw
train_episodes = set(episode_ids) - val_episodes
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

idx_all = np.arange(len(val_frame_t))
rand_idx = np.random.default_rng(1).permutation(len(val_action_t))

results = {}
for steps in args.steps:
    results[steps] = {}
    for condition in ["real", "shuffled", "zero"]:
        best_of_n_errors, sample_spreads = [], []
        with torch.no_grad():
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

                target_z = model.target_encoder(f1)
                z_t = model.online_encoder(f)
                samples = torch.stack(
                    [model.sample(z_t, a, steps=steps, source="noise") for _ in range(args.num_samples)], dim=0
                )
                per_sample_dist = torch.stack([per_example_normalized_mse(s, target_z) for s in samples], dim=0)
                best_of_n_errors.append(per_sample_dist.min(dim=0).values.mean().item())
                sample_spreads.append(samples.std(dim=0).mean().item())
        results[steps][condition] = {
            "best_of_n_error": float(np.mean(best_of_n_errors)),
            "sample_spread": float(np.mean(sample_spreads)),
        }
    print(f"steps={steps:4d}  " + "   ".join(f"{c}={results[steps][c]['best_of_n_error']:.4f}" for c in ["real", "shuffled", "zero"]))

with open(f"outputs/history_cfm_step_sweep_{tag}.json", "w") as fp:
    json.dump(results, fp, indent=2)
print(f"saved outputs/history_cfm_step_sweep_{tag}.json")
