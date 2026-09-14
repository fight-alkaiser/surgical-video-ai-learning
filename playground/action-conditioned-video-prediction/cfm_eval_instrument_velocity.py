"""Day103 (continued): Day103's first result showed the real-vs-zero
bias^2 gap does NOT grow with net instrument displacement over the H-step
window -- if anything it shrinks (seed0 clearly, seed1 mildly). One
hypothesis (raised by the project owner): net displacement isn't the same
as how *fast* the instrument moved within the window. A pair could cover a
lot of ground steadily (low peak speed) or dart back and forth (high peak
speed, similar net displacement) -- if the model can't track fast
instantaneous motion between single frames, peak per-step speed should
predict error better than net displacement does, for both conditions,
without needing to touch the target frame at all (unlike time-averaging the
frames, which would have changed what "correct" means and confounded the
comparison -- deliberately avoided here).

Per pair, computes two motion statistics from Day102's detector:
  net_motion  = || centroid(t+H) - centroid(t) ||           (Day103's metric)
  peak_speed  = max over t'=t..t+H-1 of || centroid(t'+1) - centroid(t') ||
And per-pair squared bias (mean-of-32-sample prediction vs. target, in the
same normalized latent space as cfm_eval_distribution.py), for both real
and zero action conditioning. Reports the Pearson correlation of each
motion statistic against per-pair error, for both conditions -- if
peak_speed correlates with error more strongly than net_motion does, that
points at instantaneous tracking rather than cumulative displacement.
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from cfm_model import CFMActionModel
from instrument_detector import instrument_mask

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pool-size", type=int, default=32)
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--n-pairs", type=int, default=256)
parser.add_argument("--action-mode", default="flatten")
parser.add_argument("--encoder-type", default="pretrained_resnet18")
parser.add_argument("--n-episodes", type=int, default=200)
args = parser.parse_args()
H = args.horizon

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with open("data/episode_lengths.json") as f:
    episode_lengths = json.load(f)
episode_ids = sorted(episode_lengths.keys())
original_20 = [f"episode_{i:06d}" for i in range(20)]
rng = np.random.default_rng(0)
shuffled = rng.permutation(original_20)
val_episodes = sorted(shuffled[:4])
train_episodes = sorted(set(episode_ids) - set(shuffled[:4]))


def centroid_track(mask_ep: np.ndarray) -> np.ndarray:
    T, h, w = mask_ep.shape
    centroids = np.full((T, 2), [h / 2, w / 2], dtype=np.float32)
    last = None
    for t in range(T):
        ys, xs = np.nonzero(mask_ep[t])
        if len(ys) > 0:
            last = np.array([ys.mean(), xs.mean()], dtype=np.float32)
        if last is not None:
            centroids[t] = last
    return centroids


def build_pairs_with_motion(ep_ids):
    all_frame_t, all_action_t, all_frame_t1, all_net, all_peak = [], [], [], [], []
    for ep_id in ep_ids:
        frames = np.load(f"data/episodes/{ep_id}_frames.npy")
        actions = np.load(f"data/episodes/{ep_id}_actions.npy")
        if len(frames) <= H:
            continue
        centroids = centroid_track(instrument_mask(frames))
        step_speed = np.linalg.norm(centroids[1:] - centroids[:-1], axis=1)  # (T-1,)
        net = np.linalg.norm(centroids[H:] - centroids[:-H], axis=1)  # (T-H,)
        # peak per-step speed within each [t, t+H) window
        peak = np.stack([step_speed[t : t + H].max() for t in range(len(frames) - H)])
        action_window = np.stack([actions[i : i + H] for i in range(len(actions) - H)])
        all_frame_t.append(frames[:-H])
        all_action_t.append(action_window)
        all_frame_t1.append(frames[H:])
        all_net.append(net)
        all_peak.append(peak)
    return (
        np.concatenate(all_frame_t),
        np.concatenate(all_action_t),
        np.concatenate(all_frame_t1),
        np.concatenate(all_net),
        np.concatenate(all_peak),
    )


train_frame_t, train_action_t, _, _, _ = build_pairs_with_motion(train_episodes)
action_dim_per_step = train_action_t.shape[-1]
action_mean = train_action_t.reshape(-1, action_dim_per_step).mean(axis=0)
action_std = train_action_t.reshape(-1, action_dim_per_step).std(axis=0) + 1e-6
val_frame_t, val_action_t, val_frame_t1, val_net, val_peak = build_pairs_with_motion(val_episodes)
print(f"val pairs: {len(val_frame_t)}")
print(f"net_motion:  range [{val_net.min():.2f}, {val_net.max():.2f}], corr(net, peak) = {np.corrcoef(val_net, val_peak)[0,1]:.3f}")
print(f"peak_speed:  range [{val_peak.min():.2f}, {val_peak.max():.2f}]")

rng_pairs = np.random.default_rng(2)
idx = rng_pairs.choice(len(val_frame_t), size=min(args.n_pairs, len(val_frame_t)), replace=False)
net_sel, peak_sel = val_net[idx], val_peak[idx]


def to_tensor_batch(frame_t, action_t, frame_t1, idx):
    f = torch.from_numpy(frame_t[idx]).float().permute(0, 3, 1, 2) / 255.0
    a = torch.from_numpy((action_t[idx] - action_mean) / action_std).float()
    f1 = torch.from_numpy(frame_t1[idx]).float().permute(0, 3, 1, 2) / 255.0
    return f.to(DEVICE), a.to(DEVICE), f1.to(DEVICE)


tag = f"h{H}_noise_n{args.n_episodes}_seed{args.seed}_{args.action_mode}_{args.encoder_type}"
model = CFMActionModel(
    action_dim_per_step=action_dim_per_step, horizon=H, gated=False, action_mode=args.action_mode,
    encoder_type=args.encoder_type,
).to(DEVICE)
model.load_state_dict(torch.load(f"outputs/model_cfm_{tag}.pt", map_location=DEVICE))
model.eval()

f, a_real, f1 = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, idx)
with torch.no_grad():
    target_z = model.target_encoder(f1)
    z_t = model.online_encoder(f)
target_z_n = F.normalize(target_z, dim=-1)

per_pair_error = {}
for cond_name, a in [("real", a_real), ("zero", torch.zeros_like(a_real))]:
    with torch.no_grad():
        pool = torch.stack([model.sample(z_t, a, steps=args.steps, source="noise") for _ in range(args.pool_size)], dim=0)
    pool_n = F.normalize(pool, dim=-1)
    mean_sample_n = pool_n.mean(dim=0)  # (n_pairs, embed_dim)
    per_pair_error[cond_name] = ((mean_sample_n - target_z_n) ** 2).sum(dim=-1).cpu().numpy()  # (n_pairs,)

print(f"\n=== seed{args.seed}, {len(idx)} pairs: Pearson correlation with per-pair squared bias ===")
for cond_name in ["real", "zero"]:
    err = per_pair_error[cond_name]
    r_net = np.corrcoef(net_sel, err)[0, 1]
    r_peak = np.corrcoef(peak_sel, err)[0, 1]
    print(f"{cond_name:>6s}:  corr(net_motion, error) = {r_net:+.3f}   corr(peak_speed, error) = {r_peak:+.3f}")

with open(f"outputs/day103_velocity_correlation_seed{args.seed}.json", "w") as fp:
    json.dump(
        {
            "n_pairs": len(idx),
            "corr_net_real": float(np.corrcoef(net_sel, per_pair_error["real"])[0, 1]),
            "corr_peak_real": float(np.corrcoef(peak_sel, per_pair_error["real"])[0, 1]),
            "corr_net_zero": float(np.corrcoef(net_sel, per_pair_error["zero"])[0, 1]),
            "corr_peak_zero": float(np.corrcoef(peak_sel, per_pair_error["zero"])[0, 1]),
        },
        fp,
        indent=2,
    )
print(f"saved outputs/day103_velocity_correlation_seed{args.seed}.json")
