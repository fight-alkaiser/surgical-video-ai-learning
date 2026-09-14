"""Day103: does the real-vs-zero-action advantage (Day96-98, pretrained
ResNet18 encoder) concentrate on pairs where the instrument actually moved a
lot during the H-step window, or is it spread evenly regardless?

Reuses Day82/98's bias/variance decomposition (cfm_eval_distribution.py)
unchanged -- same encoder, same predictor, same checkpoint, same latent
space. The only new thing is which (z_t, action, z_{t+H}) pairs go into
which group: instead of one pool of random pairs, pairs are stratified into
four quartiles by an "instrument motion" score computed from Day102's
per-frame detector (instrument_detector.py), then bias/variance is computed
separately per quartile.

Retraining or re-encoding cropped/masked frames was deliberately avoided:
the encoder (frozen ResNet18 or the from-scratch CNN) was trained on
whole, unmasked frames, so feeding it masked/cropped input at eval time
would shift its input distribution away from anything it has ever seen,
making the existing checkpoint's predictions meaningless. Stratifying which
*pairs* get evaluated, rather than changing what the encoder sees, keeps
this an apples-to-apples extension of the existing story instead of a new
one.

Instrument motion score: instrument_detector.instrument_mask gives a
per-frame boolean mask (True = instrument). For a pair (frame t, frame
t+H), take each frame's mask centroid (mean row/col of masked pixels; if a
frame's mask is empty, fall back to the nearest earlier frame with a
non-empty mask in that episode) and use the Euclidean distance between the
two centroids as the motion score for that pair -- how far the instrument's
visible position moved over the window, in pixels.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from cfm_model import CFMActionModel
from instrument_detector import instrument_mask

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=10)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--pool-size", type=int, default=64, help="samples drawn per condition per pair")
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--pairs-per-quartile", type=int, default=64)
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
    """mask_ep: (T, H, W) bool. Returns (T, 2) float array of (row, col)
    centroids, filling empty-mask frames from the nearest earlier non-empty
    frame (falling back to the image center for a run with no detection
    yet)."""
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
    all_frame_t, all_action_t, all_frame_t1, all_motion = [], [], [], []
    for ep_id in ep_ids:
        frames = np.load(f"data/episodes/{ep_id}_frames.npy")
        actions = np.load(f"data/episodes/{ep_id}_actions.npy")
        if len(frames) <= H:
            continue
        mask_ep = instrument_mask(frames)
        centroids = centroid_track(mask_ep)
        motion = np.linalg.norm(centroids[H:] - centroids[:-H], axis=1)  # (T-H,)
        action_window = np.stack([actions[i : i + H] for i in range(len(actions) - H)])
        all_frame_t.append(frames[:-H])
        all_action_t.append(action_window)
        all_frame_t1.append(frames[H:])
        all_motion.append(motion)
    return (
        np.concatenate(all_frame_t),
        np.concatenate(all_action_t),
        np.concatenate(all_frame_t1),
        np.concatenate(all_motion),
    )


train_frame_t, train_action_t, _, _ = build_pairs_with_motion(train_episodes)
action_dim_per_step = train_action_t.shape[-1]
action_mean = train_action_t.reshape(-1, action_dim_per_step).mean(axis=0)
action_std = train_action_t.reshape(-1, action_dim_per_step).std(axis=0) + 1e-6
val_frame_t, val_action_t, val_frame_t1, val_motion = build_pairs_with_motion(val_episodes)
print(f"val pairs: {len(val_frame_t)}, motion score range [{val_motion.min():.2f}, {val_motion.max():.2f}], "
      f"mean {val_motion.mean():.2f}")


def to_tensor_batch(frame_t, action_t, frame_t1, idx):
    f = torch.from_numpy(frame_t[idx]).float().permute(0, 3, 1, 2) / 255.0
    a = torch.from_numpy((action_t[idx] - action_mean) / action_std).float()
    f1 = torch.from_numpy(frame_t1[idx]).float().permute(0, 3, 1, 2) / 255.0
    return f.to(DEVICE), a.to(DEVICE), f1.to(DEVICE)


# quartile boundaries fixed once (over all val pairs), reused across seeds
quartile_edges = np.percentile(val_motion, [0, 25, 50, 75, 100])
quartile_names = ["Q1 (least motion)", "Q2", "Q3", "Q4 (most motion)"]
rng_pairs = np.random.default_rng(2)
quartile_idx = {}
for qi, name in enumerate(quartile_names):
    lo, hi = quartile_edges[qi], quartile_edges[qi + 1]
    in_q = np.where((val_motion >= lo) & (val_motion <= hi if qi == 3 else val_motion < hi))[0]
    n = min(args.pairs_per_quartile, len(in_q))
    quartile_idx[name] = rng_pairs.choice(in_q, size=n, replace=False)
    print(f"{name}: motion in [{lo:.2f}, {hi:.2f}), {len(in_q)} pairs available, using {n}")

all_results = {}  # seed -> quartile -> cond -> {bias_sq, variance}

for seed in args.seeds:
    tag = f"h{H}_noise_n{args.n_episodes}_seed{seed}_{args.action_mode}_{args.encoder_type}"
    model = CFMActionModel(
        action_dim_per_step=action_dim_per_step,
        horizon=H,
        gated=False,
        action_mode=args.action_mode,
        encoder_type=args.encoder_type,
    ).to(DEVICE)
    model.load_state_dict(torch.load(f"outputs/model_cfm_{tag}.pt", map_location=DEVICE))
    model.eval()

    seed_results = {}
    for name, idx in quartile_idx.items():
        f, a_real, f1 = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, idx)
        with torch.no_grad():
            target_z = model.target_encoder(f1)
            z_t = model.online_encoder(f)
        target_z_n = F.normalize(target_z, dim=-1)

        cond_results = {}
        for cond_name, a in [("real", a_real), ("zero", torch.zeros_like(a_real))]:
            with torch.no_grad():
                pool = torch.stack(
                    [model.sample(z_t, a, steps=args.steps, source="noise") for _ in range(args.pool_size)], dim=0
                )
            pool_n = F.normalize(pool, dim=-1)
            mean_sample_n = pool_n.mean(dim=0)
            bias_sq = ((mean_sample_n - target_z_n) ** 2).sum(dim=-1).mean().item()
            variance = ((pool_n - mean_sample_n[None]) ** 2).sum(dim=-1).mean().item()
            cond_results[cond_name] = {"bias_sq": bias_sq, "variance": variance}
        seed_results[name] = cond_results
        print(
            f"seed{seed} {name:20s}  real bias^2={cond_results['real']['bias_sq']:.4f}  "
            f"zero bias^2={cond_results['zero']['bias_sq']:.4f}  "
            f"gap(zero-real)={cond_results['zero']['bias_sq'] - cond_results['real']['bias_sq']:+.4f}"
        )
    all_results[seed] = seed_results

with open("outputs/day103_instrument_motion_results.json", "w") as fp:
    json.dump(
        {
            "quartile_edges": quartile_edges.tolist(),
            "quartile_mean_motion": {name: float(val_motion[idx].mean()) for name, idx in quartile_idx.items()},
            "results": all_results,
        },
        fp,
        indent=2,
    )
print("\nsaved outputs/day103_instrument_motion_results.json")

# --- plot: bias^2 gap (zero - real) per quartile, one line per seed ---
fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(quartile_names))
for seed in args.seeds:
    gaps = [all_results[seed][name]["zero"]["bias_sq"] - all_results[seed][name]["real"]["bias_sq"] for name in quartile_names]
    ax.plot(x, gaps, marker="o", label=f"seed{seed}")
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(quartile_names, rotation=15)
ax.set_ylabel("bias^2 gap (zero - real); positive = real action helps")
ax.set_title("Day103: does the real-action advantage concentrate where the instrument moves more?")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/day103_instrument_motion_gap.png", dpi=150)
print("saved outputs/day103_instrument_motion_gap.png")
