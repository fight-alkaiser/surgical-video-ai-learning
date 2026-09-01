"""Train the Conditional Flow Matching action-conditioned predictor (Day78).

Same data, same episode-level train/val split, same horizon convention as
jepa_train.py, so results are directly comparable to the Day62 JEPA model.

The evaluation goes one step further than jepa_train.py's real/shuffled/zero
action comparison: because the CFM model produces *samples* (not a single
deterministic vector), we can also ask whether the action narrows down the
spread of plausible futures, not just whether it shifts the mean prediction.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from cfm_model import CFMActionModel, per_example_normalized_mse, variance_loss
from jepa_model import normalized_mse_loss

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=10)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0, help="torch/numpy init + shuffling seed, for stability checks")
parser.add_argument("--var-weight", type=float, default=5.0, help="weight on the anti-collapse variance term")
parser.add_argument("--sample-steps", type=int, default=16, help="Euler steps for ODE sampling at eval time")
parser.add_argument("--num-samples", type=int, default=8, help="samples drawn per (z_t, action) pair for the spread check")
parser.add_argument(
    "--source",
    choices=["noise", "zt"],
    default="noise",
    help="flow start point: 'noise' (standard CFM, x0~N(0,I)) or 'zt' (Rectified-Flow-style residual, x0=z_t). "
    "Note: with 'zt' the ODE map is deterministic given z_t, so sample spread is expected to be ~0 by construction.",
)
parser.add_argument(
    "--gated",
    action="store_true",
    help="Day79: use GatedVelocityPredictor -- action is injected through a learned sigmoid gate "
    "(initialized mostly closed) instead of being concatenated in directly.",
)
parser.add_argument(
    "--action-mode",
    choices=["flatten", "sequence", "transformer"],
    default="flatten",
    help="How the (H, action_dim) window becomes one embedding. 'flatten' (Day78-85 default) "
    "concatenates all H steps into one vector, same as CHSS's own action chunking. 'sequence' "
    "(Day86) runs a small GRU over the window instead. 'transformer' (Day88) runs a small "
    "self-attention encoder over the whole window at once, closer to ACT's decoder / pi0's Action "
    "Expert design.",
)
parser.add_argument(
    "--self-forcing-prob",
    type=float,
    default=0.0,
    help="Day90: probability that a training batch uses a self-forcing target instead of the exact "
    "interpolated point -- see CFMActionModel.training_step docstring. 0.0 (default) reproduces "
    "plain CFM training (Day78-89 behavior).",
)
parser.add_argument(
    "--self-forcing-steps",
    type=int,
    default=4,
    help="number of Euler steps available for the self-forcing rollout (a random count from 1 to this "
    "many is used each time, so the model sees drift of varying severity)",
)
args = parser.parse_args()
H = args.horizon
torch.manual_seed(args.seed)
np.random.seed(args.seed)  # separate from the rng(0) used below for the train/val episode split, which stays fixed

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32

with open("data/episode_lengths.json") as f:
    episode_lengths = json.load(f)
episode_ids = sorted(episode_lengths.keys())

# Day80: val_episodes is pinned to a permutation of the ORIGINAL 20 episodes
# (Day61-79's dataset), not of however many episodes happen to be on disk.
# This keeps the held-out set identical (episode_000002/4/6/19) whether we're
# training on 20 episodes or 100+ -- otherwise adding episodes would reshuffle
# which ones are held out, and Day78/79 results would no longer be comparable.
original_20 = [f"episode_{i:06d}" for i in range(20)]
rng = np.random.default_rng(0)  # same seed as jepa_train.py / train.py -- same original val split
shuffled = rng.permutation(original_20)
val_episodes = set(shuffled[:4])
train_episodes = set(episode_ids) - val_episodes  # original 16 + any additional episodes on disk
print(f"train episodes ({len(train_episodes)}): {sorted(train_episodes)}")
print(f"val episodes ({len(val_episodes)}):   {sorted(val_episodes)}")

tag = f"h{H}_{args.source}_n{len(episode_ids)}_seed{args.seed}_{args.action_mode}" + ("_gated" if args.gated else "")
if args.self_forcing_prob > 0:
    tag += f"_sf{args.self_forcing_prob}"


def build_pairs(ep_ids):
    """action_window keeps shape (N, H, action_dim_per_step) -- unflattened.
    Day86: flattening now happens inside the model (CFMActionModel.encode_action),
    not here, so the same windows feed either action_mode without duplicating data."""
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
print(f"action window shape: {train_action_t.shape} (N, H, action_dim_per_step)")

action_dim_per_step = train_action_t.shape[-1]
# normalize per physical action dim, pooling over episode-timestep and window-position
# (position-in-window isn't a separate physical quantity, so it shouldn't get separate stats)
action_mean = train_action_t.reshape(-1, action_dim_per_step).mean(axis=0)
action_std = train_action_t.reshape(-1, action_dim_per_step).std(axis=0) + 1e-6


def to_tensor_batch(frame_t, action_t, frame_t1, idx):
    f = torch.from_numpy(frame_t[idx]).float().permute(0, 3, 1, 2) / 255.0
    a = torch.from_numpy((action_t[idx] - action_mean) / action_std).float()  # (B, H, action_dim_per_step)
    f1 = torch.from_numpy(frame_t1[idx]).float().permute(0, 3, 1, 2) / 255.0
    return f.to(DEVICE), a.to(DEVICE), f1.to(DEVICE)


model = CFMActionModel(
    action_dim_per_step=action_dim_per_step, horizon=H, gated=args.gated, action_mode=args.action_mode
).to(DEVICE)
trainable_params = list(model.online_encoder.parameters()) + list(model.velocity.parameters())
if model.action_encoder is not None:
    trainable_params += list(model.action_encoder.parameters())
opt = torch.optim.Adam(trainable_params, lr=args.lr)

history = {"train_loss": [], "val_loss": [], "z_std": [], "self_forcing_prob": args.self_forcing_prob, "self_forcing_steps": args.self_forcing_steps}
best_val_loss = float("inf")
best_epoch = -1
best_state = None

for epoch in range(args.epochs):
    model.train()
    perm = np.random.permutation(len(train_frame_t))
    epoch_losses, epoch_zstd = [], []
    for i in range(0, len(perm), BATCH_SIZE):
        batch_idx = perm[i : i + BATCH_SIZE]
        f, a, f1 = to_tensor_batch(train_frame_t, train_action_t, train_frame_t1, batch_idx)
        velocity_loss, z_t = model.training_step(
            f, a, f1, source=args.source, self_forcing_prob=args.self_forcing_prob, self_forcing_steps=args.self_forcing_steps
        )
        collapse_penalty = variance_loss(z_t)
        loss = velocity_loss + args.var_weight * collapse_penalty
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_target()
        epoch_losses.append(velocity_loss.item())
        epoch_zstd.append(z_t.detach().std(dim=0).mean().item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(3):  # average over 3 stochastic (s, x0) draws -- the CFM loss is noisy per-draw
            for i in range(0, len(val_frame_t), BATCH_SIZE):
                idx = np.arange(i, min(i + BATCH_SIZE, len(val_frame_t)))
                f, a, f1 = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, idx)
                velocity_loss, _ = model.training_step(f, a, f1, source=args.source)
                val_losses.append(velocity_loss.item())
    val_loss = float(np.mean(val_losses))
    train_loss = float(np.mean(epoch_losses))
    z_std = float(np.mean(epoch_zstd))
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["z_std"].append(z_std)

    # Day83: even averaged over 3 draws, val_loss still bounces epoch-to-epoch (the CFM
    # loss is inherently stochastic), so picking the single lowest raw value risks locking
    # onto a lucky noisy dip rather than a genuine generalization point. Smooth with a
    # trailing moving average before comparing.
    smoothed = float(np.mean(history["val_loss"][-5:]))
    if smoothed < best_val_loss:
        best_val_loss = smoothed
        best_epoch = epoch
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:4d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  z_std {z_std:.4f}")

# Day83: more data (n=200) turned out to overfit within the fixed 100-epoch budget --
# val_loss (smoothed) bottomed out early and rose for the rest of the run. Restore the
# best-smoothed-val_loss checkpoint before evaluating/saving, instead of whatever epoch
# training happened to stop at.
print(f"\nbest smoothed val_loss {best_val_loss:.4f} at epoch {best_epoch} (of {args.epochs}); restoring that checkpoint")
model.load_state_dict(best_state)
history["best_epoch"] = best_epoch
history["best_val_loss"] = best_val_loss

# --- evaluation on held-out episodes ---
#
# Day78 follow-up: comparing a *generated sample* to the one observed
# continuation is a biased metric for a generative model -- the actual
# future given (z_t, action) may be genuinely multimodal (small instrument
# jitter, camera noise), so even a perfectly-calibrated sampler will often
# land away from the one realization that happened to occur. Averaging
# several samples before comparing (what this file did until now) makes it
# worse, not better: it reconstructs the conditional-mean estimate, i.e.
# exactly the deterministic-regression behavior CFM was meant to move away
# from. Two metrics that don't have this bias:
#
# 1. paired_loss: the *training* CFM loss (velocity matching against the
#    real observed transition), evaluated with each candidate action. This
#    asks "how well does this action explain what actually happened?"
#    without ever converting the model into a point estimate -- exactly the
#    quantity the model was optimized against, so it doubles as a rough
#    likelihood proxy. Valid for both source="noise" and source="zt".
# 2. best_of_n: draw num_samples ODE samples and score the *closest* one to
#    the true target, not the average of all of them. This asks "is the
#    true continuation inside the set of futures this action makes
#    plausible?" without penalizing genuine diversity. Only meaningful for
#    source="noise" -- source="zt" is a deterministic map (spread ~0 by
#    construction), so best-of-N collapses back to the single-sample error.
model.eval()
with torch.no_grad():
    idx_all = np.arange(len(val_frame_t))
    rand_idx = np.random.default_rng(1).permutation(len(val_action_t))
    N_LOSS_REPEATS = 4  # average over this many random (s, x0) draws to denoise the paired_loss estimate

    results = {}
    for condition in ["real", "shuffled", "zero"]:
        paired_losses, best_of_n_errors, sample_spreads, gate_values = [], [], [], []
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
                velocity_loss, _ = model.training_step(f, a, f1, source=args.source)
                paired_losses.append(velocity_loss.item())

            if args.gated:
                z_t = model.online_encoder(f)
                gate_values.append(model.mean_gate(z_t, a))

            if args.source == "noise":
                target_z = model.target_encoder(f1)
                z_t = model.online_encoder(f)
                samples = torch.stack(
                    [model.sample(z_t, a, steps=args.sample_steps, source=args.source) for _ in range(args.num_samples)],
                    dim=0,
                )  # (num_samples, B, embed_dim)
                per_sample_dist = torch.stack(
                    [per_example_normalized_mse(s, target_z) for s in samples], dim=0
                )  # (num_samples, B)
                best_of_n_errors.append(per_sample_dist.min(dim=0).values.mean().item())
                sample_spreads.append(samples.std(dim=0).mean().item())

        results[condition] = {"paired_loss": float(np.mean(paired_losses))}
        if args.source == "noise":
            results[condition]["best_of_n_error"] = float(np.mean(best_of_n_errors))
            results[condition]["sample_spread"] = float(np.mean(sample_spreads))
        if args.gated:
            results[condition]["mean_gate"] = float(np.mean(gate_values))

    # copy z_t forward, in the same latent space, as the do-nothing baseline
    # (not on the same metric as paired_loss -- kept as a point of reference)
    copy_errors = []
    for i in range(0, len(idx_all), BATCH_SIZE):
        idx = idx_all[i : i + BATCH_SIZE]
        f, _, f1 = to_tensor_batch(val_frame_t, val_action_t, val_frame_t1, idx)
        target_z = model.target_encoder(f1)
        z_t = model.online_encoder(f)
        copy_errors.append(normalized_mse_loss(z_t, target_z).item())
    results["copy_zt_baseline"] = {"error_vs_target": float(np.mean(copy_errors))}

print()
for condition, r in results.items():
    parts = [f"{k}: {v:.4f}" for k, v in r.items()]
    print(f"{condition:>16s} -- " + "   ".join(parts))

history["eval"] = results
history["train_episodes"] = sorted(train_episodes)
history["val_episodes"] = sorted(val_episodes)

with open(f"outputs/history_cfm_{tag}.json", "w") as fp:
    json.dump(history, fp, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val (held-out episodes)")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("CFM velocity MSE loss")
axes[0].legend()
axes[0].set_title("Conditional Flow Matching loss")

axes[1].plot(history["z_std"])
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("mean std of z_t across batch")
axes[1].set_title("Collapse check (should stay well above 0)")

plt.tight_layout()
plt.savefig(f"outputs/loss_curve_cfm_{tag}.png", dpi=120)
print(f"saved outputs/loss_curve_cfm_{tag}.png")

torch.save(model.state_dict(), f"outputs/model_cfm_{tag}.pt")
