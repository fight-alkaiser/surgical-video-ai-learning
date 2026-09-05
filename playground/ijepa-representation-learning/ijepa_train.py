"""Day93: train the I-JEPA-style model on single frames, no actions, no
frame-to-frame pairing -- just "predict the masked patches' representations
from the visible patches' representations."

Reuses the frame data already extracted by the action-conditioned-video-
prediction project (same Open-H peg_transfer episodes), but here every frame
is an independent training example -- there's no horizon, no action window,
so this dataset is simply every frame from the training episodes pooled
together (much larger effective sample count than that project's windowed
pairs, from the same underlying video).
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ijepa_model import IJEPAModel, normalized_mse_loss, variance_loss, within_image_variance_loss
from masking import sample_mask

DATA_DIR = "../action-conditioned-video-prediction/data"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--var-weight", type=float, default=5.0)
parser.add_argument("--ema-decay", type=float, default=0.996)
parser.add_argument("--clip-grad", type=float, default=0.0, help="max grad norm; 0 disables clipping")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64

with open(f"{DATA_DIR}/episode_lengths.json") as f:
    episode_lengths = json.load(f)
episode_ids = sorted(episode_lengths.keys())

# same original-20 val split convention as the rest of this series, so "held-out
# episodes" means the same thing across projects
original_20 = [f"episode_{i:06d}" for i in range(20)]
rng = np.random.default_rng(0)
shuffled = rng.permutation(original_20)
val_episodes = set(shuffled[:4])
train_episodes = set(episode_ids) - val_episodes
print(f"train episodes: {len(train_episodes)}, val episodes: {len(val_episodes)}")


def load_all_frames(ep_ids):
    frames = [np.load(f"{DATA_DIR}/episodes/{ep}_frames.npy") for ep in ep_ids]
    return np.concatenate(frames)


train_frames = load_all_frames(train_episodes)
val_frames = load_all_frames(val_episodes)
print(f"train frames: {len(train_frames)}, val frames: {len(val_frames)}")


def to_tensor(frames, idx):
    f = torch.from_numpy(frames[idx]).float().permute(0, 3, 1, 2) / 255.0
    return f.to(DEVICE)


model = IJEPAModel(ema_decay=args.ema_decay).to(DEVICE)
trainable = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
opt = torch.optim.Adam(trainable, lr=args.lr)

history = {"train_loss": [], "val_loss": [], "ctx_std": [], "ctx_cos_sim": [], "ctx_within_cos_sim": []}
best_val_loss = float("inf")
best_epoch = -1
best_state = None

for epoch in range(args.epochs):
    model.train()
    perm = np.random.permutation(len(train_frames))
    epoch_losses, epoch_std, epoch_cos_sim, epoch_within_cos_sim = [], [], [], []
    for i in range(0, len(perm), BATCH_SIZE):
        idx = perm[i : i + BATCH_SIZE]
        if len(idx) < 2:
            continue
        f = to_tensor(train_frames, idx)
        ctx_idx, tgt_idx = sample_mask()
        B = f.shape[0]
        ctx_pos = torch.tensor(ctx_idx, device=DEVICE).unsqueeze(0).expand(B, -1)
        tgt_pos = torch.tensor(tgt_idx, device=DEVICE).unsqueeze(0).expand(B, -1)

        pred, target, ctx_tokens = model(f, ctx_pos, tgt_pos)
        loss = normalized_mse_loss(pred, target)
        # Day93 fix: normalized_mse_loss only compares *directions* (both sides are
        # F.normalize'd), so an anti-collapse term on raw magnitude can be satisfied
        # while every vector still points the same way -- checked post-hoc and found
        # exactly this: cosine similarity 1.0 between different images/patches despite
        # a "healthy" raw std. Apply the variance term to normalized vectors instead,
        # with gamma rescaled for unit-norm vectors in embed_dim dimensions (a random
        # unit vector's per-dimension variance is ~1/embed_dim, so std ~1/sqrt(embed_dim)).
        normed_ctx = F.normalize(ctx_tokens, dim=-1)
        gamma = 1.0 / (ctx_tokens.shape[-1] ** 0.5)
        collapse_penalty = variance_loss(normed_ctx, gamma=gamma)
        # second collapse check found after the first fix: different images were
        # distinguishable on average, but every patch *within* one image still
        # mapped to the same vector regardless of position -- variance_loss pools
        # batch and patch together so it couldn't see that. Penalize within-image
        # collapse directly too.
        within_image_penalty = within_image_variance_loss(normed_ctx, gamma=gamma)
        total_loss = loss + args.var_weight * (collapse_penalty + within_image_penalty)
        opt.zero_grad()
        total_loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
        opt.step()
        model.update_target()
        epoch_losses.append(loss.item())
        epoch_std.append(ctx_tokens.detach().std(dim=(0, 1)).mean().item())
        # directional-collapse monitors. Across-image: different examples' [0]-th
        # context token (fooled by magnitude alone before Day93's first fix).
        # Within-image: different patches of the SAME image, example 0 -- this is
        # the one that stayed collapsed (1.0) even after that first fix.
        with torch.no_grad():
            v = F.normalize(ctx_tokens[:, 0, :].detach(), dim=-1)
            sim = v @ v.T
            off_diag = sim[~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)]
            epoch_cos_sim.append(off_diag.mean().item())

            w = F.normalize(ctx_tokens[0].detach(), dim=-1)  # (Nc, D), one image
            sim2 = w @ w.T
            off_diag2 = sim2[~torch.eye(sim2.shape[0], dtype=torch.bool, device=sim2.device)]
            epoch_within_cos_sim.append(off_diag2.mean().item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(3):
            for i in range(0, len(val_frames), BATCH_SIZE):
                idx = np.arange(i, min(i + BATCH_SIZE, len(val_frames)))
                if len(idx) < 2:
                    continue
                f = to_tensor(val_frames, idx)
                ctx_idx, tgt_idx = sample_mask()
                B = f.shape[0]
                ctx_pos = torch.tensor(ctx_idx, device=DEVICE).unsqueeze(0).expand(B, -1)
                tgt_pos = torch.tensor(tgt_idx, device=DEVICE).unsqueeze(0).expand(B, -1)
                pred, target, _ = model(f, ctx_pos, tgt_pos)
                val_losses.append(normalized_mse_loss(pred, target).item())
    val_loss = float(np.mean(val_losses))
    train_loss = float(np.mean(epoch_losses))
    ctx_std = float(np.mean(epoch_std))
    ctx_cos_sim = float(np.mean(epoch_cos_sim))
    ctx_within_cos_sim = float(np.mean(epoch_within_cos_sim))
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["ctx_std"].append(ctx_std)
    history["ctx_cos_sim"].append(ctx_cos_sim)
    history["ctx_within_cos_sim"].append(ctx_within_cos_sim)

    smoothed = float(np.mean(history["val_loss"][-5:]))
    if smoothed < best_val_loss:
        best_val_loss = smoothed
        best_epoch = epoch
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == args.epochs - 1:
        print(
            f"epoch {epoch:4d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
            f"ctx_std {ctx_std:.4f}  cos_sim(across-img) {ctx_cos_sim:.4f}  cos_sim(within-img) {ctx_within_cos_sim:.4f}"
        )

print(f"\nbest smoothed val_loss {best_val_loss:.4f} at epoch {best_epoch} (of {args.epochs}); restoring that checkpoint")
model.load_state_dict(best_state)
history["best_epoch"] = best_epoch
history["best_val_loss"] = best_val_loss

tag = f"seed{args.seed}_lr{args.lr}_ema{args.ema_decay}" + (f"_clip{args.clip_grad}" if args.clip_grad > 0 else "")
torch.save(model.state_dict(), f"outputs/model_ijepa_{tag}.pt")
with open(f"outputs/history_ijepa_{tag}.json", "w") as f:
    json.dump(history, f, indent=2)

plt.figure(figsize=(7, 5))
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.axvline(best_epoch, color="gray", linestyle="--", label=f"best_epoch={best_epoch}")
plt.xlabel("epoch")
plt.ylabel("normalized MSE (masked patch prediction)")
plt.legend()
plt.title(f"Day93: I-JEPA training curve ({tag})")
plt.tight_layout()
plt.savefig(f"outputs/loss_curve_ijepa_{tag}.png", dpi=150)
print(f"saved outputs/loss_curve_ijepa_{tag}.png")
