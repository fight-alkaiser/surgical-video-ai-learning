"""Day99 retrospective aid: the original Day93 collapsed checkpoint no longer
exists (each re-run overwrote the same filename), so to look at *real* before/
after patches concretely -- not just the aggregate cosine-similarity numbers --
this reproduces the original bug on purpose: apply variance_loss to the raw
(non-normalized) ctx_tokens, exactly as the first Day93 attempt did, for a
short run, just long enough to reach the collapsed state again. Not meant to
be trained to convergence -- purely for side-by-side inspection with the
actual fixed checkpoint.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

from ijepa_model import IJEPAModel, normalized_mse_loss, variance_loss
from masking import sample_mask

DATA_DIR = "../action-conditioned-video-prediction/data"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5  # Day93's original run collapsed well within this many

torch.manual_seed(0)
np.random.seed(0)

with open(f"{DATA_DIR}/episode_lengths.json") as f:
    episode_lengths = json.load(f)
episode_ids = sorted(episode_lengths.keys())
original_20 = [f"episode_{i:06d}" for i in range(20)]
rng = np.random.default_rng(0)
shuffled = rng.permutation(original_20)
val_episodes = set(shuffled[:4])
train_episodes = set(episode_ids) - val_episodes


def load_all_frames(ep_ids):
    return np.concatenate([np.load(f"{DATA_DIR}/episodes/{ep}_frames.npy") for ep in sorted(ep_ids)])


train_frames = load_all_frames(train_episodes)


def to_tensor(frames, idx):
    return (torch.from_numpy(frames[idx]).float().permute(0, 3, 1, 2) / 255.0).to(DEVICE)


model = IJEPAModel().to(DEVICE)
trainable = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
opt = torch.optim.Adam(trainable, lr=1e-3)

for epoch in range(EPOCHS):
    perm = np.random.permutation(len(train_frames))
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
        # THE ORIGINAL DAY93 BUG: variance_loss on raw (non-normalized) ctx_tokens,
        # while the actual loss above only ever compares normalized directions.
        collapse_penalty = variance_loss(ctx_tokens)
        total_loss = loss + 5.0 * collapse_penalty
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        model.update_target()
    print(f"epoch {epoch} done")

torch.save(model.state_dict(), "outputs/model_ijepa_day93_collapsed_repro.pt")
print("saved outputs/model_ijepa_day93_collapsed_repro.pt")
