"""Day154: does "lower shuffled val SIGReg" mean a better representation?

Day153 found that LeJEPA's label-free selection criterion only behaves when val
batches are shuffled. This script asks the next question: along one training run,
does the checkpoint the (shuffled) criterion prefers also carry more downstream
signal? For every saved checkpoint of the given runs it reports

  - shuffled val SIGReg (same measurement as day153_val_batch_check.py), and
  - Day94's action probe: frozen context encoder on the full image, mean-pooled,
    small MLP probe -> per-frame action, R^2 against the mean-action baseline.

Probe protocol is copied from probe_representation_quality.py (Day94) so numbers
are comparable to that table (including its known optimism: best val_mse over the
probe's 50 epochs). One change: the probe and its data order are seeded, so two
checkpoints are compared under the same probe randomness. Random (untrained)
encoders of the same architecture are probed with 3 init seeds to give a sense of
how much R^2 moves for reasons that have nothing to do with training.

Day155: two checks for the Day154 caveats. --standardize z-scores the pooled
features with the train set's per-dimension mean/std before probing (so a
checkpoint whose features are simply larger or smaller is not easier or harder
for the probe to fit), and --probe-seeds runs the probe several times per
checkpoint. Each checkpoint is encoded once and probed under every combination
of {raw, standardized} x probe seeds. feat_scale (mean per-dimension std of the
train features) is recorded so the scale confound can be looked at directly.

Usage:
  python day154_probe_checkpoints.py --runs seed0_sigreg-image_lam0.05_valshuf ...
  (each run name is the checkpoint tag; all outputs/checkpoints/model_ijepa_<tag>_epoch*.pt are probed)
"""

import argparse
import glob
import json
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ijepa_model import NUM_PATCHES, IJEPAModel, patchify, sigreg_loss

AC_DATA_DIR = "../action-conditioned-video-prediction/data"

parser = argparse.ArgumentParser()
parser.add_argument("--runs", nargs="+", required=True)
parser.add_argument("--probe-epochs", type=int, default=50)
parser.add_argument("--out", default="outputs/day154_probe_checkpoints.json")
parser.add_argument("--probe-seeds", nargs="+", type=int, default=[0], help="Day155: probe seeds per checkpoint")
parser.add_argument("--both-scalings", action="store_true", help="Day155: probe raw AND standardized features")
args = parser.parse_args()

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64

# --- same split and frame/action alignment as Day94 ---
with open(f"{AC_DATA_DIR}/episode_lengths.json") as f:
    episode_ids = sorted(json.load(f).keys())
original_20 = [f"episode_{i:06d}" for i in range(20)]
val_episodes = set(np.random.default_rng(0).permutation(original_20)[:4])
train_episodes = set(episode_ids) - val_episodes


def load_frames_and_actions(ep_ids):
    all_frames, all_actions = [], []
    for ep in sorted(ep_ids):
        frames = np.load(f"{AC_DATA_DIR}/episodes/{ep}_frames.npy")
        actions = np.load(f"{AC_DATA_DIR}/episodes/{ep}_actions.npy")
        n = min(len(frames), len(actions))
        all_frames.append(frames[:n])
        all_actions.append(actions[:n])
    return np.concatenate(all_frames), np.concatenate(all_actions)


train_frames, train_actions = load_frames_and_actions(train_episodes)
val_frames, val_actions = load_frames_and_actions(val_episodes)
action_mean = train_actions.mean(axis=0)
action_std = train_actions.std(axis=0) + 1e-6
train_act_t = torch.from_numpy((train_actions - action_mean) / action_std).float()
val_act_t = torch.from_numpy((val_actions - action_mean) / action_std).float()
mean_baseline_mse = F.mse_loss(train_act_t.mean(0, keepdim=True).expand_as(val_act_t), val_act_t).item()
print(f"train frames {len(train_frames)}, val frames {len(val_frames)}, mean-action baseline val_mse {mean_baseline_mse:.4f}")

# val frames here are the probe's val set (frames trimmed to len(actions)); the
# shuffled order for SIGReg is drawn over them with the same seed as Day153's check.
shuffled_val = np.random.default_rng(0).permutation(len(val_frames))


def to_tensor(frames, idx):
    return (torch.from_numpy(frames[idx]).float().permute(0, 3, 1, 2) / 255.0).to(DEVICE)


@torch.no_grad()
def encode(model, frames, order=None):
    """(N, 64, D) tokens are too big to keep; return pooled (N, D) and, if order is
    given, the shuffled-batch SIGReg computed on the fly."""
    order = np.arange(len(frames)) if order is None else order
    pooled = []
    for i in range(0, len(order), BATCH_SIZE):
        idx = order[i : i + BATCH_SIZE]
        f = to_tensor(frames, idx)
        pos = torch.arange(NUM_PATCHES, device=DEVICE).unsqueeze(0).expand(len(idx), -1)
        pooled.append(model.context_encoder(patchify(f), pos).mean(1).cpu())
    return torch.cat(pooled)


@torch.no_grad()
def shuffled_sigreg(model, on):
    sigs = []
    for i in range(0, len(shuffled_val) - 63, 64):
        idx = shuffled_val[i : i + 64]
        f = to_tensor(val_frames, idx)
        pos = torch.arange(NUM_PATCHES, device=DEVICE).unsqueeze(0).expand(len(idx), -1)
        tok = model.context_encoder(patchify(f), pos)
        z = tok.mean(1) if on == "image" else tok.reshape(-1, tok.shape[-1])
        g = torch.Generator(device=DEVICE).manual_seed(0)
        sigs.append(sigreg_loss(z, num_slices=256, generator=g).item())
    return float(np.mean(sigs))


class Probe(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def probe_r2(train_rep, val_rep, seed=0):
    torch.manual_seed(seed)
    probe = Probe(train_rep.shape[-1], train_act_t.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    tr, ta, vr, va = train_rep.to(DEVICE), train_act_t.to(DEVICE), val_rep.to(DEVICE), val_act_t.to(DEVICE)
    best = float("inf")
    for _ in range(args.probe_epochs):
        probe.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(perm), 256):
            idx = perm[i : i + 256]
            loss = F.mse_loss(probe(tr[idx]), ta[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            best = min(best, F.mse_loss(probe(vr), va).item())
    return best, 1 - best / mean_baseline_mse


def probe_all(train_rep, val_rep):
    """Day155: probe every (scaling, probe seed) combination for one encoder.
    Returns {"raw": [r2 per seed], "std": [...]} (std only with --both-scalings)
    and the feature scale. Standardization uses train statistics only."""
    out = {"feat_scale": train_rep.std(0).mean().item()}
    scalings = ["raw", "std"] if args.both_scalings else ["raw"]
    for sc in scalings:
        tr, vr = train_rep, val_rep
        if sc == "std":
            mu, sd = train_rep.mean(0, keepdim=True), train_rep.std(0, keepdim=True) + 1e-6
            tr, vr = (train_rep - mu) / sd, (val_rep - mu) / sd
        out[sc] = [probe_r2(tr, vr, seed=ps)[1] for ps in args.probe_seeds]
    return out


def fmt(o):
    parts = [f"scale {o['feat_scale']:.3f}"]
    for sc in ["raw", "std"]:
        if sc in o:
            parts.append(f"R2[{sc}] {np.mean(o[sc]):.3f}+-{np.std(o[sc]):.3f}")
    return "  ".join(parts)


results = {"mean_baseline_mse": mean_baseline_mse, "probe_seeds": args.probe_seeds, "random": [], "runs": {}}

for init_seed in [123, 1, 2]:
    torch.manual_seed(init_seed)
    m = IJEPAModel().to(DEVICE).eval()
    o = probe_all(encode(m, train_frames), encode(m, val_frames))
    o["init_seed"] = init_seed
    o["r2"] = float(np.mean(o["raw"]))
    results["random"].append(o)
    print(f"random encoder (init {init_seed:3d}): {fmt(o)}")

for run in args.runs:
    on = "image" if "sigreg-image" in run else "patch"
    paths = sorted(glob.glob(f"outputs/checkpoints/model_ijepa_{run}_epoch*.pt"))
    if not paths:
        print(f"no checkpoints for {run}")
        continue
    rows = []
    for p in paths:
        epoch = int(re.search(r"epoch(\d+)", p).group(1))
        m = IJEPAModel().to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.eval()
        sig = shuffled_sigreg(m, on)
        o = probe_all(encode(m, train_frames), encode(m, val_frames))
        # "r2" keeps Day154's meaning (raw features; mean over probe seeds)
        rows.append({"epoch": epoch, "shuffled_val_sigreg": sig, "r2": float(np.mean(o["raw"])), **o})
        print(f"{run:40s} epoch {epoch:2d}  shuffled SIGReg {sig:6.2f}  {fmt(o)}")
    results["runs"][run] = rows
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

with open(args.out, "w") as f:
    json.dump(results, f, indent=2)
print(f"saved {args.out}")
