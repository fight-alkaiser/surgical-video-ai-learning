import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------
# Day42 established zero-training baselines for forecasting the
# slave-right tooltip's future position from its past kinematic
# state, and found a crossover: constant-velocity wins short-term
# (+0.1s) but loses to doing nothing at all by +1.0s, because real
# motion changes direction within a second more often than not.
# Today's question, previewed in Day42's Reflection: can a learned
# model beat BOTH baselines across the horizon, by learning something
# about upcoming direction change that neither closed-form baseline
# can see?
#
# Model: a single GRU encodes the past 1s window (xyz + translational
# velocity, 6-dim per frame -- velocity is already in the kinematics,
# giving the model more to work with than raw position alone) into a
# hidden state, then one linear layer maps that hidden state directly
# to the flattened future 30-frame x 3-dim (xyz only, to stay
# comparable to Day42's evaluation metric) position sequence. This is
# a single-shot multi-step prediction (no autoregressive rollout) --
# the simplest sequence model that could plausibly beat both
# baselines, kept deliberately small rather than reaching for
# anything more complex on a first attempt.
#
# Split: JIGSAWS' own UserOut design (Day41) -- leave one subject's
# trials out entirely for testing, train on the rest. This mirrors
# fold "1_Out" of the official Suturing/.../UserOut split (confirmed
# in Day41 to hold out subject B), reproduced directly here by
# subject letter rather than parsing the split's file lists.
#
# Anti-fabrication rule (Day41, binding for this whole arc): the GRU
# only ever sees frames strictly before the prediction window during
# both training and inference -- nothing at or after t is in its
# input. Held-out subject's trials never appear during training.
# ----------------------------------------

JIGSAWS_ROOT = Path("/Users/katsutoshimakino/Datasets/JIGSAWS")
TASK = "Suturing"
KINEMATICS_DIR = JIGSAWS_ROOT / TASK / "kinematics" / "AllGestures"
META_PATH = JIGSAWS_ROOT / TASK / f"meta_file_{TASK}.txt"

FPS = 30.0
DT = 1.0 / FPS
INPUT_FRAMES = 30
OUTPUT_FRAMES = 30
HORIZON_CHECKPOINTS_S = [0.1, 0.3, 0.5, 1.0]
HELD_OUT_SUBJECT = "B"  # matches official UserOut fold "1_Out"

TRAIN_STRIDE = 5   # denser sampling for more training windows
TEST_STRIDE = 30   # non-overlapping, matching Day42's evaluation exactly

HIDDEN_SIZE = 64
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

SLAVE_RIGHT_XYZ_COLS = [57, 58, 59]
SLAVE_RIGHT_VEL_COLS = [69, 70, 71]
SLAVE_RIGHT_GRIPPER_COL = 75

trial_names = [
    line.split()[0] for line in META_PATH.read_text().splitlines() if line.strip()
]


def load_trial_signal(trial_name):
    path = KINEMATICS_DIR / f"{trial_name}.txt"
    if not path.exists():
        return None
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 76:
            continue
        values = [float(parts[i]) for i in
                  SLAVE_RIGHT_XYZ_COLS + SLAVE_RIGHT_VEL_COLS + [SLAVE_RIGHT_GRIPPER_COL]]
        rows.append(values)
    return np.array(rows) if rows else None


trial_signals = {}
for name in trial_names:
    signal = load_trial_signal(name)
    if signal is not None and len(signal) > INPUT_FRAMES + OUTPUT_FRAMES:
        trial_signals[name] = signal

train_names = [n for n in trial_signals if n.split("_")[-1][0] != HELD_OUT_SUBJECT]
test_names = [n for n in trial_signals if n.split("_")[-1][0] == HELD_OUT_SUBJECT]

print(f"Train trials ({len(train_names)}), held-out subject "
      f"'{HELD_OUT_SUBJECT}' test trials ({len(test_names)}): {test_names}")


def make_windows(names, stride):
    windows = []
    for name in names:
        signal = trial_signals[name]
        t = INPUT_FRAMES
        while t + OUTPUT_FRAMES <= len(signal):
            windows.append((name, t))
            t += stride
    return windows


train_windows = make_windows(train_names, TRAIN_STRIDE)
test_windows = make_windows(test_names, TEST_STRIDE)
print(f"Train windows: {len(train_windows)}, Test windows: {len(test_windows)}")

# ----------------------------------------
# Normalize inputs using TRAIN-set statistics only (test/held-out
# subject stats are never used for fitting, avoiding any leakage of
# the held-out subject's distribution into preprocessing).
# ----------------------------------------

train_inputs_all = np.concatenate([
    trial_signals[name][t - INPUT_FRAMES:t, 0:6] for name, t in train_windows
])
feature_mean = train_inputs_all.mean(axis=0)
feature_std = train_inputs_all.std(axis=0) + 1e-8


def build_arrays(windows):
    inputs, targets = [], []
    for name, t in windows:
        signal = trial_signals[name]
        x = (signal[t - INPUT_FRAMES:t, 0:6] - feature_mean) / feature_std
        y = signal[t:t + OUTPUT_FRAMES, 0:3]  # position targets stay in meters
        inputs.append(x)
        targets.append(y)
    return np.stack(inputs).astype(np.float32), np.stack(targets).astype(np.float32)


train_x, train_y = build_arrays(train_windows)
test_x, test_y = build_arrays(test_windows)

# Model predicts DISPLACEMENT from the last observed position, not
# absolute position -- an easier target (small values centered near
# zero) that also makes the model's job closer to "how does position
# change" rather than "what is the absolute coordinate."
train_last_pos = np.stack([
    trial_signals[name][t - 1, 0:3] for name, t in train_windows
]).astype(np.float32)
test_last_pos = np.stack([
    trial_signals[name][t - 1, 0:3] for name, t in test_windows
]).astype(np.float32)
train_y_delta = train_y - train_last_pos[:, None, :]
test_y_delta = test_y - test_last_pos[:, None, :]

# ----------------------------------------
# Model: GRU encoder -> linear decoder, single-shot multi-step.
# ----------------------------------------


class TrajectoryGRU(nn.Module):

    def __init__(self, input_dim=6, hidden_size=HIDDEN_SIZE, output_frames=OUTPUT_FRAMES):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_frames * 3)
        self.output_frames = output_frames

    def forward(self, x):
        _, h_n = self.gru(x)
        out = self.decoder(h_n.squeeze(0))
        return out.view(-1, self.output_frames, 3)


model = TrajectoryGRU().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

train_x_t = torch.from_numpy(train_x)
train_y_t = torch.from_numpy(train_y_delta)
num_train = train_x_t.shape[0]

print("\n=== Training GRU trajectory model ===")
for epoch in range(NUM_EPOCHS):
    model.train()
    permutation = torch.randperm(num_train)
    epoch_loss = 0.0
    num_batches = 0
    for start in range(0, num_train, BATCH_SIZE):
        idx = permutation[start:start + BATCH_SIZE]
        batch_x = train_x_t[idx].to(device)
        batch_y = train_y_t[idx].to(device)
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: train MSE = {epoch_loss / num_batches:.8f}")

# ----------------------------------------
# Evaluate: GRU model vs. the two Day42 baselines, ALL recomputed on
# the SAME held-out subject's windows for a fair, apples-to-apples
# comparison (Day42's published numbers were aggregated over every
# subject, not just this held-out one).
# ----------------------------------------

model.eval()
with torch.no_grad():
    test_pred_delta = model(torch.from_numpy(test_x).to(device)).cpu().numpy()
gru_pred_xyz = test_pred_delta + test_last_pos[:, None, :]


def predict_last_position_held(signal, t):
    last_xyz = signal[t - 1, 0:3]
    return np.tile(last_xyz, (OUTPUT_FRAMES, 1))


def predict_constant_velocity(signal, t):
    last_xyz = signal[t - 1, 0:3]
    last_vel = signal[t - 1, 3:6]
    steps = (np.arange(1, OUTPUT_FRAMES + 1) * DT).reshape(-1, 1)
    return last_xyz + last_vel * steps


horizon_frame_indices = [int(round(s * FPS)) - 1 for s in HORIZON_CHECKPOINTS_S]

methods = {
    "gru_model": gru_pred_xyz,
    "last_position_held": np.stack([
        predict_last_position_held(trial_signals[name], t) for name, t in test_windows
    ]),
    "constant_velocity": np.stack([
        predict_constant_velocity(trial_signals[name], t) for name, t in test_windows
    ]),
}

results = {
    "held_out_subject": HELD_OUT_SUBJECT,
    "test_trials": test_names,
    "num_test_windows": len(test_windows),
    "methods": {},
}

print(f"\n=== Held-out subject '{HELD_OUT_SUBJECT}' evaluation "
      f"({len(test_windows)} windows) ===")
for method_name, pred_xyz in methods.items():
    displacement_error = np.linalg.norm(pred_xyz - test_y, axis=2)  # (windows, frames)
    mean_full = float(displacement_error.mean())
    by_checkpoint = {
        str(s): float(displacement_error[:, idx].mean())
        for s, idx in zip(HORIZON_CHECKPOINTS_S, horizon_frame_indices)
    }
    results["methods"][method_name] = {
        "mean_error_full_horizon_m": mean_full,
        "mean_error_by_checkpoint_m": by_checkpoint,
    }
    print(f"\n{method_name}:")
    print(f"  Mean error (full 1s horizon): {mean_full * 1000:.2f} mm")
    for s in HORIZON_CHECKPOINTS_S:
        print(f"  at +{s:.1f}s: {by_checkpoint[str(s)] * 1000:.2f} mm")

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {output_dir / 'results.json'}")

torch.save(model.state_dict(), output_dir / "gru_trajectory_model.pt")
np.savez(output_dir / "normalization_stats.npz", mean=feature_mean, std=feature_std)
