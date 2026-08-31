import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------
# Day43's single-shot GRU (past 1s -> future 30 frames, decoded in
# one linear layer) beat both of Day42's baselines on mean
# displacement error, but an independent smoothness check showed its
# predicted path is over 3x jitterier than real tooltip motion --
# because nothing in that architecture constrains frame k's
# prediction relative to frame k+1's.
#
# Today tests the structural fix that targets this mechanism
# directly: an autoregressive decoder. A GRUCell starts from the same
# past-window encoding Day43 used, then predicts ONE step's
# displacement at a time, each step conditioned on the previous
# step's displacement -- so consecutive predictions are chained
# through the recurrence rather than produced independently. Training
# uses teacher forcing (the decoder sees the TRUE previous delta,
# standard practice for stabilizing training); evaluation uses free
# running (the decoder sees its OWN previous prediction, since that's
# the only information available at real prediction time and is what
# actually produces the trajectory being evaluated).
#
# Known risk, stated before evaluating rather than after: autoregressive
# rollout can accumulate its own one-step errors over 30 steps
# (exposure bias), potentially trading Day43's jaggedness problem for
# a drift problem instead. Both the displacement-error metric (Day42/43)
# and the smoothness metric (Day43) are checked here, not just one.
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
HELD_OUT_SUBJECT = "B"  # matches official UserOut fold "1_Out", same as Day43

TRAIN_STRIDE = 5
TEST_STRIDE = 30

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
        y = signal[t - 1:t + OUTPUT_FRAMES, 0:3]  # include frame t-1 to compute step-0 delta
        inputs.append(x)
        targets.append(y)
    return np.stack(inputs).astype(np.float32), np.stack(targets).astype(np.float32)


train_x, train_y_pos = build_arrays(train_windows)   # train_y_pos: (N, 31, 3), frame t-1 .. t+29
test_x, test_y_pos = build_arrays(test_windows)

# True per-step deltas: delta_k = pos[t-1+k] - pos[t-1+k-1], k=1..30
train_y_delta = np.diff(train_y_pos, axis=1)  # (N, 30, 3)
test_y_delta = np.diff(test_y_pos, axis=1)

# ----------------------------------------
# Model: GRU encoder (same as Day43) -> GRUCell decoder, one step at
# a time, chained through both hidden state and previous delta.
# ----------------------------------------


class AutoregressiveTrajectoryGRU(nn.Module):

    def __init__(self, input_dim=6, hidden_size=HIDDEN_SIZE, output_frames=OUTPUT_FRAMES):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.decoder_cell = nn.GRUCell(3, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 3)
        self.output_frames = output_frames

    def forward(self, x, teacher_deltas=None):
        """teacher_deltas: (batch, output_frames, 3) ground-truth deltas
        for teacher forcing during training. If None, runs free
        (uses its own previous prediction) -- the evaluation mode."""
        _, h_enc = self.encoder(x)
        h = h_enc.squeeze(0)
        batch_size = x.shape[0]
        prev_delta = torch.zeros(batch_size, 3, device=x.device)

        predicted_deltas = []
        for step in range(self.output_frames):
            h = self.decoder_cell(prev_delta, h)
            delta_pred = self.output_layer(h)
            predicted_deltas.append(delta_pred)
            if teacher_deltas is not None:
                prev_delta = teacher_deltas[:, step, :]
            else:
                prev_delta = delta_pred

        return torch.stack(predicted_deltas, dim=1)  # (batch, output_frames, 3)


model = AutoregressiveTrajectoryGRU().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

train_x_t = torch.from_numpy(train_x)
train_y_delta_t = torch.from_numpy(train_y_delta)
num_train = train_x_t.shape[0]

print("\n=== Training autoregressive GRU (teacher forcing) ===")
for epoch in range(NUM_EPOCHS):
    model.train()
    permutation = torch.randperm(num_train)
    epoch_loss = 0.0
    num_batches = 0
    for start in range(0, num_train, BATCH_SIZE):
        idx = permutation[start:start + BATCH_SIZE]
        batch_x = train_x_t[idx].to(device)
        batch_y_delta = train_y_delta_t[idx].to(device)
        optimizer.zero_grad()
        pred_deltas = model(batch_x, teacher_deltas=batch_y_delta)
        loss = criterion(pred_deltas, batch_y_delta)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: train MSE (per-step delta) = "
              f"{epoch_loss / num_batches:.8f}")

# ----------------------------------------
# Evaluate: FREE-RUNNING rollout (no teacher forcing -- the model
# only ever sees its own previous prediction, matching real use),
# on the held-out subject, compared against Day42/43's baselines
# (recomputed here on the identical held-out windows) and Day43's
# single-shot GRU is referenced from its own saved results.
# ----------------------------------------

model.eval()
with torch.no_grad():
    pred_deltas = model(torch.from_numpy(test_x).to(device), teacher_deltas=None).cpu().numpy()

test_last_pos = test_y_pos[:, 0, :]  # position at t-1
ar_pred_xyz = test_last_pos[:, None, :] + np.cumsum(pred_deltas, axis=1)
test_true_xyz = test_y_pos[:, 1:, :]  # positions t .. t+29


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
    "autoregressive_gru": ar_pred_xyz,
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
      f"({len(test_windows)} windows), free-running rollout ===")
for method_name, pred_xyz in methods.items():
    displacement_error = np.linalg.norm(pred_xyz - test_true_xyz, axis=2)
    step_size = np.linalg.norm(np.diff(pred_xyz, axis=1), axis=2)
    mean_full = float(displacement_error.mean())
    by_checkpoint = {
        str(s): float(displacement_error[:, idx].mean())
        for s, idx in zip(HORIZON_CHECKPOINTS_S, horizon_frame_indices)
    }
    results["methods"][method_name] = {
        "mean_error_full_horizon_m": mean_full,
        "mean_error_by_checkpoint_m": by_checkpoint,
        "mean_step_size_m": float(step_size.mean()),
        "median_step_size_m": float(np.median(step_size)),
        "max_step_size_m": float(step_size.max()),
    }
    print(f"\n{method_name}:")
    print(f"  Mean displacement error (full 1s horizon): {mean_full * 1000:.2f} mm")
    for s in HORIZON_CHECKPOINTS_S:
        print(f"  at +{s:.1f}s: {by_checkpoint[str(s)] * 1000:.2f} mm")
    print(f"  Frame-to-frame step size: mean={step_size.mean()*1000:.3f}mm "
          f"median={np.median(step_size)*1000:.3f}mm max={step_size.max()*1000:.3f}mm")

# Ground-truth path's own step size, for reference (same as Day43).
true_step_size = np.linalg.norm(np.diff(test_true_xyz, axis=1), axis=2)
results["ground_truth_step_size_m"] = {
    "mean": float(true_step_size.mean()),
    "median": float(np.median(true_step_size)),
    "max": float(true_step_size.max()),
}
print(f"\nGround truth path step size: mean={true_step_size.mean()*1000:.3f}mm "
      f"median={np.median(true_step_size)*1000:.3f}mm max={true_step_size.max()*1000:.3f}mm")

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {output_dir / 'results.json'}")

torch.save(model.state_dict(), output_dir / "autoregressive_gru_model.pt")
np.savez(output_dir / "normalization_stats.npz", mean=feature_mean, std=feature_std)
