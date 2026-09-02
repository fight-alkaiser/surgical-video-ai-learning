import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------
# This arc's three learned models so far, none winning on both axes:
#   Day43: single-shot GRU decoder -- beats both Day42 baselines on
#          accuracy (mean error 4.14mm), but the predicted path is
#          jagged (3x jitterier than real motion), because nothing in
#          a one-shot linear decode constrains frame k's prediction
#          relative to frame k+1's.
#   Day44: autoregressive GRUCell decoder -- targets the jaggedness
#          mechanism directly, but trained with pure teacher forcing,
#          fails catastrophically under free-running evaluation
#          (exposure bias): 21.28mm mean error.
#   Day45: same autoregressive decoder + scheduled sampling -- fixes
#          the exposure bias (mean error down to 6.51mm, smoothness
#          now matching real motion almost exactly), but still
#          doesn't beat the baselines on accuracy.
#
# Today tries the alternative named in Day44's Reflection and never
# attempted: go back to Day43's single-shot architecture (which
# already has the accuracy edge) and add a smoothness penalty
# directly to the loss, introducing no autoregression and therefore
# no exposure-bias risk at all. The penalty is a discrete jerk term
# (second finite difference of predicted position, anchored to the
# real last-observed position so the transition from known history
# into the prediction is smooth too, not just the prediction's own
# interior) -- squared jerk is the standard way to penalize
# physically implausible acceleration changes in a predicted
# trajectory.
#
# The regularization weight (lambda) is swept over several values
# rather than picked once, following this project's standing practice
# of checking a hyperparameter's effect directly rather than assuming
# a single value is right (e.g. Day26's pos_weight, Day33's batch
# size).
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
HELD_OUT_SUBJECT = "B"

TRAIN_STRIDE = 5
TEST_STRIDE = 30

HIDDEN_SIZE = 64
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

LAMBDA_SWEEP = [0.0, 1.0, 10.0, 50.0]

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
    inputs, targets, last_positions = [], [], []
    for name, t in windows:
        signal = trial_signals[name]
        x = (signal[t - INPUT_FRAMES:t, 0:6] - feature_mean) / feature_std
        y = signal[t:t + OUTPUT_FRAMES, 0:3]
        inputs.append(x)
        targets.append(y)
        last_positions.append(signal[t - 1, 0:3])
    return (np.stack(inputs).astype(np.float32), np.stack(targets).astype(np.float32),
            np.stack(last_positions).astype(np.float32))


train_x, train_y, train_last_pos = build_arrays(train_windows)
test_x, test_y, test_last_pos = build_arrays(test_windows)
train_y_delta = train_y - train_last_pos[:, None, :]

# ----------------------------------------
# Model: identical to Day43's single-shot GRU (GRU encoder -> one
# linear layer decoding the flattened future 30x3 displacement).
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


def jerk_penalty(pred_pos, last_pos):
    """Mean squared second finite difference of position, anchored to
    the real last-observed frame so the join between known history and
    predicted future is smooth too, not just the prediction's interior."""
    full = torch.cat([last_pos.unsqueeze(1), pred_pos], dim=1)  # (batch, 31, 3)
    second_diff = full[:, 2:, :] - 2 * full[:, 1:-1, :] + full[:, :-2, :]
    return (second_diff ** 2).sum(dim=-1).mean()


def predict_last_position_held(signal, t):
    last_xyz = signal[t - 1, 0:3]
    return np.tile(last_xyz, (OUTPUT_FRAMES, 1))


def predict_constant_velocity(signal, t):
    last_xyz = signal[t - 1, 0:3]
    last_vel = signal[t - 1, 3:6]
    steps = (np.arange(1, OUTPUT_FRAMES + 1) * DT).reshape(-1, 1)
    return last_xyz + last_vel * steps


horizon_frame_indices = [int(round(s * FPS)) - 1 for s in HORIZON_CHECKPOINTS_S]

all_results = {"held_out_subject": HELD_OUT_SUBJECT, "test_trials": test_names, "lambdas": {}}

train_x_t = torch.from_numpy(train_x)
train_y_delta_t = torch.from_numpy(train_y_delta)
train_last_pos_t = torch.from_numpy(train_last_pos)
num_train = train_x_t.shape[0]

for lam in LAMBDA_SWEEP:

    print(f"\n{'=' * 60}\nlambda_smooth = {lam}\n{'=' * 60}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    model = TrajectoryGRU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        permutation = torch.randperm(num_train)
        epoch_loss, epoch_mse, epoch_jerk, num_batches = 0.0, 0.0, 0.0, 0
        for start in range(0, num_train, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            batch_x = train_x_t[idx].to(device)
            batch_y_delta = train_y_delta_t[idx].to(device)
            batch_last_pos = train_last_pos_t[idx].to(device)

            optimizer.zero_grad()
            pred_delta = model(batch_x)
            pred_pos = pred_delta + batch_last_pos.unsqueeze(1)
            true_pos = batch_y_delta + batch_last_pos.unsqueeze(1)

            loss_mse = mse(pred_pos, true_pos)
            loss_jerk = jerk_penalty(pred_pos, batch_last_pos)
            loss = loss_mse + lam * loss_jerk
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += loss_mse.item()
            epoch_jerk += loss_jerk.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: total={epoch_loss/num_batches:.8f} "
                  f"mse={epoch_mse/num_batches:.8f} jerk={epoch_jerk/num_batches:.8f}")

    model.eval()
    with torch.no_grad():
        test_pred_delta = model(torch.from_numpy(test_x).to(device)).cpu().numpy()
    pred_xyz = test_pred_delta + test_last_pos[:, None, :]

    displacement_error = np.linalg.norm(pred_xyz - test_y, axis=2)
    step_size = np.linalg.norm(np.diff(pred_xyz, axis=1), axis=2)
    mean_full = float(displacement_error.mean())
    by_checkpoint = {
        str(s): float(displacement_error[:, idx].mean())
        for s, idx in zip(HORIZON_CHECKPOINTS_S, horizon_frame_indices)
    }

    all_results["lambdas"][str(lam)] = {
        "mean_error_full_horizon_m": mean_full,
        "mean_error_by_checkpoint_m": by_checkpoint,
        "mean_step_size_m": float(step_size.mean()),
        "median_step_size_m": float(np.median(step_size)),
        "max_step_size_m": float(step_size.max()),
    }
    print(f"\nHeld-out eval: mean error={mean_full*1000:.2f}mm, "
          f"step size mean={step_size.mean()*1000:.3f}mm")

    if lam == LAMBDA_SWEEP[-1] or lam == 10.0:
        torch.save(model.state_dict(), Path(__file__).parent / f"model_lambda_{lam}.pt")

# ----------------------------------------
# Baselines, recomputed on the same held-out windows, for reference.
# ----------------------------------------

methods = {
    "last_position_held": np.stack([
        predict_last_position_held(trial_signals[name], t) for name, t in test_windows
    ]),
    "constant_velocity": np.stack([
        predict_constant_velocity(trial_signals[name], t) for name, t in test_windows
    ]),
}

all_results["baselines"] = {}
for method_name, pred_xyz in methods.items():
    displacement_error = np.linalg.norm(pred_xyz - test_y, axis=2)
    step_size = np.linalg.norm(np.diff(pred_xyz, axis=1), axis=2)
    all_results["baselines"][method_name] = {
        "mean_error_full_horizon_m": float(displacement_error.mean()),
        "mean_error_by_checkpoint_m": {
            str(s): float(displacement_error[:, idx].mean())
            for s, idx in zip(HORIZON_CHECKPOINTS_S, horizon_frame_indices)
        },
        "mean_step_size_m": float(step_size.mean()),
    }

true_step_size = np.linalg.norm(np.diff(test_y, axis=1), axis=2)
all_results["ground_truth_step_size_m"] = {
    "mean": float(true_step_size.mean()), "median": float(np.median(true_step_size)),
}

print("\n" + "=" * 70)
print("SUMMARY: lambda sweep vs baselines")
print("=" * 70)
print(f"{'':30s} {'mean err (mm)':>14s} {'+1.0s (mm)':>12s} {'step size (mm)':>16s}")
for lam in LAMBDA_SWEEP:
    r = all_results["lambdas"][str(lam)]
    print(f"{'lambda=' + str(lam):30s} {r['mean_error_full_horizon_m']*1000:14.2f} "
          f"{r['mean_error_by_checkpoint_m']['1.0']*1000:12.2f} "
          f"{r['mean_step_size_m']*1000:16.3f}")
for name, r in all_results["baselines"].items():
    print(f"{name:30s} {r['mean_error_full_horizon_m']*1000:14.2f} "
          f"{r['mean_error_by_checkpoint_m']['1.0']*1000:12.2f} "
          f"{r['mean_step_size_m']*1000:16.3f}")
print(f"{'ground truth (reference)':30s} {'':>14s} {'':>12s} "
      f"{all_results['ground_truth_step_size_m']['mean']*1000:16.3f}")

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved results to {output_dir / 'results.json'}")

np.savez(output_dir / "normalization_stats.npz", mean=feature_mean, std=feature_std)
