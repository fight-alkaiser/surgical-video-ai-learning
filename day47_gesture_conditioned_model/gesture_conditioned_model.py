import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------
# Day46's smoothness-regularized model achieved this arc's best
# accuracy/step-size trade-off, but a third diagnostic (path
# efficiency) showed its predictions still don't make purposeful net
# progress the way real motion does. This led to a broader question:
# is UNCONDITIONED trajectory forecasting even well-posed? The
# instrument's future path is chosen by the surgeon's real-time
# judgment -- information not present in 30 frames of past motion --
# so a model with no way to know what maneuver is coming next may be
# trying to predict something genuinely under-determined by its
# input.
#
# Today reframes the problem: instead of guessing an undetermined
# future choice, model the dynamics of a KNOWN gesture. JIGSAWS'
# transcriptions already label which of ~10 gesture categories is
# active at every frame (Day41). This day assumes the gesture for
# the prediction window is given (an oracle input, analogous to
# Day37's ground-truth instrument conditioning in the CholecT50
# series) -- a deliberately idealized first test of "does knowing the
# intent help," before attempting a realistic version with a
# predicted (not ground-truth) gesture label, the way Day38 followed
# Day37 there.
#
# Windows are restricted to those where a SINGLE gesture covers the
# entire 60-frame span (30 past + 30 future) -- if the gesture changes
# mid-window, the "known gesture" premise doesn't cleanly apply to
# either the input or the target, so those windows are excluded
# rather than assigned an ambiguous label.
#
# Architecture: Day46's best configuration (GRU encoder, single-shot
# linear decoder, jerk penalty at lambda=50), with one addition -- the
# future window's gesture one-hot is concatenated to the GRU's final
# hidden state before decoding.
# ----------------------------------------

JIGSAWS_ROOT = Path("/Users/katsutoshimakino/Datasets/JIGSAWS")
TASK = "Suturing"
KINEMATICS_DIR = JIGSAWS_ROOT / TASK / "kinematics" / "AllGestures"
TRANSCRIPTIONS_DIR = JIGSAWS_ROOT / TASK / "transcriptions"
META_PATH = JIGSAWS_ROOT / TASK / f"meta_file_{TASK}.txt"

FPS = 30.0
DT = 1.0 / FPS
INPUT_FRAMES = 30
OUTPUT_FRAMES = 30
HORIZON_CHECKPOINTS_S = [0.1, 0.3, 0.5, 1.0]
HELD_OUT_SUBJECT = "B"
LAMBDA_SMOOTH = 50.0  # Day46's best setting

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


def load_trial_gesture_labels(trial_name, num_frames):
    """Per-frame gesture ID array, -1 where no transcription covers
    the frame (start/end padding some trials have)."""
    path = TRANSCRIPTIONS_DIR / f"{trial_name}.txt"
    labels = np.full(num_frames, -1, dtype=np.int64)
    if not path.exists():
        return labels
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        start, end, gesture = int(parts[0]), int(parts[1]), int(parts[2].lstrip("G"))
        end = min(end, num_frames - 1)
        if start <= end:
            labels[start:end + 1] = gesture
    return labels


trial_signals = {}
trial_gestures = {}
all_gesture_ids = set()
for name in trial_names:
    signal = load_trial_signal(name)
    if signal is not None and len(signal) > INPUT_FRAMES + OUTPUT_FRAMES:
        trial_signals[name] = signal
        gestures = load_trial_gesture_labels(name, len(signal))
        trial_gestures[name] = gestures
        all_gesture_ids.update(int(g) for g in np.unique(gestures) if g != -1)

gesture_id_list = sorted(all_gesture_ids)
gesture_to_index = {g: i for i, g in enumerate(gesture_id_list)}
NUM_GESTURES = len(gesture_id_list)
print(f"Gesture vocabulary ({NUM_GESTURES}): {gesture_id_list}")

train_names = [n for n in trial_signals if n.split("_")[-1][0] != HELD_OUT_SUBJECT]
test_names = [n for n in trial_signals if n.split("_")[-1][0] == HELD_OUT_SUBJECT]

print(f"Train trials ({len(train_names)}), held-out subject "
      f"'{HELD_OUT_SUBJECT}' test trials ({len(test_names)}): {test_names}")


def make_windows(names, stride):
    """Only windows where a single gesture covers frames
    [t-INPUT_FRAMES, t+OUTPUT_FRAMES) entirely."""
    windows = []
    dropped_mixed = 0
    for name in names:
        signal = trial_signals[name]
        gestures = trial_gestures[name]
        t = INPUT_FRAMES
        while t + OUTPUT_FRAMES <= len(signal):
            span = gestures[t - INPUT_FRAMES:t + OUTPUT_FRAMES]
            unique_vals = np.unique(span)
            if len(unique_vals) == 1 and unique_vals[0] != -1:
                windows.append((name, t, int(unique_vals[0])))
            else:
                dropped_mixed += 1
            t += stride
    print(f"  ({name if False else ''}dropped {dropped_mixed} windows spanning "
          f"a gesture change or unlabeled frames)")
    return windows


train_windows = make_windows(train_names, TRAIN_STRIDE)
test_windows = make_windows(test_names, TEST_STRIDE)
print(f"Train windows: {len(train_windows)}, Test windows: {len(test_windows)}")

train_inputs_all = np.concatenate([
    trial_signals[name][t - INPUT_FRAMES:t, 0:6] for name, t, _ in train_windows
])
feature_mean = train_inputs_all.mean(axis=0)
feature_std = train_inputs_all.std(axis=0) + 1e-8


def build_arrays(windows):
    inputs, targets, last_positions, gesture_onehots = [], [], [], []
    for name, t, gesture_id in windows:
        signal = trial_signals[name]
        x = (signal[t - INPUT_FRAMES:t, 0:6] - feature_mean) / feature_std
        y = signal[t:t + OUTPUT_FRAMES, 0:3]
        inputs.append(x)
        targets.append(y)
        last_positions.append(signal[t - 1, 0:3])
        onehot = np.zeros(NUM_GESTURES, dtype=np.float32)
        onehot[gesture_to_index[gesture_id]] = 1.0
        gesture_onehots.append(onehot)
    return (np.stack(inputs).astype(np.float32), np.stack(targets).astype(np.float32),
            np.stack(last_positions).astype(np.float32), np.stack(gesture_onehots))


train_x, train_y, train_last_pos, train_gesture = build_arrays(train_windows)
test_x, test_y, test_last_pos, test_gesture = build_arrays(test_windows)
train_y_delta = train_y - train_last_pos[:, None, :]

# ----------------------------------------
# Models: unconditioned (Day46 recipe, recomputed on THIS filtered
# window set for a fair comparison) and gesture-conditioned (gesture
# one-hot concatenated to the GRU's final hidden state).
# ----------------------------------------


class TrajectoryGRU(nn.Module):
    """Day46's unconditioned model, recomputed here as the baseline
    for comparison on the gesture-filtered window set."""

    def __init__(self, input_dim=6, hidden_size=HIDDEN_SIZE, output_frames=OUTPUT_FRAMES):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_frames * 3)
        self.output_frames = output_frames

    def forward(self, x, gesture_onehot=None):
        _, h_n = self.gru(x)
        out = self.decoder(h_n.squeeze(0))
        return out.view(-1, self.output_frames, 3)


class GestureConditionedTrajectoryGRU(nn.Module):

    def __init__(self, input_dim=6, hidden_size=HIDDEN_SIZE, num_gestures=NUM_GESTURES,
                 output_frames=OUTPUT_FRAMES):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size + num_gestures, output_frames * 3)
        self.output_frames = output_frames

    def forward(self, x, gesture_onehot):
        _, h_n = self.gru(x)
        h = torch.cat([h_n.squeeze(0), gesture_onehot], dim=-1)
        out = self.decoder(h)
        return out.view(-1, self.output_frames, 3)


def jerk_penalty(pred_pos, last_pos):
    full = torch.cat([last_pos.unsqueeze(1), pred_pos], dim=1)
    second_diff = full[:, 2:, :] - 2 * full[:, 1:-1, :] + full[:, :-2, :]
    return (second_diff ** 2).sum(dim=-1).mean()


def train_model(model_cls, use_gesture, tag):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse = nn.MSELoss()

    train_x_t = torch.from_numpy(train_x)
    train_y_delta_t = torch.from_numpy(train_y_delta)
    train_last_pos_t = torch.from_numpy(train_last_pos)
    train_gesture_t = torch.from_numpy(train_gesture)
    num_train = train_x_t.shape[0]

    print(f"\n=== Training {tag} ===")
    for epoch in range(NUM_EPOCHS):
        model.train()
        permutation = torch.randperm(num_train)
        epoch_loss, num_batches = 0.0, 0
        for start in range(0, num_train, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            batch_x = train_x_t[idx].to(device)
            batch_y_delta = train_y_delta_t[idx].to(device)
            batch_last_pos = train_last_pos_t[idx].to(device)
            batch_gesture = train_gesture_t[idx].to(device) if use_gesture else None

            optimizer.zero_grad()
            pred_delta = model(batch_x, batch_gesture)
            pred_pos = pred_delta + batch_last_pos.unsqueeze(1)
            true_pos = batch_y_delta + batch_last_pos.unsqueeze(1)

            loss = mse(pred_pos, true_pos) + LAMBDA_SMOOTH * jerk_penalty(pred_pos, batch_last_pos)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: loss = {epoch_loss / num_batches:.8f}")

    return model


model_unconditioned = train_model(TrajectoryGRU, use_gesture=False, tag="unconditioned (Day46 recipe)")
model_conditioned = train_model(GestureConditionedTrajectoryGRU, use_gesture=True,
                                 tag="gesture-conditioned (oracle gesture label)")

# ----------------------------------------
# Evaluate both on the identical held-out, gesture-filtered windows.
# ----------------------------------------

horizon_frame_indices = [int(round(s * FPS)) - 1 for s in HORIZON_CHECKPOINTS_S]


def evaluate(model, use_gesture):
    model.eval()
    test_x_t = torch.from_numpy(test_x).to(device)
    test_gesture_t = torch.from_numpy(test_gesture).to(device) if use_gesture else None
    with torch.no_grad():
        pred_delta = model(test_x_t, test_gesture_t).cpu().numpy()
    pred_xyz = pred_delta + test_last_pos[:, None, :]

    displacement_error = np.linalg.norm(pred_xyz - test_y, axis=2)
    step_size = np.linalg.norm(np.diff(pred_xyz, axis=1), axis=2)

    net_disp = np.linalg.norm(pred_xyz[:, -1, :] - test_last_pos, axis=1)
    full_path = np.concatenate([test_last_pos[:, None, :], pred_xyz], axis=1)
    total_path = np.linalg.norm(np.diff(full_path, axis=1), axis=2).sum(axis=1)
    path_efficiency = np.where(total_path > 0, net_disp / total_path, 1.0)

    return {
        "mean_error_full_horizon_m": float(displacement_error.mean()),
        "mean_error_by_checkpoint_m": {
            str(s): float(displacement_error[:, idx].mean())
            for s, idx in zip(HORIZON_CHECKPOINTS_S, horizon_frame_indices)
        },
        "mean_step_size_m": float(step_size.mean()),
        "mean_path_efficiency": float(path_efficiency.mean()),
        "median_path_efficiency": float(np.median(path_efficiency)),
    }


results = {
    "held_out_subject": HELD_OUT_SUBJECT,
    "test_trials": test_names,
    "num_test_windows": len(test_windows),
    "gesture_vocabulary": gesture_id_list,
}

results["unconditioned"] = evaluate(model_unconditioned, use_gesture=False)
results["gesture_conditioned"] = evaluate(model_conditioned, use_gesture=True)

true_step_size = np.linalg.norm(np.diff(test_y, axis=1), axis=2)
net_disp_true = np.linalg.norm(test_y[:, -1, :] - test_last_pos, axis=1)
full_path_true = np.concatenate([test_last_pos[:, None, :], test_y], axis=1)
total_path_true = np.linalg.norm(np.diff(full_path_true, axis=1), axis=2).sum(axis=1)
path_efficiency_true = np.where(total_path_true > 0, net_disp_true / total_path_true, 1.0)

results["ground_truth"] = {
    "mean_step_size_m": float(true_step_size.mean()),
    "mean_path_efficiency": float(path_efficiency_true.mean()),
    "median_path_efficiency": float(np.median(path_efficiency_true)),
}

print("\n" + "=" * 70)
print("SUMMARY (held-out subject, gesture-filtered windows)")
print("=" * 70)
for name in ["unconditioned", "gesture_conditioned"]:
    r = results[name]
    print(f"\n{name}:")
    print(f"  Mean error (full 1s horizon): {r['mean_error_full_horizon_m']*1000:.2f} mm")
    for s in HORIZON_CHECKPOINTS_S:
        print(f"  at +{s:.1f}s: {r['mean_error_by_checkpoint_m'][str(s)]*1000:.2f} mm")
    print(f"  Step size: {r['mean_step_size_m']*1000:.3f} mm")
    print(f"  Path efficiency: mean={r['mean_path_efficiency']:.3f} "
          f"median={r['median_path_efficiency']:.3f}")

print(f"\nGround truth: step size={results['ground_truth']['mean_step_size_m']*1000:.3f}mm, "
      f"path efficiency mean={results['ground_truth']['mean_path_efficiency']:.3f} "
      f"median={results['ground_truth']['median_path_efficiency']:.3f}")

output_dir = Path(__file__).parent
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {output_dir / 'results.json'}")

torch.save(model_unconditioned.state_dict(), output_dir / "model_unconditioned.pt")
torch.save(model_conditioned.state_dict(), output_dir / "model_gesture_conditioned.pt")
np.savez(output_dir / "normalization_stats.npz", mean=feature_mean, std=feature_std)
with open(output_dir / "gesture_vocabulary.json", "w") as f:
    json.dump(gesture_id_list, f)
