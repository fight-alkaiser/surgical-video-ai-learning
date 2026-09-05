import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

JIGSAWS_ROOT = Path("/Users/katsutoshimakino/Datasets/JIGSAWS")
TASK = "Suturing"
KINEMATICS_DIR = JIGSAWS_ROOT / TASK / "kinematics" / "AllGestures"
TRANSCRIPTIONS_DIR = JIGSAWS_ROOT / TASK / "transcriptions"

INPUT_FRAMES = 30
OUTPUT_FRAMES = 30
HIDDEN_SIZE = 64

SLAVE_RIGHT_XYZ_COLS = [57, 58, 59]
SLAVE_RIGHT_VEL_COLS = [69, 70, 71]
SLAVE_RIGHT_GRIPPER_COL = 75

script_dir = Path(__file__).parent
with open(script_dir / "results.json") as f:
    results = json.load(f)
test_trials = results["test_trials"]

with open(script_dir / "gesture_vocabulary.json") as f:
    gesture_id_list = json.load(f)
gesture_to_index = {g: i for i, g in enumerate(gesture_id_list)}
NUM_GESTURES = len(gesture_id_list)

stats = np.load(script_dir / "normalization_stats.npz")
feature_mean, feature_std = stats["mean"], stats["std"]


class TrajectoryGRU(nn.Module):
    def __init__(self, input_dim=6, hidden_size=HIDDEN_SIZE, output_frames=OUTPUT_FRAMES):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_frames * 3)
        self.output_frames = output_frames

    def forward(self, x, gesture_onehot=None):
        _, h_n = self.gru(x)
        return self.decoder(h_n.squeeze(0)).view(-1, self.output_frames, 3)


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
        return self.decoder(h).view(-1, self.output_frames, 3)


model_uncond = TrajectoryGRU()
model_uncond.load_state_dict(torch.load(script_dir / "model_unconditioned.pt", map_location="cpu"))
model_uncond.eval()

model_cond = GestureConditionedTrajectoryGRU()
model_cond.load_state_dict(
    torch.load(script_dir / "model_gesture_conditioned.pt", map_location="cpu"))
model_cond.eval()


def load_trial_signal(trial_name):
    path = KINEMATICS_DIR / f"{trial_name}.txt"
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 76:
            continue
        values = [float(parts[i]) for i in
                  SLAVE_RIGHT_XYZ_COLS + SLAVE_RIGHT_VEL_COLS + [SLAVE_RIGHT_GRIPPER_COL]]
        rows.append(values)
    return np.array(rows)


def load_trial_gesture_labels(trial_name, num_frames):
    path = TRANSCRIPTIONS_DIR / f"{trial_name}.txt"
    labels = np.full(num_frames, -1, dtype=np.int64)
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        start, end, gesture = int(parts[0]), int(parts[1]), int(parts[2].lstrip("G"))
        end = min(end, num_frames - 1)
        if start <= end:
            labels[start:end + 1] = gesture
    return labels


fig = plt.figure(figsize=(16, 8))
panel = 0

for trial_name in test_trials:
    if panel >= 4:
        break
    signal = load_trial_signal(trial_name)
    gestures = load_trial_gesture_labels(trial_name, len(signal))

    t = INPUT_FRAMES
    found = False
    while t + OUTPUT_FRAMES <= len(signal):
        span = gestures[t - INPUT_FRAMES:t + OUTPUT_FRAMES]
        if len(np.unique(span)) == 1 and span[0] != -1:
            found = True
            break
        t += 30
    if not found:
        continue

    gesture_id = int(gestures[t])
    input_raw = signal[t - INPUT_FRAMES:t, 0:6]
    last_pos = signal[t - 1, 0:3]
    true_xyz = signal[t:t + OUTPUT_FRAMES, 0:3]

    x_norm = (input_raw - feature_mean) / feature_std
    x_t = torch.from_numpy(x_norm[None].astype(np.float32))
    onehot = np.zeros((1, NUM_GESTURES), dtype=np.float32)
    onehot[0, gesture_to_index[gesture_id]] = 1.0
    onehot_t = torch.from_numpy(onehot)

    with torch.no_grad():
        delta_uncond = model_uncond(x_t).numpy()[0]
        delta_cond = model_cond(x_t, onehot_t).numpy()[0]
    uncond_xyz = delta_uncond + last_pos
    cond_xyz = delta_cond + last_pos

    ax = fig.add_subplot(2, 2, panel + 1, projection="3d")
    input_mm = input_raw[:, 0:3] * 1000
    true_mm = true_xyz * 1000
    uncond_mm = uncond_xyz * 1000
    cond_mm = cond_xyz * 1000

    ax.plot(*input_mm.T, color="gray", linewidth=1.5, label="input (past 1s)")
    ax.plot(*np.vstack([input_mm[-1], true_mm]).T, color="black",
            linewidth=2.5, label="ground truth (future 1s)")
    ax.plot(*np.vstack([input_mm[-1], uncond_mm]).T, color="tab:blue",
            linewidth=1.5, linestyle="-.", label="pred: unconditioned (Day46)")
    ax.plot(*np.vstack([input_mm[-1], cond_mm]).T, color="tab:orange",
            linewidth=1.8, linestyle="--", label="pred: gesture-conditioned")
    ax.scatter(*input_mm[-1], color="red", s=40, zorder=5, label="t (prediction start)")

    ax.set_title(f"{trial_name}, t={t}, gesture=G{gesture_id}", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    if panel == 0:
        ax.legend(fontsize=6.5, loc="upper left")
    panel += 1

plt.suptitle(
    "Day47: gesture-conditioned (oracle gesture) vs. unconditioned prediction, "
    "held-out subject -- prediction uses only frames before t (red dot)",
    fontsize=10,
)
plt.tight_layout()

output_path = script_dir / "example_predictions.png"
plt.savefig(output_path, dpi=150)
print(f"Saved {output_path}")
