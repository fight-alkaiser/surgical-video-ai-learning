import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------
# Regenerates a handful of example held-out-subject predictions
# (using the saved model checkpoint + normalization stats, no
# retraining) for 3D visualization. Same anti-fabrication discipline
# as Day42: predictions use only frames before t, drawn in distinct
# colors/styles from the ground truth.
# ----------------------------------------

JIGSAWS_ROOT = Path("/Users/katsutoshimakino/Datasets/JIGSAWS")
TASK = "Suturing"
KINEMATICS_DIR = JIGSAWS_ROOT / TASK / "kinematics" / "AllGestures"

INPUT_FRAMES = 30
OUTPUT_FRAMES = 30
FPS = 30.0
DT = 1.0 / FPS
HIDDEN_SIZE = 64

SLAVE_RIGHT_XYZ_COLS = [57, 58, 59]
SLAVE_RIGHT_VEL_COLS = [69, 70, 71]
SLAVE_RIGHT_GRIPPER_COL = 75

script_dir = Path(__file__).parent
with open(script_dir / "results.json") as f:
    results = json.load(f)
test_trials = results["test_trials"]

stats = np.load(script_dir / "normalization_stats.npz")
feature_mean, feature_std = stats["mean"], stats["std"]


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


model = TrajectoryGRU()
model.load_state_dict(torch.load(script_dir / "gru_trajectory_model.pt", map_location="cpu"))
model.eval()


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


fig = plt.figure(figsize=(16, 8))

for i, trial_name in enumerate(test_trials[:4]):
    signal = load_trial_signal(trial_name)
    t = INPUT_FRAMES + 60  # a representative window, well inside the trial

    input_raw = signal[t - INPUT_FRAMES:t, 0:6]
    last_pos = signal[t - 1, 0:3]
    last_vel = signal[t - 1, 3:6]
    true_xyz = signal[t:t + OUTPUT_FRAMES, 0:3]

    x_norm = (input_raw - feature_mean) / feature_std
    with torch.no_grad():
        pred_delta = model(torch.from_numpy(x_norm[None].astype(np.float32))).numpy()[0]
    gru_pred_xyz = pred_delta + last_pos

    held_pred_xyz = np.tile(last_pos, (OUTPUT_FRAMES, 1))
    steps = (np.arange(1, OUTPUT_FRAMES + 1) * DT).reshape(-1, 1)
    velocity_pred_xyz = last_pos + last_vel * steps

    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    input_mm = input_raw[:, 0:3] * 1000
    true_mm = true_xyz * 1000
    held_mm = held_pred_xyz * 1000
    velocity_mm = velocity_pred_xyz * 1000
    gru_mm = gru_pred_xyz * 1000

    ax.plot(*input_mm.T, color="gray", linewidth=1.5, label="input (past 1s)")
    ax.plot(*np.vstack([input_mm[-1], true_mm]).T, color="black",
            linewidth=2.5, label="ground truth (future 1s)")
    ax.plot(*np.vstack([input_mm[-1], held_mm]).T, color="tab:blue",
            linewidth=1.3, linestyle="--", label="pred: last-position-held")
    ax.plot(*np.vstack([input_mm[-1], velocity_mm]).T, color="tab:orange",
            linewidth=1.3, linestyle=":", label="pred: constant-velocity")
    ax.plot(*np.vstack([input_mm[-1], gru_mm]).T, color="tab:green",
            linewidth=1.8, linestyle="-.", label="pred: GRU model")
    ax.scatter(*input_mm[-1], color="red", s=40, zorder=5, label="t (prediction start)")

    ax.set_title(f"{trial_name}, t={t} (held-out subject)", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    if i == 0:
        ax.legend(fontsize=6.5, loc="upper left")

plt.suptitle(
    "Day43: slave-right tooltip trajectory, held-out subject B -- predictions use only "
    "frames before t (red dot); black = actual future, colored lines = 3 prediction methods",
    fontsize=10,
)
plt.tight_layout()

output_path = script_dir / "example_predictions.png"
plt.savefig(output_path, dpi=150)
print(f"Saved {output_path}")
