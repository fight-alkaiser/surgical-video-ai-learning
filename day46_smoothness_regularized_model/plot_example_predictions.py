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

INPUT_FRAMES = 30
OUTPUT_FRAMES = 30
HIDDEN_SIZE = 64
PLOT_LAMBDA = 50.0

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


model_smooth = TrajectoryGRU()
model_smooth.load_state_dict(
    torch.load(script_dir / f"model_lambda_{PLOT_LAMBDA}.pt", map_location="cpu"))
model_smooth.eval()

model_jagged = TrajectoryGRU()
model_jagged.load_state_dict(
    torch.load(script_dir.parent / "day43_gru_trajectory_model" / "gru_trajectory_model.pt",
               map_location="cpu"))
model_jagged.eval()


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
    t = INPUT_FRAMES + 60

    input_raw = signal[t - INPUT_FRAMES:t, 0:6]
    last_pos = signal[t - 1, 0:3]
    true_xyz = signal[t:t + OUTPUT_FRAMES, 0:3]

    x_norm = (input_raw - feature_mean) / feature_std
    x_t = torch.from_numpy(x_norm[None].astype(np.float32))
    with torch.no_grad():
        delta_smooth = model_smooth(x_t).numpy()[0]
        delta_jagged = model_jagged(x_t).numpy()[0]
    smooth_xyz = delta_smooth + last_pos
    jagged_xyz = delta_jagged + last_pos

    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    input_mm = input_raw[:, 0:3] * 1000
    true_mm = true_xyz * 1000
    smooth_mm = smooth_xyz * 1000
    jagged_mm = jagged_xyz * 1000

    ax.plot(*input_mm.T, color="gray", linewidth=1.5, label="input (past 1s)")
    ax.plot(*np.vstack([input_mm[-1], true_mm]).T, color="black",
            linewidth=2.5, label="ground truth (future 1s)")
    ax.plot(*np.vstack([input_mm[-1], jagged_mm]).T, color="tab:green",
            linewidth=1.2, linestyle=":", alpha=0.7,
            label="pred: Day43 (lambda=0, jagged)")
    ax.plot(*np.vstack([input_mm[-1], smooth_mm]).T, color="tab:blue",
            linewidth=2.0, linestyle="-.", label=f"pred: Day46 (lambda={PLOT_LAMBDA})")
    ax.scatter(*input_mm[-1], color="red", s=40, zorder=5, label="t (prediction start)")

    ax.set_title(f"{trial_name}, t={t} (held-out subject)", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    if i == 0:
        ax.legend(fontsize=6.5, loc="upper left")

plt.suptitle(
    f"Day46: smoothness-regularized single-shot GRU (lambda={PLOT_LAMBDA}) vs. Day43's "
    "unregularized version -- prediction uses only frames before t (red dot)",
    fontsize=10,
)
plt.tight_layout()

output_path = script_dir / "example_predictions.png"
plt.savefig(output_path, dpi=150)
print(f"Saved {output_path}")
