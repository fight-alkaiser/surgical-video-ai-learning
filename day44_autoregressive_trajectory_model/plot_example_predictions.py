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

SLAVE_RIGHT_XYZ_COLS = [57, 58, 59]
SLAVE_RIGHT_VEL_COLS = [69, 70, 71]
SLAVE_RIGHT_GRIPPER_COL = 75

script_dir = Path(__file__).parent
with open(script_dir / "results.json") as f:
    results = json.load(f)
test_trials = results["test_trials"]

stats = np.load(script_dir / "normalization_stats.npz")
feature_mean, feature_std = stats["mean"], stats["std"]


class AutoregressiveTrajectoryGRU(nn.Module):
    def __init__(self, input_dim=6, hidden_size=HIDDEN_SIZE, output_frames=OUTPUT_FRAMES):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.decoder_cell = nn.GRUCell(3, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 3)
        self.output_frames = output_frames

    def forward(self, x, teacher_deltas=None):
        _, h_enc = self.encoder(x)
        h = h_enc.squeeze(0)
        batch_size = x.shape[0]
        prev_delta = torch.zeros(batch_size, 3, device=x.device)
        predicted_deltas = []
        for step in range(self.output_frames):
            h = self.decoder_cell(prev_delta, h)
            delta_pred = self.output_layer(h)
            predicted_deltas.append(delta_pred)
            prev_delta = delta_pred if teacher_deltas is None else teacher_deltas[:, step, :]
        return torch.stack(predicted_deltas, dim=1)


model = AutoregressiveTrajectoryGRU()
model.load_state_dict(torch.load(script_dir / "autoregressive_gru_model.pt", map_location="cpu"))
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
    t = INPUT_FRAMES + 60

    input_raw = signal[t - INPUT_FRAMES:t, 0:6]
    last_pos = signal[t - 1, 0:3]
    true_xyz = signal[t:t + OUTPUT_FRAMES, 0:3]

    x_norm = (input_raw - feature_mean) / feature_std
    with torch.no_grad():
        pred_deltas = model(torch.from_numpy(x_norm[None].astype(np.float32)),
                             teacher_deltas=None).numpy()[0]
    ar_pred_xyz = last_pos + np.cumsum(pred_deltas, axis=0)

    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    input_mm = input_raw[:, 0:3] * 1000
    true_mm = true_xyz * 1000
    ar_mm = ar_pred_xyz * 1000

    ax.plot(*input_mm.T, color="gray", linewidth=1.5, label="input (past 1s)")
    ax.plot(*np.vstack([input_mm[-1], true_mm]).T, color="black",
            linewidth=2.5, label="ground truth (future 1s)")
    ax.plot(*np.vstack([input_mm[-1], ar_mm]).T, color="tab:purple",
            linewidth=1.8, linestyle="-.", label="pred: autoregressive GRU")
    ax.scatter(*input_mm[-1], color="red", s=40, zorder=5, label="t (prediction start)")

    ax.set_title(f"{trial_name}, t={t} (held-out subject)", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    if i == 0:
        ax.legend(fontsize=7, loc="upper left")

plt.suptitle(
    "Day44: autoregressive GRU, free-running rollout -- prediction uses only frames "
    "before t (red dot); black = actual future, purple = model's own free-running prediction",
    fontsize=10,
)
plt.tight_layout()

output_path = script_dir / "example_predictions.png"
plt.savefig(output_path, dpi=150)
print(f"Saved {output_path}")
