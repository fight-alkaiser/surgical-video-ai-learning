import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------
# Visualizes the example windows saved by trajectory_forecasting_baseline.py
# as 3D trajectory plots. Per the anti-fabrication rule from Day41:
# input context, ground truth, and each baseline's prediction are
# drawn with distinct colors/line styles, all shown together, so a
# reader can see exactly what was predicted vs. what actually
# happened -- no path is ambiguous about which one it is.
# ----------------------------------------

results_path = Path(__file__).parent / "results.json"
with open(results_path) as f:
    results = json.load(f)

examples = results["example_windows"]

fig = plt.figure(figsize=(16, 8))

for i, example in enumerate(examples):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")

    input_xyz = np.array(example["input_xyz"]) * 1000  # meters -> mm
    true_xyz = np.array(example["true_future_xyz"]) * 1000
    held_xyz = np.array(example["last_position_held_pred_xyz"]) * 1000
    velocity_xyz = np.array(example["constant_velocity_pred_xyz"]) * 1000

    # Connect input's last point to each future path for visual continuity.
    ax.plot(*input_xyz.T, color="gray", linewidth=1.5, label="input (past 1s)")
    ax.plot(*np.vstack([input_xyz[-1], true_xyz]).T, color="black",
            linewidth=2.5, label="ground truth (future 1s)")
    ax.plot(*np.vstack([input_xyz[-1], held_xyz]).T, color="tab:blue",
            linewidth=1.5, linestyle="--", label="pred: last-position-held")
    ax.plot(*np.vstack([input_xyz[-1], velocity_xyz]).T, color="tab:orange",
            linewidth=1.5, linestyle=":", label="pred: constant-velocity")

    ax.scatter(*input_xyz[-1], color="red", s=40, zorder=5, label="t (prediction start)")

    ax.set_title(f"{example['trial']}, t={example['t']}", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    if i == 0:
        ax.legend(fontsize=7, loc="upper left")

plt.suptitle(
    "Day42: slave-right tooltip trajectory -- predictions use only frames "
    "before t (red dot); black = actual future, dashed/dotted = baseline predictions",
    fontsize=10,
)
plt.tight_layout()

output_path = Path(__file__).parent / "example_trajectories.png"
plt.savefig(output_path, dpi=150)
print(f"Saved {output_path}")
