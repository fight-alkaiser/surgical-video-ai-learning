import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

script_dir = Path(__file__).parent
with open(script_dir / "results.json") as f:
    results = json.load(f)

colors = {"N": "tab:blue", "I": "tab:orange", "E": "tab:green"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: deviation (self-proclaimed-expert reference) vs GRS
ax = axes[0]
tr = results["self_proclaimed_expert_reference"]["trial_results"]
for skill in ["N", "I", "E"]:
    xs = [v["grs"] for v in tr.values() if v["skill"] == skill and v["mean_deviation"] is not None]
    ys = [v["mean_deviation"] for v in tr.values() if v["skill"] == skill and v["mean_deviation"] is not None]
    ax.scatter(xs, ys, color=colors[skill], label=f"skill={skill}", alpha=0.8)
r1 = results["self_proclaimed_expert_reference"]["correlation_deviation_vs_grs"]
ax.set_title(f"Deviation (self-proclaimed-E reference)\nvs GRS, r={r1:.2f}")
ax.set_xlabel("GRS score")
ax.set_ylabel("Mean deviation score")
ax.legend()

# Panel 2: deviation (top-GRS reference) vs GRS
ax = axes[1]
tr2 = results["top_grs_reference"]["trial_results"]
for skill in ["N", "I", "E"]:
    xs = [v["grs"] for v in tr2.values() if v["skill"] == skill and v["mean_deviation"] is not None]
    ys = [v["mean_deviation"] for v in tr2.values() if v["skill"] == skill and v["mean_deviation"] is not None]
    ax.scatter(xs, ys, color=colors[skill], label=f"skill={skill}", alpha=0.8)
r2 = results["top_grs_reference"]["correlation_deviation_vs_grs"]
ax.set_title(f"Deviation (top-GRS reference)\nvs GRS, r={r2:.2f}")
ax.set_xlabel("GRS score")
ax.set_ylabel("Mean deviation score")
ax.legend()

# Panel 3: mean speed vs GRS (the diagnosed confound)
ax = axes[2]
JIGSAWS_ROOT = Path("/Users/katsutoshimakino/Datasets/JIGSAWS")
KIN_DIR = JIGSAWS_ROOT / "Suturing" / "kinematics" / "AllGestures"
META_PATH = JIGSAWS_ROOT / "Suturing" / "meta_file_Suturing.txt"
meta = {}
for line in META_PATH.read_text().splitlines():
    if not line.strip():
        continue
    parts = line.split()
    meta[parts[0]] = {"skill": parts[1], "grs": int(parts[2])}


def mean_speed(name):
    rows = []
    for line in (KIN_DIR / f"{name}.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) < 76:
            continue
        rows.append([float(parts[i]) for i in [69, 70, 71]])
    arr = np.array(rows)
    return float(np.linalg.norm(arr, axis=1).mean())


by_skill_speed = {"N": ([], []), "I": ([], []), "E": ([], [])}
for name, m in meta.items():
    path = KIN_DIR / f"{name}.txt"
    if not path.exists():
        continue
    speed = mean_speed(name)
    by_skill_speed[m["skill"]][0].append(m["grs"])
    by_skill_speed[m["skill"]][1].append(speed * 1000)  # m/s -> mm/s

all_grs, all_speed = [], []
for skill in ["N", "I", "E"]:
    xs, ys = by_skill_speed[skill]
    ax.scatter(xs, ys, color=colors[skill], label=f"skill={skill}", alpha=0.8)
    all_grs.extend(xs)
    all_speed.extend(ys)
r3 = float(np.corrcoef(all_grs, all_speed)[0, 1])
ax.set_title(f"Mean tooltip speed vs GRS\nr={r3:.2f}")
ax.set_xlabel("GRS score")
ax.set_ylabel("Mean speed (mm/s)")
ax.legend()

plt.tight_layout()
output_path = script_dir / "diagnosis_plots.png"
plt.savefig(output_path, dpi=150)
print(f"Saved {output_path}")
