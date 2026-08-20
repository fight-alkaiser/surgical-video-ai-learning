"""Day78 summary figure: paired_loss (real/shuffled/zero action) across 3 seeds,
source=noise, corrected evaluation. Reproducibility is the point of this figure --
compare to Day62's day62_jepa_action_sensitivity.png, which showed a single run.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

seeds = [0, 1, 2]
conditions = ["zero", "real", "shuffled"]
colors = {"zero": "#9e9e9e", "real": "#2e7d32", "shuffled": "#c62828"}

data = {}
for seed in seeds:
    with open(f"outputs/history_cfm_h10_noise_seed{seed}.json") as f:
        h = json.load(f)
    data[seed] = {c: h["eval"][c]["paired_loss"] for c in conditions}

fig, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(seeds))
width = 0.25

for i, cond in enumerate(conditions):
    values = [data[s][cond] for s in seeds]
    offset = (i - 1) * width
    ax.bar(x + offset, values, width, label=cond, color=colors[cond])

ax.set_xticks(x)
ax.set_xticklabels([f"seed {s}" for s in seeds])
ax.set_ylabel("paired_loss (CFM velocity loss vs. real transition, lower = better)")
ax.set_title("Day78: action pathway is directionally correct, but net harmful\n(zero < real < shuffled, reproducible across 3 seeds)")
ax.legend(title="action given to predictor")
ax.set_ylim(0, max(max(d.values()) for d in data.values()) * 1.2)

plt.tight_layout()
plt.savefig("outputs/day78_cfm_paired_loss_summary.png", dpi=150)
print("saved outputs/day78_cfm_paired_loss_summary.png")
