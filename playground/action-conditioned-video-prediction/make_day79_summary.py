"""Day79 summary figure: does giving the action pathway an explicit gate
(GatedVelocityPredictor) change the Day78 finding? Two panels:
  left  -- paired_loss, ungated vs gated, both source=noise, 3 seeds
  right -- mean_gate value per condition (gated model only) -- does the gate
           actually open more for real/shuffled than for zero action?
"""

import json

import matplotlib.pyplot as plt
import numpy as np

seeds = [0, 1, 2]
conditions = ["zero", "real", "shuffled"]
colors = {"zero": "#9e9e9e", "real": "#2e7d32", "shuffled": "#c62828"}

ungated, gated = {}, {}
for seed in seeds:
    with open(f"outputs/history_cfm_h10_noise_seed{seed}.json") as f:
        h = json.load(f)
    ungated[seed] = {c: h["eval"][c]["paired_loss"] for c in conditions}
    with open(f"outputs/history_cfm_h10_noise_seed{seed}_gated.json") as f:
        h = json.load(f)
    gated[seed] = {c: h["eval"][c]["paired_loss"] for c in conditions}
    gated_gate = {c: h["eval"][c]["mean_gate"] for c in conditions}
    gated.setdefault("_gate", {})[seed] = gated_gate

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# left: paired_loss, ungated vs gated
x = np.arange(len(seeds))
width = 0.12
ax = axes[0]
for i, cond in enumerate(conditions):
    ax.bar(x + (i - 1) * width * 2 - width / 2, [ungated[s][cond] for s in seeds], width, color=colors[cond], alpha=0.45)
    ax.bar(x + (i - 1) * width * 2 + width / 2, [gated[s][cond] for s in seeds], width, color=colors[cond], alpha=1.0)
ax.set_xticks(x)
ax.set_xticklabels([f"seed {s}" for s in seeds])
ax.set_ylabel("paired_loss (lower = better)")
ax.set_title("ungated (pale) vs gated (solid)\nzero still wins in both")
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in conditions]
ax.legend(handles, conditions, title="action")

# right: mean_gate per condition
ax = axes[1]
for i, cond in enumerate(conditions):
    values = [gated["_gate"][s][cond] for s in seeds]
    ax.bar(x + (i - 1) * width * 2, values, width * 2, color=colors[cond], label=cond)
ax.set_xticks(x)
ax.set_xticklabels([f"seed {s}" for s in seeds])
ax.set_ylabel("mean gate value (0 = fully closed, 1 = fully open)")
ax.set_title("the gate DOES open more for real/shuffled\nthan for zero -- but it doesn't help")
ax.legend(title="action")

plt.suptitle("Day79: an explicit opt-out gate doesn't flip Day78's result", y=1.02)
plt.tight_layout()
plt.savefig("outputs/day79_gating_summary.png", dpi=150, bbox_inches="tight")
print("saved outputs/day79_gating_summary.png")
