"""Day81 summary figure: does increasing ODE integration steps close the
Day80 paired_loss / best_of_n_error gap between real and zero action?
Line plot of best_of_n_error vs. step count, one panel per seed.
"""

import json

import matplotlib.pyplot as plt

seeds = [0, 1, 2]
conditions = ["zero", "real", "shuffled"]
colors = {"zero": "#9e9e9e", "real": "#2e7d32", "shuffled": "#c62828"}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

for ax, seed in zip(axes, seeds):
    with open(f"outputs/history_cfm_step_sweep_h10_noise_n100_seed{seed}.json") as f:
        h = json.load(f)
    steps = sorted(int(s) for s in h.keys())
    for cond in conditions:
        values = [h[str(s)][cond]["best_of_n_error"] for s in steps]
        ax.plot(steps, values, marker="o", color=colors[cond], label=cond)
    ax.set_xscale("log", base=2)
    ax.set_xticks(steps)
    ax.set_xticklabels(steps)
    ax.set_xlabel("Euler integration steps")
    ax.set_title(f"seed {seed}")
    if seed == 0:
        ax.set_ylabel("best_of_n_error (lower = better)")
    ax.legend(fontsize=8)

plt.suptitle("Day81: more ODE steps doesn't close the real-vs-zero gap (only seed 1 nearly converges)", y=1.04)
plt.tight_layout()
plt.savefig("outputs/day81_step_sweep_summary.png", dpi=150, bbox_inches="tight")
print("saved outputs/day81_step_sweep_summary.png")
