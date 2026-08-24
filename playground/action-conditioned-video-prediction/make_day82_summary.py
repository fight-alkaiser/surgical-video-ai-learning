"""Day82 summary figure: (left) does more samples close the real/zero gap
(a la Day81's step sweep, but for sample count)? (right) bias/variance
decomposition -- is the gap about the cluster's spread, or where it's centered?
"""

import json

import matplotlib.pyplot as plt
import numpy as np

seeds = [0, 1, 2]
conditions = ["zero", "real", "shuffled"]
colors = {"zero": "#9e9e9e", "real": "#2e7d32", "shuffled": "#c62828"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# left: best-of-N vs N, averaged across seeds
ax = axes[0]
all_data = []
for seed in seeds:
    with open(f"outputs/history_cfm_distribution_h10_noise_n100_seed{seed}.json") as f:
        all_data.append(json.load(f))
n_list = sorted(int(n) for n in all_data[0]["best_of_n"].keys())
for cond in conditions:
    means = [np.mean([d["best_of_n"][str(n)][cond] for d in all_data]) for n in n_list]
    ax.plot(n_list, means, marker="o", color=colors[cond], label=cond)
ax.set_xscale("log", base=2)
ax.set_xticks(n_list)
ax.set_xticklabels(n_list)
ax.set_xlabel("N (samples drawn, best-of-N)")
ax.set_ylabel("best_of_n_error, mean over 3 seeds (lower = better)")
ax.set_title("More samples doesn't close the gap either\n(N=8 -> N=256, 32x more draws)")
ax.legend()

# right: bias/variance decomposition, mean over 3 seeds
ax = axes[1]
x = np.arange(len(conditions))
width = 0.35
bias_means = [np.mean([d["bias_variance"][c]["bias_sq"] for d in all_data]) for c in conditions]
var_means = [np.mean([d["bias_variance"][c]["variance"] for d in all_data]) for c in conditions]
ax.bar(x, bias_means, width, label="bias$^2$ (cluster center vs. true target)", color=[colors[c] for c in conditions])
ax.bar(x, var_means, width, bottom=bias_means, label="variance (spread within cluster)", color=[colors[c] for c in conditions], alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel("mean squared distance (normalized latent space)")
ax.set_title("The gap is almost entirely bias, not variance\n(spread is nearly identical across conditions)")
ax.legend(fontsize=8)

plt.suptitle("Day82: real action's samples are consistently biased away from the target, not just more spread out", y=1.03)
plt.tight_layout()
plt.savefig("outputs/day82_distribution_summary.png", dpi=150, bbox_inches="tight")
print("saved outputs/day82_distribution_summary.png")
