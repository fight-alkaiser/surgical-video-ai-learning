"""Day80 summary figure: does expanding the dataset 5x (20 -> 100 episodes)
flip the Day78/79 finding that using the action beats ignoring it? Compares
mean paired_loss (averaged over 3 seeds) for real/zero/shuffled action,
at both dataset sizes, for both the plain and gated predictors.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

seeds = [0, 1, 2]
conditions = ["zero", "real", "shuffled"]
colors = {"zero": "#9e9e9e", "real": "#2e7d32", "shuffled": "#c62828"}


def mean_paired_loss(tag_fn):
    out = {}
    for cond in conditions:
        vals = []
        for seed in seeds:
            with open(f"outputs/history_cfm_{tag_fn(seed)}.json") as f:
                h = json.load(f)
            vals.append(h["eval"][cond]["paired_loss"])
        out[cond] = (np.mean(vals), np.std(vals))
    return out


results = {
    ("ungated", "n=20"): mean_paired_loss(lambda s: f"h10_noise_seed{s}"),
    ("ungated", "n=100"): mean_paired_loss(lambda s: f"h10_noise_n100_seed{s}"),
    ("gated", "n=20"): mean_paired_loss(lambda s: f"h10_noise_seed{s}_gated"),
    ("gated", "n=100"): mean_paired_loss(lambda s: f"h10_noise_n100_seed{s}_gated"),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

for ax, arch in zip(axes, ["ungated", "gated"]):
    x = np.arange(len(conditions))
    width = 0.32
    for i, n in enumerate(["n=20", "n=100"]):
        means = [results[(arch, n)][c][0] for c in conditions]
        stds = [results[(arch, n)][c][1] for c in conditions]
        bar_colors = [colors[c] for c in conditions]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3, color=bar_colors, alpha=(0.45 if n == "n=20" else 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("mean paired_loss across 3 seeds (lower = better)")
    ax.set_title(f"{arch} predictor\npale = 20 episodes, solid = 100 episodes")

plt.suptitle("Day80: 5x more data flips the result -- real action now beats zero, in both architectures", y=1.03)
plt.tight_layout()
plt.savefig("outputs/day80_data_scale_summary.png", dpi=150, bbox_inches="tight")
print("saved outputs/day80_data_scale_summary.png")

for (arch, n), r in results.items():
    print(arch, n, {c: round(v[0], 4) for c, v in r.items()})
