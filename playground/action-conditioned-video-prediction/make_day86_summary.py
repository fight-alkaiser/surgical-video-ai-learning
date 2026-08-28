"""Day86 summary figure: does encoding the action window with a small GRU
(instead of flattening it into one vector) change the Day83-84 finding that
zero action ties or beats real action? Compares mean best_of_n_error
(averaged over 3 seeds) for real/zero/shuffled action, flatten vs. sequence
encoding, both at n=200 episodes / 300 epochs.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

seeds = [0, 1, 2]
conditions = ["zero", "real", "shuffled"]
colors = {"zero": "#9e9e9e", "real": "#2e7d32", "shuffled": "#c62828"}


def mean_best_of_n(tag_fn):
    out = {}
    for cond in conditions:
        vals = []
        for seed in seeds:
            with open(f"outputs/history_cfm_{tag_fn(seed)}.json") as f:
                h = json.load(f)
            vals.append(h["eval"][cond]["best_of_n_error"])
        out[cond] = (np.mean(vals), np.std(vals))
    return out


results = {
    "flatten": mean_best_of_n(lambda s: f"h10_noise_n200_seed{s}"),
    "sequence (GRU)": mean_best_of_n(lambda s: f"h10_noise_n200_seed{s}_sequence"),
}

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(conditions))
width = 0.32
for i, mode in enumerate(["flatten", "sequence (GRU)"]):
    means = [results[mode][c][0] for c in conditions]
    stds = [results[mode][c][1] for c in conditions]
    bar_colors = [colors[c] for c in conditions]
    offset = (i - 0.5) * width
    ax.bar(x + offset, means, width, yerr=stds, capsize=3, color=bar_colors, alpha=(1.0 if mode == "flatten" else 0.55))

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel("mean best_of_n_error across 3 seeds (lower = better)")
ax.set_title("Day86: encoding the action window with a GRU instead of flattening\nsolid = flatten, pale = sequence (GRU) -- zero still ties/beats real either way")
plt.tight_layout()
plt.savefig("outputs/day86_action_encoder_summary.png", dpi=150, bbox_inches="tight")
print("saved outputs/day86_action_encoder_summary.png")

for mode, r in results.items():
    print(mode, {c: round(v[0], 5) for c, v in r.items()})
