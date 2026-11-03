"""Day153 figure: what a "consecutive" vs a "shuffled" val batch actually looks
like on this project's frames, next to the SIGReg score each gives the same
checkpoints (numbers read from outputs/day153_val_batch_check.txt).
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.argv = [sys.argv[0], "--anti-collapse", "sigreg"]
src = open("ijepa_train.py").read()
exec(src[: src.index("model = IJEPAModel")])  # same val split as training

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (dataviz default palette)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"

# one real consecutive val batch (the first one the training loop scores) and one
# shuffled batch; show every 8th frame of each so 8 thumbnails stand for 64 frames
consec_idx = np.arange(0, 64)[::8]
shuf_idx = np.random.default_rng(0).permutation(len(val_frames))[:64][::8]

rows = {}
for line in open("outputs/day153_val_batch_check.txt").read().splitlines()[1:]:
    parts = line.split()
    rows[(parts[0], int(parts[2]))] = (float(parts[3]), float(parts[4]))
epochs = [0, 10, 20, 29]
consec = [rows[("sigreg-image", e)][0] for e in epochs]
shuf = [rows[("sigreg-image", e)][1] for e in epochs]

fig = plt.figure(figsize=(13, 4.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.25, wspace=0.12)

for r, (idx, title, color) in enumerate([
    (consec_idx, "A. What the val batch was: 64 consecutive frames of one episode", BLUE),
    (shuf_idx, "B. Shuffled: 64 frames drawn across the 4 val episodes", ORANGE),
]):
    sub = gs[r, 0].subgridspec(1, 8, wspace=0.05)
    for k, i in enumerate(idx):
        ax = fig.add_subplot(sub[0, k])
        ax.imshow(val_frames[i], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(color); s.set_linewidth(2)
        if k == 0:
            ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8, x=0)

ax = fig.add_subplot(gs[:, 1])
ax.plot(epochs, consec, color=BLUE, lw=2, marker="o", ms=8, label="consecutive batches (A)")
ax.plot(epochs, shuf, color=ORANGE, lw=2, marker="o", ms=8, label="shuffled batches (B)")
ax.annotate("auto-selected:\nepoch 0 = collapsed", xy=(0, consec[0]), xytext=(2, 5),
            fontsize=10, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
ax.text(epochs[-1], consec[-1] + 2, f"{consec[-1]:.0f}", ha="center", fontsize=10, color=INK)
ax.text(epochs[-1], shuf[-1] - 4, f"{shuf[-1]:.0f}", ha="center", fontsize=10, color=INK)
ax.set_xticks(epochs)
ax.set_xlabel("training epoch (same saved checkpoints)", color=MUTED)
ax.set_ylabel("SIGReg on val (lower = more spread out, better)", color=MUTED)
ax.set_title("C. Same models, scored on A vs B", loc="left", fontsize=11, color=INK)
ax.set_ylim(0, 58)
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.legend(frameon=False, loc="upper left", fontsize=10)

fig.savefig("outputs/day153_consecutive_vs_shuffled.png", dpi=170, bbox_inches="tight", facecolor="white")
print("saved outputs/day153_consecutive_vs_shuffled.png")
