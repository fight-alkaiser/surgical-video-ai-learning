"""Day102: visual sanity check for instrument_detector.py -- overlays the
detected mask (red) on sampled frames from several episodes and saves a grid,
so the detector's quality can be judged by eye before wiring it into any
evaluation. Not a quantitative benchmark (no ground-truth instrument mask
exists for this dataset); this is deliberately a qualitative check only.
"""

import numpy as np
from PIL import Image

from instrument_detector import instrument_mask

EPISODES = ["episode_000000", "episode_000003", "episode_000007", "episode_000012"]
SAMPLES_PER_EPISODE = 5

rows = []
for ep in EPISODES:
    frames = np.load(f"data/episodes/{ep}_frames.npy")
    mask = instrument_mask(frames)
    T = len(frames)
    idxs = [int(T * f) for f in np.linspace(0.1, 0.9, SAMPLES_PER_EPISODE)]
    row = []
    for i in idxs:
        overlay = frames[i].copy()
        overlay[mask[i]] = [255, 0, 0]
        row.append(overlay)
    rows.append(np.concatenate(row, axis=1))

grid = np.concatenate(rows, axis=0)
out_path = "outputs/day102_instrument_detector_check.png"
Image.fromarray(grid).resize(
    (64 * SAMPLES_PER_EPISODE * 3, 64 * len(EPISODES) * 3), Image.NEAREST
).save(out_path)
print(f"saved {out_path}")
