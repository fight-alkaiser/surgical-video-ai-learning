"""Day102: a real, per-frame instrument-region detector, replacing
data/motion_weight_map.npy's single (64, 64) map computed once across the
whole dataset (see train.py's --weighted-loss). That map is a fixed prior --
it can't tell where the instrument is in any specific frame, only where it
tended to be on average.

No pretrained detector transfers to this domain: this is a dVRK
peg-transfer dry-lab view (pegboard, two black instrument shafts, teal
drape), not a laparoscopic surgery scene, so surgical-tool detectors trained
on EndoVis/Cholec-style data don't apply, and generic object detectors
(COCO, etc.) have no "robot instrument" class at all.

Plain color thresholding (the instrument shafts are near-black) doesn't
work either: at this resolution the frame's corners are also near-black
from lens vignetting, and HSV saturation for very dark pixels is an
unreliable discriminator (small RGB noise on a near-zero-value pixel
produces large, meaningless swings in hue/saturation) -- so corner vignette
and true instrument pixels look almost identical in raw HSV terms.

What does separate them is time: the vignette is a static property of the
camera, but the instrument moves. Per-episode background subtraction (median
brightness per pixel across the episode, as a background estimate; a
pixel darker than its own median by more than a threshold is foreground)
isolates the instrument specifically, because it only reacts to real
change, not to whichever pixels happen to be dark in every frame.
"""

import matplotlib.colors as mcolors
import numpy as np


def instrument_mask(frames: np.ndarray, darkness_margin: float = 0.12) -> np.ndarray:
    """frames: (T, H, W, 3) uint8, one episode. Returns (T, H, W) bool mask,
    True where that frame is significantly darker than its own per-pixel
    median across the episode -- i.e. where the (dark) instrument currently
    is, not wherever happens to be dark in every frame (background/vignette)."""
    v = mcolors.rgb_to_hsv(frames.astype(np.float32) / 255.0)[..., 2]
    baseline = np.median(v, axis=0)
    return (baseline - v) > darkness_margin
